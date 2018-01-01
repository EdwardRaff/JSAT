/*
 * Copyright (C) 2018 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package jsat.clustering;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.DoubleStream;
import jsat.DataSet;
import static jsat.clustering.SeedSelectionMethods.selectIntialPoints;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.utils.IntSet;
import jsat.utils.concurrent.ParallelUtils;

/**
 * This class implements the TRIKMEDS algorithm for PAM clustering. It returns
 * the exact same result that would have been computed by PAM, but uses the
 * triangle inequality to avoid unnecessary distance calculations. Expected
 * runtime is O( n sqrt(n)), but still has worst case complexity
 * O(n<sup>2</sup>). It also requires that the distance metric used be a valid
 * distance metric.
 *
 * @author Edward Raff
 */
public class TRIKMEDS extends PAM
{

    public TRIKMEDS(DistanceMetric dm, Random rand, SeedSelectionMethods.SeedSelection seedSelection)
    {
        super(dm, rand, seedSelection);
    }

    public TRIKMEDS(DistanceMetric dm, Random rand)
    {
        super(dm, rand);
    }

    public TRIKMEDS(DistanceMetric dm)
    {
        super(dm);
    }

    public TRIKMEDS()
    {
        super();
    }

    @Override
    public void setDistanceMetric(DistanceMetric dm)
    {
        if(!dm.isValidMetric())
            throw new IllegalArgumentException("TRIKMEDS requires a valid distance metric, but " + dm.toString() + " does not obey all distance metric properties");
        super.setDistanceMetric(dm); 
    }

    @Override
    protected double cluster(DataSet data, boolean doInit, int[] medioids, int[] assignments, List<Double> cacheAccel, boolean parallel)
    {
        LongAdder changes = new LongAdder();
        Arrays.fill(assignments, -1);//-1, invalid category!
        
        List<Vec> X = data.getDataVectors();
        final List<Double> accel;
        
        if(doInit)
        {
            TrainableDistanceMetric.trainIfNeeded(dm, data);
            accel = dm.getAccelerationCache(X);
            selectIntialPoints(data, medioids, dm, accel, rand, seedSelection);
        }
        else
            accel = cacheAccel;
        
        //N : number of training samples
        final int N = data.getSampleSize();
        final int K = medioids.length;
        //medioids = m
        //m(k) : index of current medoid of cluster k,m(k) ∈ {1, . . . ,N}
        final int[] m = medioids;
        //c(k) : current medoid of cluster k, that is c(k) = x(m(k))
        final int[] c = new int[K];
        //n_1(i) : cluster index of centroid nearest to x(i)
        
        //a(i) : cluster to which x(i) is currently assigned
        final int[] a = assignments;
        
        //d(i) : distance from x(i) to c(a(i))
        final double[] d = new double[N];
        final double[] d_tilde = new double[N];
        //v(k) : number of samples assigned to cluster k
        final double[] v = new double[K];
        //V (k) : number of samples assigned to a cluster of index less than k +1
        //We don't use V in this implementation. Paper uses it as a weird way of simplifying algorithm description. But not needed
        
        //lc(i, k) : lowerbound on distance from x(i) tom(k)
        double[][] lc = new double[N][K];
        
        //ls(i) : lowerbound on
        double[] ls = new double[N];
        
        //p(k) : distance moved (teleported) by m(k) in last update
        double[] p = new double[K];
        
        //s(k) : sum of distances of samples in cluster k to medoid k
        double[] s = new double[K];
        

        IntSet[] ownedBy = new IntSet[K];
        for(int i = 0; i < K; i++)
            ownedBy[i] = new IntSet();
        
        //Working sets used in updates
        double[] delta_n_in = new double[K];
        double[] delta_n_out = new double[K];
        double[] delta_s_in = new double[K];
        double[] delta_s_out = new double[K];
        
        // initialise //
        System.arraycopy(m, 0, c, 0, K);
        ParallelUtils.run(parallel, N, (start, end) -> 
        {
            for(int i = start; i < end; i++)
            {
                double a_min_val = Double.POSITIVE_INFINITY;
                int a_min_k = 0;
                for(int k = 0; k < K; k++)
                {
                    //Tightly initialise lower bounds on data-to-medoid distances
                    lc[i][k] = dm.dist(i, m[k], X, accel);
                    if(lc[i][k] <= a_min_val)
                    {
                        a_min_val = lc[i][k];
                        a_min_k = k;
                    }
                }
                //Set assignments and distances to nearest (assigned) medoid
                a[i] = a_min_k;
                d[i] = a_min_val;
                //Update cluster count
                v[a[i]]++;
                ownedBy[a_min_k].add(i);
                //Update sum of distances to medoid
                s[a[i]] += d[i];
                //Initialise lower bound on sum of in-cluster distances to x(i) to zero
                ls[i] = 0;
            }
        });
        
        for(int k = 0; k < K; k++)
            ls[m[k]] = s[k];
        //end initialization
        
        int iter = 0;
        do
        {
            changes.reset();
            
            ///// update-medoids() //////
            for(int k = 0; k < K; k++)
            {
                boolean medoid_k_changed = false;
                for(int i : ownedBy[k])
                {
                    // If the bound test cannot exclude i asm(k)
                    if(ls[i] < s[k])
                    {
                        // Make ls(i) tight by computing and cumulating all in-cluster distances to x(i),
                        ls[i] = 0;
                        for(int j : ownedBy[k])
                        {
                            d_tilde[j] = dm.dist(i, j, X, accel);
                            ls[i] += d_tilde[j];
                        }
                        //// Re-perform the test for i as candidate for m(k), now with exact sums. 
                        //If i is the new best candidate, update some cluster information
                        if(ls[i] < s[k])
                        {
                            s[k] = ls[i];
                            m[k] = i;
                            medoid_k_changed = true;
                            for(int j : ownedBy[k])
                                d[j] = d_tilde[j];
                        }
                        //Use computed distances to i to improve lower bounds on sums for all samples in cluster k (see Figure X)
                        for(int j : ownedBy[k])
                            ls[j] = Math.max(ls[j], Math.abs(d_tilde[j]*v[k]-ls[i]));
                        
                    }
                }
                // If the medoid of cluster k has changed, update cluster information
                if(medoid_k_changed)
                {
                    p[k] = dm.dist(c[k], m[k], X, accel);
                    c[k] = m[k];
                }
            }
            ///// assign-to-clusters()  //////
            Arrays.fill(delta_n_in, 0);
            Arrays.fill(delta_n_out, 0);
            Arrays.fill(delta_s_in, 0);
            Arrays.fill(delta_s_out, 0);
            for(int i = 0; i < N; i++)
            {
                // Update lower bounds on distances to medoids based on distances moved by medoids
                for(int k = 0; k < K; k++)
                    lc[i][k] -= p[k];
                // Use the exact distance of current assignment to keep bound tight (might save future calcs)
                lc[i][a[i]] = d[i];
                // Record current assignment and distance aold
                int a_old = a[i];
                double d_old = d[i];
                // Determine nearest medoid, using bounds to eliminate distance calculations
                for(int k = 0; k < K; k++)
                    if(lc[i][k] < d[i])
                    {
                        lc[i][k] = dm.dist(i, c[k], X, accel);
                        if(lc[i][k] < d[i])
                        {
                            a[i] = k;
                            d[i] = lc[i][k];
                        }
                    }
                // If the assignment has changed, update statistics
                if(a_old != a[i])
                {
                    v[a_old]--;
                    v[a[i]]++;
                    changes.increment();
                    ownedBy[a_old].remove(i);
                    ownedBy[a[i]].add(i);
                    ls[i] = 0;
                    delta_n_in[a[i]]++;
                    delta_n_out[a_old]++;
                    delta_s_in[a[i]] += d[i];
                    delta_s_out[a_old] += d_old;
                }
            }
            ///// update-sum-bounds() ///////
            for(int k = 0; k < K; k++)
            {
                double J_abs_s = delta_s_in[k] + delta_s_out[k];
                double J_net_s = delta_s_in[k] - delta_s_out[k];
                double J_abs_n = delta_n_in[k] + delta_n_out[k];
                double J_net_n = delta_n_in[k] - delta_n_out[k];
                
                for(int i : ownedBy[k])
                    ls[i] -= Math.min(J_abs_s-J_net_n*d[i], J_abs_n*d[i] - J_net_s);
            }
        }
        while( changes.sum() > 0 && iter++ < iterLimit);
        
        return DoubleStream.of(d).map(x->x*x).sum();
    }
}
