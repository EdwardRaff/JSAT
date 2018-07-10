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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.DoubleStream;
import jsat.DataSet;
import static jsat.clustering.SeedSelectionMethods.selectIntialPoints;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.utils.concurrent.AtomicDoubleArray;
import static jsat.utils.concurrent.ParallelUtils.*;
import static java.lang.Math.*;
import java.util.*;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.concurrent.AtomicDouble;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

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
        final int N = data.size();
        final int K = medioids.length;
        //medioids = m
        //m(k) : index of current medoid of cluster k,m(k) ∈ {1, . . . ,N}
        final AtomicIntegerArray m = new AtomicIntegerArray(K);
        //c(k) : current medoid of cluster k, that is c(k) = x(m(k))
        final int[] c = medioids;
        //n_1(i) : cluster index of centroid nearest to x(i)
        
        //a(i) : cluster to which x(i) is currently assigned
        final int[] a = assignments;
        
        //d(i) : distance from x(i) to c(a(i))
        final double[] d = new double[N];
        final double[] d_tilde = new double[N];
        //v(k) : number of samples assigned to cluster k
        final AtomicDoubleArray v = new AtomicDoubleArray(K);
        //V (k) : number of samples assigned to a cluster of index less than k +1
        //We don't use V in this implementation. Paper uses it as a weird way of simplifying algorithm description. But not needed
        
        //lc(i, k) : lowerbound on distance from x(i) tom(k)
        double[][] lc = new double[N][K];
        
        //ls(i) : lowerbound on
        AtomicDoubleArray ls = new AtomicDoubleArray(N);
        
        //p(k) : distance moved (teleported) by m(k) in last update
        double[] p = new double[K];
        
        //s(k) : sum of distances of samples in cluster k to medoid k
        AtomicDoubleArray s = new AtomicDoubleArray(K);

        List<Set<Integer>> ownedBy = new ArrayList<>(K);
        for(int i = 0; i < K; i++)
            ownedBy.add(new ConcurrentSkipListSet<>());
        
        //Working sets used in updates
        final AtomicDoubleArray delta_n_in = new AtomicDoubleArray(K);
        final AtomicDoubleArray delta_n_out = new AtomicDoubleArray(K);
        final AtomicDoubleArray delta_s_in = new AtomicDoubleArray(K);
        final AtomicDoubleArray delta_s_out = new AtomicDoubleArray(K);
        
        // initialise //
        for(int k = 0; k < K; k++)
            m.set(k, c[k]);
        run(parallel, N, (start, end) -> 
        {
            for(int i = start; i < end; i++)
            {
                double a_min_val = Double.POSITIVE_INFINITY;
                int a_min_k = 0;
                for(int k = 0; k < K; k++)
                {
                    //Tightly initialise lower bounds on data-to-medoid distances
                    lc[i][k] = dm.dist(i, m.get(k), X, accel);
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
                v.getAndAdd(a[i], 1);
                ownedBy.get(a_min_k).add(i);
                //Update sum of distances to medoid
                s.addAndGet(a[i], d[i]);
                //Initialise lower bound on sum of in-cluster distances to x(i) to zero
                ls.set(i, 0.0);
            }
        });
        
        for(int k = 0; k < K; k++)
            ls.set(m.get(k), s.get(k));
        //end initialization
        
        int iter = 0;
        do
        {
            changes.reset();
            
            ///// update-medoids() //////
            boolean[] medioid_changed = new boolean[K];
            Arrays.fill(medioid_changed, false);
            run(parallel, N, (i)->
            {
                for(int k = 0; k < K; k++)
                {
                    // If the bound test cannot exclude i asm(k)
                    if(ls.get(i) < s.get(k))
                    {
                        // Make ls(i) tight by computing and cumulating all in-cluster distances to x(i),
                        double ls_i_new = 0;
                        for(int j : ownedBy.get(k))
                        {
                            d_tilde[j] = dm.dist(i, j, X, accel);
                            ls_i_new += d_tilde[j];
                        }
                        ls.set(i, ls_i_new);
                        //// Re-perform the test for i as candidate for m(k), now with exact sums. 
                        //If i is the new best candidate, update some cluster information
                        
                        if(ls_i_new < s.get(k))
                        {
                            /* Normally we would just check once. But if we are 
                             * doing this in parallel, we need to make the 
                             * switch out safe. So syncrhonize and re-peat the 
                             * check to avoid any race condition. We do the
                             * check twice b/c the check may happen often, but 
                             * only return true a few times. So lets avoid 
                             * contention and just do a re-check after we found 
                             * out we needed to do an update. */
                            synchronized (s)
                            {
                                if (ls_i_new < s.get(k))
                                {
                                    s.set(k, ls_i_new);
                                    m.set(k, i);
                                    medioid_changed[k] = true;
                                    for(int j : ownedBy.get(k))
                                        d[j] = d_tilde[j];
                                }
                            }
                        }
                        //Use computed distances to i to improve lower bounds on sums for all samples in cluster k (see Figure X)
                        for(int j : ownedBy.get(k))
                            ls.accumulateAndGet(j, d[j]*v.get(k), (ls_j, d_jXv_k) -> max(ls_j, abs(d_jXv_k-ls_j)));
                    }
                }
            });
            
            // If the medoid of cluster k has changed, update cluster information
            run(parallel, K, (k)->
            {
                if(medioid_changed[k])
                {
                    p[k] = dm.dist(c[k], m.get(k), X, accel);
                    c[k] = m.get(k);
                }
                
                //lets sneak in zero-ing out the delta arrays for the next stwp while are are doing a parallel loop
                delta_n_in.set(k, 0.0);
                delta_n_out.set(k, 0.0);
                delta_s_in.set(k, 0.0);
                delta_s_out.set(k, 0.0);
            });
            ///// assign-to-clusters()  //////
            run(parallel, N, (i)->
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
                    v.getAndDecrement(a_old);
                    v.getAndIncrement(a[i]);
                    changes.increment();
                    ownedBy.get(a_old).remove(i);
                    ownedBy.get(a[i]).add(i);
                    ls.set(i, 0.0);
                    delta_n_in.getAndIncrement(a[i]);
                    delta_n_out.getAndIncrement(a_old);
                    delta_s_in.getAndAdd(a[i], d[i]);
                    delta_s_in.getAndAdd(a_old, d_old);
                }
            });
            ///// update-sum-bounds() ///////
            double[] J_abs_s = new double[K];
            double[] J_net_s = new double[K];
            double[] J_abs_n = new double[K];
            double[] J_net_n = new double[K];
            for(int k = 0; k < K; k++)
            {
                J_abs_s[k] = delta_s_in.get(k) + delta_s_out.get(k);
                J_net_s[k] = delta_s_in.get(k) - delta_s_out.get(k);
                J_abs_n[k] = delta_n_in.get(k) + delta_n_out.get(k);
                J_net_n[k] = delta_n_in.get(k) - delta_n_out.get(k);
            }
            run(parallel, N, (start, end)->
            {
                for(int i = start; i < end; i++)
                {
                    double ls_i_delta = 0;
                    for(int k = 0; k < K; k++)
                        ls_i_delta -= min(J_abs_s[k]-J_net_n[k]*d[i], J_abs_n[k]*d[i] - J_net_s[k]);
                    ls.getAndAdd(i, ls_i_delta);
                }
            });
        }
        while( changes.sum() > 0 && iter++ < iterLimit);
        
        return streamP(DoubleStream.of(d), parallel).map(x->x*x).sum();
    }
    
    /**
     * Computes the medoid of the data 
     * @param parallel whether or not the computation should be done using multiple cores
     * @param X the list of all data
     * @param dm the distance metric to get the medoid with respect to
     * @return the index of the point in <tt>X</tt> that is the medoid
     */
    public static int medoid(boolean parallel, List<? extends Vec> X, DistanceMetric dm)
    {
        IntList order = new IntList(X.size());
        ListUtils.addRange(order, 0, X.size(), 1);
        List<Double> accel = dm.getAccelerationCache(X, parallel);
        return medoid(parallel, order, X, dm, accel);
    }
    
    /**
     * Computes the medoid of a sub-set of data
     * @param parallel whether or not the computation should be done using multiple cores
     * @param indecies the indexes of the points to get the medoid of 
     * @param X the list of all data
     * @param dm the distance metric to get the medoid with respect to
     * @param accel the acceleration cache for the distance metric
     * @return the index value contained within indecies that is the medoid 
     */
    public static int medoid(boolean parallel, Collection<Integer> indecies, List<? extends Vec> X, DistanceMetric dm, List<Double> accel)
    {
        final int N = X.size();
        AtomicDoubleArray l = new AtomicDoubleArray(N);
        
        AtomicDouble e_cl = new AtomicDouble(Double.POSITIVE_INFINITY);
        
        IntList rand_order = new IntList(indecies);
        Collections.shuffle(rand_order, RandomUtil.getRandom());
        ThreadLocal<double[]> d_local = ThreadLocal.withInitial(() -> new double[N]);
        ParallelUtils.streamP(rand_order.streamInts(), parallel).forEach(i ->
        {
            double[] d = d_local.get();
            double d_avg = 0;
            if (l.get(i) < e_cl.get())
            {
                for (int j : indecies)
                    d_avg += (d[j] = dm.dist(i, j, X, accel));
                d_avg /= indecies.size() - 1;
                final double l_i = d_avg;
                // set l(i) to to be tight, that is l(i) = E(i)
                l.set(i, l_i);
                if (l_i < e_cl.get())//We might be the best?
                    e_cl.getAndUpdate(val -> min(val, l_i));//atomic set to maximum
                //l(j)←max(l(j), |l(i)−d(j)|) // using ||x(i)−x(j)|| to possibly improve bound.
                for (int j : indecies)
                    l.getAndUpdate(j, (l_j) -> max(l_j, abs(l_i - d[j]) ) );
            }
        });
        
        //OK, who had the lowest energy? Thats our median. 
        for(int i : indecies)
            if(l.get(i) == e_cl.get())//THIS IS SAFE, we explicily set l and e_cl to the same exact values in the algorithm. So we can do a hard equality check on the doubles
                return i;

        return -1;//Some error occred
    }
}
