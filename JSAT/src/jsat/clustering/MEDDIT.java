/*
 * Copyright (C) 2018 Edward Raff <Raff.Edward@gmail.com>
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

import java.util.*;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;
import jsat.DataSet;
import static jsat.clustering.SeedSelectionMethods.selectIntialPoints;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.math.OnLineStatistics;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.AtomicDoubleArray;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class MEDDIT extends PAM
{
    private double tolerance = 0.01;
    
    public MEDDIT(DistanceMetric dm, Random rand, SeedSelectionMethods.SeedSelection seedSelection)
    {
        super(dm, rand, seedSelection);
    }

    public MEDDIT(DistanceMetric dm, Random rand)
    {
        super(dm, rand);
    }

    public MEDDIT(DistanceMetric dm)
    {
        super(dm);
    }

    public MEDDIT()
    {
        super();
    }

    public void setTolerance(double tolerance)
    {
        this.tolerance = tolerance;
    }

    public double getTolerance()
    {
        return tolerance;
    }
    
    
    @Override
    protected double cluster(DataSet data, boolean doInit, int[] medioids, int[] assignments, List<Double> cacheAccel, boolean parallel)
    {
        DoubleAdder totalDistance =new DoubleAdder();
        LongAdder changes = new LongAdder();
        Arrays.fill(assignments, -1);//-1, invalid category!
        
        List<Vec> X = data.getDataVectors();
        final List<Double> accel;
        final int N = data.size();
        
        if(doInit)
        {
            TrainableDistanceMetric.trainIfNeeded(dm, data);
            accel = dm.getAccelerationCache(X);
            selectIntialPoints(data, medioids, dm, accel, rand, seedSelection);
        }
        else
            accel = cacheAccel;
        
        double tol;
        if(tolerance < 0)
            tol = 1.0/data.size();
        else
            tol = tolerance;

        int iter = 0;
        do
        {
            changes.reset();
            totalDistance.reset();
            
            ParallelUtils.run(parallel, N, (start, end)->
            {
                for(int i = start; i < end; i++)
                {
                    int assignment = 0;
                    double minDist = dm.dist(medioids[0], i, X, accel);

                    for (int k = 1; k < medioids.length; k++)
                    {
                        double dist = dm.dist(medioids[k], i, X, accel);
                        if (dist < minDist)
                        {
                            minDist = dist;
                            assignment = k;
                        }
                    }

                    //Update which cluster it is in
                    if (assignments[i] != assignment)
                    {
                        changes.increment();
                        assignments[i] = assignment;
                    }
                    totalDistance.add(minDist * minDist);
                }
            });
            
            //Update the medoids
            IntList owned_by_k = new IntList(N);
            for(int k = 0; k < medioids.length; k++)
            {
                owned_by_k.clear();
                for(int i = 0; i < N; i++)
                    if(assignments[i] == k)
                        owned_by_k.add(i);
                if(owned_by_k.isEmpty())
                    continue;
                
                medioids[k] = medoid(parallel, owned_by_k, tol, X, dm, accel);
                

            }
        }
        while( changes.sum() > 0 && iter++ < iterLimit);
        
        return totalDistance.sum();
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
        return medoid(parallel, X, 1.0/X.size(), dm);
    }
    
    
    /**
     * Computes the medoid of the data 
     * @param parallel whether or not the computation should be done using multiple cores
     * @param X the list of all data
     * @param tol
     * @param dm the distance metric to get the medoid with respect to
     * @return the index of the point in <tt>X</tt> that is the medoid
     */
    public static int medoid(boolean parallel, List<? extends Vec> X, double tol, DistanceMetric dm)
    {
        IntList order = new IntList(X.size());
        ListUtils.addRange(order, 0, X.size(), 1);
        List<Double> accel = dm.getAccelerationCache(X, parallel);
        return medoid(parallel, order, tol, X, dm, accel);
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
        return medoid(parallel, indecies, 1.0/indecies.size(), X, dm, accel);
    }
    
    /**
     * Computes the medoid of a sub-set of data
     * @param parallel whether or not the computation should be done using multiple cores
     * @param indecies the indexes of the points to get the medoid of 
     * @param tol 
     * @param X the list of all data
     * @param dm the distance metric to get the medoid with respect to
     * @param accel the acceleration cache for the distance metric
     * @return the index value contained within indecies that is the medoid 
     */
    public static int medoid(boolean parallel, Collection<Integer> indecies, double tol, List<? extends Vec> X, DistanceMetric dm, List<Double> accel)
    {
        final int N = indecies.size();
        
        if(tol <= 0 || N < SystemInfo.LogicalCores)//Really just not enough points, lets simplify
            return PAM.medoid(parallel, indecies, X, dm, accel);
        
        
        final double log2d = Math.log(1)-Math.log(tol);
        
        /**
         * Online estimate of the standard deviation that will be used
         */
        final OnLineStatistics distanceStats;
        /**
         * This array contains the current sum of all distance computations done
         * for each index. Corresponds to mu in the paper.
         */
        AtomicDoubleArray totalDistSum = new AtomicDoubleArray(N);
        /**
         * This array contains the current number of distance computations that
         * have been done for each feature index. Corresponds to T_i in the
         * paper.
         */
        AtomicIntegerArray totalDistCount = new AtomicIntegerArray(N);
        final int[] indx_map = indecies.stream().mapToInt(i->i).toArray();
        final boolean symetric = dm.isSymmetric();
        final double[] lower_bound_est = new double[N];
        final double[] upper_bound_est = new double[N];

        ThreadLocal<Random> localRand = ThreadLocal.withInitial(RandomUtil::getRandom);

        //First pass, lets pull every "arm" (compute a dsitance) for each datumn at least once, so that we have estiamtes to work with. 
        distanceStats = ParallelUtils.run(parallel, N, (start, end)->
        {
            Random rand = localRand.get();
            OnLineStatistics localStats = new OnLineStatistics();
            for(int i = start; i < end; i++)
            {
                int j = rand.nextInt(N);
                while(j == i)
                    j = rand.nextInt(N);
                

                double d_ij = dm.dist(indx_map[i], indx_map[j], X, accel);
                localStats.add(d_ij);
                totalDistSum.addAndGet(i, d_ij);
                totalDistCount.incrementAndGet(i);
                if(symetric)
                {
                    totalDistSum.addAndGet(j, d_ij);
                    totalDistCount.incrementAndGet(j);
                }
            }
            
            return localStats;
        }, (a,b)-> OnLineStatistics.add(a, b));
        
        //Now lets prepare the lower and upper bound estimates
        ConcurrentSkipListSet<Integer> lowerQ = new ConcurrentSkipListSet<>((Integer o1, Integer o2) -> 
        {
            int cmp = Double.compare(lower_bound_est[o1], lower_bound_est[o2]);
            if(cmp == 0)//same bounds, but sort by identity to avoid issues
                cmp = o1.compareTo(o2);
            return cmp;
        });
        
        ConcurrentSkipListSet<Integer> upperQ = new ConcurrentSkipListSet<>((Integer o1, Integer o2) -> 
        {
            int cmp = Double.compare(upper_bound_est[o1], upper_bound_est[o2]);
            if(cmp == 0)//same bounds, but sort by identity to avoid issues
                cmp = o1.compareTo(o2);
            return cmp;
        });

        ParallelUtils.run(parallel, N, (start, end)->
        {
            double v = distanceStats.getVarance();
            for(int i = start; i < end; i++)
            {
                int T_i = totalDistCount.get(i);
                double c_i = Math.sqrt(2*v*log2d/T_i);
                lower_bound_est[i] = totalDistSum.get(i)/T_i - c_i;
                upper_bound_est[i] = totalDistSum.get(i)/T_i + c_i;
                lowerQ.add(i);
                upperQ.add(i);
            }
        });
        
        
        //Now lets start sampling! 
        
        //how many points should we pick and sample? Not really discussed in paper- but a good idea for efficency (dont want to pay that Q cost as much as possible)
        /**
         * to-pull is how many arms we will select per iteration
         */
        int num_to_pull;
        /**
         * to sample is how many random pairs we will pick for each pulled arm
         */
        int samples;
        
        if(parallel)
        {
            num_to_pull = Math.max(SystemInfo.LogicalCores, 32);
            samples = Math.min(32, N-1);
        }
        else
        {
            num_to_pull = Math.min(32, N);
            samples = Math.min(32, N-1);
        }
        
        /**
         * The levers we will pull this iteration, and then add back in
         */
        IntList to_pull = new IntList();
        /**
         * the levers we must add back in but not update b/c they hit max evaluations and the confidence bound is tight
         */
        IntList toAddBack = new IntList();
        boolean[] isExact = new boolean[N];
        Arrays.fill(isExact, false);
        int numExact = 0;
        
        
        while(numExact < N)//loop should break out before this ever happens
        {
            to_pull.clear();
            toAddBack.clear();

            //CONVERGENCE CEHCK
            if(upper_bound_est[upperQ.first()] < lower_bound_est[lowerQ.first()])
            {
                //WE are done!
                return indx_map[upperQ.first()];
            }
            

            while(to_pull.size() < num_to_pull)
            {
                
                if(lowerQ.isEmpty())
                    break;//we've basically evaluated everyone
                int i = lowerQ.pollFirst();
                
                
                if(totalDistCount.get(i) >= N-1 && !isExact[i])//Lets just replace with exact value
                {
                    double avg_d_i = ParallelUtils.run(parallel, N, (start, end)->
                    {
                        double d = 0;
                        for (int j = start; j < end; j++)
                            if (i != j)
                                d += dm.dist(indx_map[i], indx_map[j], X, accel);
                        return d;
                    }, (a, b)->a+b);
                    avg_d_i /= N-1;
                    
                    upperQ.remove(i);
                    lower_bound_est[i] = upper_bound_est[i] = avg_d_i;
                    totalDistSum.set(i, avg_d_i);
                    totalDistCount.set(i, N);
                    isExact[i] = true;
                    numExact++;
//                    System.out.println("Num Exact: " + numExact);
                    //OK, exavt value for datumn I is set. 
                    toAddBack.add(i);
                }
                

                if(!isExact[i])
                    to_pull.add(i);
            }
            
            //OK, lets now pull a bunch of levers / measure distances
            
            OnLineStatistics changeInStats = ParallelUtils.run(parallel, to_pull.size(), (start, end)->
            {
                Random rand = localRand.get();
                OnLineStatistics localStats = new OnLineStatistics();
                for(int i_count = start; i_count < end; i_count++)
                {
                    int i = to_pull.get(i_count);
                    for(int j_count = 0; j_count < samples; j_count++)
                    {
                        int j = rand.nextInt(N);
                        while(j == i)
                            j = rand.nextInt(N);
                        
                        double d_ij = dm.dist(indx_map[i], indx_map[j], X, accel);
                        localStats.add(d_ij);
                        totalDistSum.addAndGet(i, d_ij);
                        totalDistCount.incrementAndGet(i);
                        if(symetric && !isExact[j])
                        {
                            totalDistSum.addAndGet(j, d_ij);
                            totalDistCount.incrementAndGet(j);
                        }
                    }
                }
                
                return localStats;
            }, (a,b) -> OnLineStatistics.add(a, b));
            
            if(!to_pull.isEmpty())//might be empty if everyone went over the threshold
                distanceStats.add(changeInStats);
            
            //update bounds and re-insert
            double v = distanceStats.getVarance();
            //we are only updating the bounds on the levers we pulled
            //that may mean some old bounds are stale
            //these values are exact
            lowerQ.addAll(toAddBack);
            upperQ.addAll(toAddBack);
            upperQ.removeAll(to_pull);
            for(int i : to_pull)
            {
                int T_i = totalDistCount.get(i);
                double c_i = Math.sqrt(2*v*log2d/T_i);
                lower_bound_est[i] = totalDistSum.get(i)/T_i - c_i;
                upper_bound_est[i] = totalDistSum.get(i)/T_i + c_i;
                lowerQ.add(i);
                upperQ.add(i);
            }
        }
        
        //We can reach this point on small N or low D datasets. Iterate and return the correct value
        int bestIndex = 0;
        for(int i = 1; i < N; i++)
            if(lower_bound_est[i] < lower_bound_est[bestIndex])
                bestIndex = i;
        
        return bestIndex;
    }
}
