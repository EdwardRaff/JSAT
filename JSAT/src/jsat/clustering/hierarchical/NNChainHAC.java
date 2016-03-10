/*
 * Copyright (C) 2016 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.clustering.hierarchical;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import static jsat.clustering.ClustererBase.createClusterListFromAssignmentArray;
import jsat.clustering.KClustererBase;
import jsat.clustering.dissimilarity.LanceWilliamsDissimilarity;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.math.OnLineStatistics;
import jsat.utils.FakeExecutor;
import jsat.utils.IndexTable;
import jsat.utils.IntDoubleMap;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class NNChainHAC extends KClustererBase
{
    private LanceWilliamsDissimilarity distMeasure;
    private DistanceMetric dm;
    
    /**
     * Stores the merge list, each merge is in a pair of 2 values. The final 
     * merge list should contain the last merged pairs at the front of the array
     * (index  0, 1), and the first merges at the end of the array. The left 
     * value in each pair is the index of the data point that the clusters were 
     * merged under, while the right value is the index that was merged in and 
     * treated as no longer its own cluster. 
     */
    private int[] merges;
    
    public NNChainHAC(LanceWilliamsDissimilarity distMeasure)
    {
        this(distMeasure, new EuclideanDistance());
    }
    
    public NNChainHAC(LanceWilliamsDissimilarity distMeasure, DistanceMetric distance)
    {
        this.distMeasure = distMeasure;
        this.dm = distance;
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    protected NNChainHAC(NNChainHAC toCopy)
    {
        this.distMeasure = toCopy.distMeasure.clone();
        this.dm = toCopy.dm.clone();
        if(toCopy.merges != null)
            this.merges = Arrays.copyOf(toCopy.merges, toCopy.merges.length);
    }
    
    @Override
    public NNChainHAC clone()
    {
        return new NNChainHAC(this);
    }

    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        return cluster(dataSet, new FakeExecutor(), designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, 2, (int) Math.sqrt(dataSet.getSampleSize()), threadpool, designations);
    }

    private double getDist(int a, int j, int[] size, List<Vec> vecs, List<Double> cache, IntDoubleMap[] d_xk)
    {
        double dist = -1;//negative value nosense
        if (size[j] == 1 && size[a] == 1)
            dist = dm.dist(a, j, vecs, cache);
        else
        {
            if(d_xk[j] != null)
            {
                Double tmp = d_xk[j].get(a);
                if(tmp != null)
                    dist = tmp;
            }
            if(dist < 0)//wasn't found searching d_xk
                dist = d_xk[a].get(j);//has to be found now
        }
        
        if(Double.isNaN(dist))
            return getDist(a, j, size, vecs, cache, d_xk);
        
        return dist;
    }

    /**
     * Returns the assignment array for that would have been computed for the 
     * previous data set with the desired number of clusters. 
     * 
     * @param designations the array to store the assignments in
     * @param clusters the number of clusters desired
     * @return the original array passed in, or <tt>null</tt> if no data set has been clustered. 
     * @see #hasStoredClustering() 
     */
    public int[] getClusterDesignations(int[] designations, int clusters)
    {
        if(merges == null)
            return null;
        return PriorityHAC.assignClusterDesignations(designations, clusters, merges);
    }
    
    /**
     * Returns the assignment array for that would have been computed for the 
     * previous data set with the desired number of clusters.
     * 
     * @param clusters the number of clusters desired
     * @param data
     * @return the list of data points in each cluster, or <tt>null</tt> if no 
     * data set has been clustered. 
     * @see #hasStoredClustering() 
     */
    public List<List<DataPoint>> getClusterDesignations(int clusters, DataSet data)
    {
        if(merges == null || (merges.length+2)/2 != data.getSampleSize())
            return null;
        int[] assignments = new int[data.getSampleSize()];
        assignments = getClusterDesignations(assignments, clusters);
        return createClusterListFromAssignmentArray(assignments, data);
    }
    
    @Override
    public int[] cluster(DataSet dataSet, int clusters, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, clusters, clusters, threadpool, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, int[] designations)
    {
        return cluster(dataSet, clusters, new FakeExecutor(), designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        
        int N = dataSet.getSampleSize();
        
        merges = new int[N*2-2];
        /**
         * Keeps track of which index was removed from the list due to being
         * merged at the given point
         */
        IntList merge_removed = new IntList();
        /**
         * Keeps track of which index was kept and was in the merge at the given
         * point
         */
        IntList merge_kept = new IntList();
        /**
         * The number of items in the cluster denoted at the given index
         */
        int[] size = new int[N];
        Arrays.fill(size, 1);
        

        /**
         * Stores the distance between the two clusters that were merged at each step. 
         */
        double[] mergedDistance = new double[N-1];
        int L_pos = 0;
        
        IntList S = new IntList(N);
        ListUtils.addRange(S, 0, N, 1);
        
        
        IntDoubleMap[] dist_map = new IntDoubleMap[N];
        
        List<Vec> vecs = dataSet.getDataVectors();
        List<Double> cache = dm.getAccelerationCache(vecs, threadpool);
        
        int[] chain = new int[N];
        int chainPos = 0;
        
        while(S.size() > 1)
        {
            int a, b;
            if(chainPos <= 3)
            {
                // 7: a←(any element of S) >  E.g. S[0]
                a = S.getI(0);
                // 8: chain ←[a]
                chainPos = 0;
                chain[chainPos++] = a;
                // 9: b←(any element of S \ {a})   > E.g. S[1]
                b = S.getI(1);
            }
            else
            {
                // 11: a←chain[−4]  > 4th to last element
                a = chain[chainPos-4];
                // 12: b←chain[−3]  > 3rd to last element
                b = chain[chainPos-3];
                // 13: Remove chain[−1], chain[−2] and chain[−3]  >  Cut the tail (x, y,x)
                chainPos -= 3;
            }
            
            double dist_ab;
            do//15:
            {
                // 16: c ← argmin_{ x!=a} d[x, a] with preference for b
                
                int c = b;
                double minDist = getDist(a, c, size, vecs, cache, dist_map);
                
                for(int j : S)
                {
                    if(j == a || j == c)
                        continue;//we already have these guys! just not removed from S yet
                    
                    double dist = getDist(a, j, size, vecs, cache, dist_map);
                    if(dist < minDist)
                    {
                        minDist = dist;
                        c = j;
                    }
                }
                dist_ab = minDist;
                      
                //17: a, b ← c, a
                b = a;
                a = c;
                //18: Append a to chain
                chain[chainPos++] = a;
                
            }
            while (chainPos < 3 || a != chain[chainPos-3]); //19: until length(chain) ≥ 3 and a = chain[−3]  > a, b are reciprocal
            
            int n = Math.max(a, b);
            int removed = Math.min(a, b);
            
            // 20: Append (a, b, d[a, b]) to L  >  nearest neighbors.
            merges[merges.length-L_pos*2-1] = removed;
            merges[merges.length-L_pos*2-2] = n;
            
            merge_removed.add(removed);
            merge_kept.add(n);
            
            mergedDistance[L_pos] = dist_ab;
            
            L_pos++;
            // 21: Remove a, b from S
            S.removeAll(Arrays.asList(a, b));
//            System.out.println("Removed " + a + " " + b + " S=" + S + " chain=" + IntList.view(chain, chainPos));
            // 22: n←(new node label)
            
            for(int i = 0; i < chainPos; i++)//bug in paper? we end with [a, b, a] in the chain, but one of them is a bad index now
                if(chain[i] == removed)
                    chain[i] = n;
            // 23: size[n]←size[a]+size[b]
            int size_a = size[a], size_b = size[b];
            //set defered till later to make sure we don't muck anything needed in computatoin
            
            // 24: Update d with the information d[n,x], for all x ∈ S.
            
            IntDoubleMap map_n = S.isEmpty() ? null : new IntDoubleMap(S.size());
            for(int x : S)
            {
                double d_ax = getDist(a, x, size, vecs, cache, dist_map);
                double d_bx = getDist(b, x, size, vecs, cache, dist_map);
                double d_xn = distMeasure.dissimilarity(size_a, size_b, size[x], dist_ab, d_ax, d_bx);
                if(Double.isNaN(d_xn))
                    d_xn = distMeasure.dissimilarity(size_a, size_b, size[x], dist_ab, d_ax, d_bx);
                if(dist_map[x] == null)
                {
//                    dist_map[x] = new IntDoubleMap(1);
//                    dist_map[x].put(n, d_xn);
                }
                else //if(dist_map[x] != null)
                {
                    dist_map[x].remove(b);
                    dist_map[x].put(n, d_xn);
                }
                
                map_n.put(x, d_xn);
                
            }
            
            dist_map[removed] = null;//no longer in use no mater what
            dist_map[n] = map_n;
            
            // 25: S ← S ∪ {n}
            size[n] = size_a + size_b;
            S.add(n);
        }
        
        //Now that we are done clustering, we need to re-order the merges so that the smallest distances are mergered first
        IndexTable it = new IndexTable(mergedDistance);
        it.apply(merge_kept);
        it.apply(merge_removed);
        it.apply(mergedDistance);
        for(int i = 0; i < it.length(); i++)
        {
            merges[merges.length-i*2-1] = merge_removed.get(i);
            merges[merges.length-i*2-2] = merge_kept.get(i);
        }

        //Now lets figure out a guess at the cluster size
        /*
         * Keep track of the average dist when merging, mark when it becomes abnormaly large as a guess at K
         */
        OnLineStatistics distChange = new OnLineStatistics();
        double maxStndDevs = Double.MIN_VALUE;

        /**
         * How many clusters to return
         */
        int clusterSize = lowK;

        for(int i = 0; i < mergedDistance.length; i++)
        {
            
            //Keep track of the changes in cluster size, and mark if this one was abnormall large
            distChange.add(mergedDistance[i]);

            int curK = N - i;
            if (curK >= lowK && curK <= highK)//In the cluster window?
            {
                double stndDevs = (mergedDistance[i] - distChange.getMean()) / distChange.getStandardDeviation();
                if (stndDevs > maxStndDevs)
                {
                    maxStndDevs = stndDevs;
                    clusterSize = curK;
                }
            }
        }
        
        PriorityHAC.assignClusterDesignations(designations, clusterSize, merges);
        
        return designations;
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        return cluster(dataSet, lowK, highK, new FakeExecutor(), designations);
    }
    
}
