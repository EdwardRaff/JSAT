package jsat.clustering.hierarchical;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.KClustererBase;
import jsat.clustering.dissimilarity.UpdatableClusterDissimilarity;
import jsat.math.OnLineStatistics;

import static jsat.clustering.dissimilarity.AbstractClusterDissimilarity.*;
import jsat.utils.IntPriorityQueue;

/**
 *
 * @author Edward Raff
 */
public class PriorityHAC extends KClustererBase
{

    private static final long serialVersionUID = -702489462117567542L;
    private UpdatableClusterDissimilarity distMeasure;
    
    /**
     * Stores the merge list, each merge is in a pair of 2 values. The final 
     * merge list should contain the last merged pairs at the front of the array
     * (index  0, 1), and the first merges at the end of the array. The left 
     * value in each pair is the index of the data point that the clusters were 
     * merged under, while the right value is the index that was merged in and 
     * treated as no longer its own cluster. 
     */
    private int[] merges;
    private DataSet curDataSet;

    public PriorityHAC(UpdatableClusterDissimilarity dissMeasure)
    {
        this.distMeasure = dissMeasure;
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public PriorityHAC(PriorityHAC toCopy)
    {
        this.distMeasure = toCopy.distMeasure.clone();
        if(toCopy.merges != null)
            this.merges = Arrays.copyOf(toCopy.merges, toCopy.merges.length);
        this.curDataSet = toCopy.curDataSet.shallowClone();
    }
    
    
    
    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()), designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()), threadpool, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, clusters, clusters, threadpool, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, int[] designations)
    {
        return cluster(dataSet, clusters, clusters, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, lowK, highK, designations);
    }

    
    private void updateDistanceTableAndQueues(List<IntPriorityQueue> P, int[] I, int k1, int k2, final double[][] distanceMatrix)
    {
        IntPriorityQueue Pk1 = P.get(k1);
        for(int i = 0; i < P.size(); i++)
        {
            if(I[i] == 0 || i == k2 || i == k1)
                continue;
            IntPriorityQueue curTargetQ = P.get(i);
            
            curTargetQ.remove(k1);
            curTargetQ.remove(k2);
            
            double dis = distMeasure.dissimilarity(k1, I[k1], k2, I[k2], i, I[i], distanceMatrix);
            setDistance(distanceMatrix, i, k1, dis);
            curTargetQ.add(k1);
            Pk1.add(i);
        }
    }
    
    private List<IntPriorityQueue> setUpProrityQueue(int[] I, final double[][] distanceMatrix)
    {
        List<IntPriorityQueue> P = new ArrayList<IntPriorityQueue>(I.length);
        for(int i = 0; i < I.length; i++)
        {
            //The row index we are considering
            final int supremeIndex = i;
            IntPriorityQueue pq = new IntPriorityQueue(I.length, new Comparator<Integer>() 
            {
                @Override
                public int compare(Integer o1, Integer o2)
                {
                    double d1 = getDistance(distanceMatrix, supremeIndex, o1);
                    double d2 = getDistance(distanceMatrix, supremeIndex, o2);
                    
                    return Double.compare(d1, d2);
                }
            }, IntPriorityQueue.Mode.BOUNDED);
            
            
            //Fill up the priority que
            for(int j = 0; j < I.length; j++ )
            {
                if(i == j)
                    continue;
                pq.add(j);
            }
            
            P.add(pq);
        }
        
        return P;
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        this.curDataSet = dataSet;
        merges = new int[dataSet.getSampleSize()*2-2];
        /**
         * Keeps track of the current cluster size for the data point. If zero, 
         * the data point has been merged and is no longer a candidate for 
         * future consideration. If non zero, it indicates the number of data 
         * points in its implicit cluster. All points start out in their own 
         * implicit cluster. 
         */
        int[] I = new int[dataSet.getSampleSize()];
        Arrays.fill(I, 1);
        this.curDataSet = dataSet;
        
        /*
         * Keep track of the average dist when merging, stop when it becomes abnormaly large
         */
        OnLineStatistics distChange = new OnLineStatistics();
        
        final double[][] distanceMatrix = createDistanceMatrix(dataSet, distMeasure);
        
        //Create priority ques for each data point
        List<IntPriorityQueue> P = setUpProrityQueue(I, distanceMatrix);
        
        //We will choose the cluster size as the most abnormal jump in dissimilarity from a merge
        int clusterSize = lowK;
        double maxStndDevs = Double.MIN_VALUE;
        
        //We now know the dissimilarity matrix & Qs we can begin merging
        
        //We will perform all merges, and store them - and then return a clustering level from the merge history
        for(int k = 0; k < I.length-1; k++)
        {
            int k1 = -1, k2 = -1;
            double dk1 = Double.MAX_VALUE, tmp;
            
            for(int i = 0; i < P.size(); i++)
                if( I[i] > 0 &&  (tmp = getDistance(distanceMatrix, i, P.get(i).element())) < dk1)
                {
                    dk1 = tmp;
                    k1 = i;
                    k2 = P.get(i).element();
                }
            
            //Keep track of the changes in cluster size, and mark if this one was abnormall large
            distChange.add(dk1);
            
            if( (I.length - k) >= lowK && (I.length - k) <= highK)//IN the cluster window?
            {
                double stndDevs = (dk1-distChange.getMean())/distChange.getStandardDeviation();
                if(stndDevs > maxStndDevs)
                {
                    maxStndDevs = stndDevs;
                    clusterSize = I.length-k;
                }
            }
            
                
            
            //We have now found the smalles pair in O(n), first we will update the Qs and matrix. k1 will be the new merged cluster
            P.get(k1).clear();//This Q will need all new values
            P.get(k2).clear();//This Q will no longer be used
            
            updateDistanceTableAndQueues(P, I, k1, k2, distanceMatrix);
            
            //Now we fix up designations
            //Note which clusters were just merged
            merges[k*2] = k2;
            merges[k*2+1] = k1;
            //Update counts
            I[k1] += I[k2];
            I[k2] = 0;
        }
        reverseMergeArray();
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        designations = assignClusterDesignations(designations, clusterSize);
        
        
        return designations;
    }

    
    /**
     * Reverses the merge array so that the front contains the last merges instead of the first.
     * This is done so that creating new clusters is accessed in order which is cache friendly. <br>
     * This method must be called once before using {@link #assignClusterDesignations(int[], int) }
     */
    private void reverseMergeArray()
    {
        for(int i = 0; i < merges.length/2; i++)
        {
            int tmp = merges[i];
            merges[i] = merges[merges.length-i-1];
            merges[merges.length-i-1] = tmp;
        }
    }
    
    /**
     * The PriorityHAC stores its merging order, so that multiple clusterings 
     * can of different sizes can be obtained without having to recluster the 
     * data set. This is possible in part because HAC is deterministic. <br>
     * This returns <tt>true</tt> if there is currently a data set and its merge
     * order stored. 
     * 
     * @return <tt>true</tt> if you can call for more clusterings, 
     * <tt>false</tt> if no data set has been clustered. 
     */
    public boolean hasStoredClustering()
    {
        return curDataSet != null;
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
        if(!hasStoredClustering())
            return null;
        return assignClusterDesignations(designations, clusters);
    }
    
    /**
     * Returns the assignment array for that would have been computed for the 
     * previous data set with the desired number of clusters.
     * 
     * @param clusters the number of clusters desired
     * @return the list of data points in each cluster, or <tt>null</tt> if no 
     * data set has been clustered. 
     * @see #hasStoredClustering() 
     */
    public List<List<DataPoint>> getClusterDesignations(int clusters)
    {
        if(!hasStoredClustering())
            return null;
        int[] assignments = new int[curDataSet.getSampleSize()];
        return createClusterListFromAssignmentArray(assignments, curDataSet);
    }

    /**
     * Goes through the <tt>merge</tt> array in order from last merge to first, and sets the cluster assignment for each data point based on the merge list. 
     * @param designations the array to store the designations in, or null to have a new one created automatically. 
     * @param clusters the number of clusters to assume
     * @return the array storing the designations. A new one will be created and returned if <tt>designations</tt> was null. 
     */
    private int[] assignClusterDesignations(int[] designations, int clusters)
    {
        return assignClusterDesignations(designations, clusters, merges);
    }
    
    /**
     * Goes through the <tt>merge</tt> array in order from last merge to first, and sets the cluster assignment for each data point based on the merge list. 
     * @param designations the array to store the designations in, or null to have a new one created automatically. 
     * @param clusters the number of clusters to assume
     * @param merges the array of merge pairs
     * @return the array storing the designations. A new one will be created and returned if <tt>designations</tt> was null. 
     */
    protected static int[] assignClusterDesignations(int[] designations, int clusters, int[] merges)
    {
        int curCluster = 0;
        Arrays.fill(designations, -1);
        for(int i = 0; i < merges.length; i++)
        {
            if(designations[merges[i]] == -1)//it has not been assigned
            {
                if(curCluster < clusters)//It will be a top level cluster
                    designations[merges[i]] = curCluster++;
                else
                    designations[merges[i]] = designations[merges[i-1]];//The new cluster is always in an odd index, so its parrent is the even index to the left 
            }
        }
        return designations;
    }

    @Override
    public PriorityHAC clone()
    {
        return new PriorityHAC(this);
    }
    
}
