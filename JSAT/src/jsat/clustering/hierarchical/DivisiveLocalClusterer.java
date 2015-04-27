
package jsat.clustering.hierarchical;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.*;
import jsat.clustering.evaluation.ClusterEvaluation;

/**
 * DivisiveLocalClusterer is a hierarchical clustering method that works by 
 * splitting the data set into sub trees from the top down. Unlike many top-up 
 * methods, such as {@link SimpleHAC}, top-down methods require another 
 * clustering method to perform the splitting at each iteration. If the base
 * method is not deterministic, then the top-down method will not be 
 * deterministic. 
 * <br><br>
 * Specifically, DivisiveLocalClusterer greedily chooses the cluster to split 
 * based on an evaluation of only the cluster being split. Because of this local
 * search of the world, DivisiveLocalClusterer has poor performance in 
 * determining the number of clusters in the data set. As such, only the methods
 * where the exact number of clusters are recommended. 
 * <br>
 * This greedy strategy can also lead to drilling down clusters into small 
 * parts, and works best when only a small number of clusters are needed. 
 * 
 * @author Edward Raff
 */
public class DivisiveLocalClusterer extends KClustererBase
{

	private static final long serialVersionUID = 8616401472810067778L;
	private KClusterer baseClusterer;
    private ClusterEvaluation clusterEvaluation;

    public DivisiveLocalClusterer(KClusterer baseClusterer, ClusterEvaluation clusterEvaluation) 
    {
        this.baseClusterer = baseClusterer;
        this.clusterEvaluation = clusterEvaluation;
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public DivisiveLocalClusterer(DivisiveLocalClusterer toCopy)
    {
        this(toCopy.baseClusterer.clone(), toCopy.clusterEvaluation.clone());
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
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        /**
         * For each current cluster, we store the clustering results if we 
         * attempt to split it into two.
         * <br>
         * Each row needs to be re-set since the clustering methods will use the length of the cluster size 
         */
        final int[][] subDesignation = new int[clusters][];
        /**
         * Stores the index from the sub data set into the full data set
         */
        final int[][] originalPositions = new int[clusters][dataSet.getSampleSize()];
        
        /**
         * Stores the dissimilarity of the splitting of the cluster with the same index value. Negative value indicates not set
         */
        final double[] splitEvaluation = new double[clusters];
        
        PriorityQueue<Integer> clusterToSplit = new PriorityQueue<Integer>(clusters, new Comparator<Integer>() {

            @Override
            public int compare(Integer t, Integer t1) 
            {
                return Double.compare(splitEvaluation[t], splitEvaluation[t1]);
            }
        });
        clusterToSplit.add(0);//We must start out spliting the one cluster of everyone!
        
        Arrays.fill(designations, 0);
        
        
        //Create initial split we will start from
        if(threadpool == null)
            baseClusterer.cluster(dataSet, 2, designations);
        else
            baseClusterer.cluster(dataSet, 2, threadpool, designations);
        subDesignation[0] = Arrays.copyOf(designations, designations.length);
        for(int i = 0; i < originalPositions[0].length; i++)
            originalPositions[0][i] = i;
        
        
        List<DataPoint> dpSubC1 = new ArrayList<DataPoint>();
        List<DataPoint> dpSubC2 = new ArrayList<DataPoint>();
        
        /*
         * TODO it could be updated to use the split value to do a search range 
         * and stop when a large jump occurs. This will perform poorl and 
         * underestimate the number of clusters, becase it will decend one path 
         * as splitting clusters improves untill it gets to the correct cluster 
         * size. Popping back up will then casue an increase, which will cause 
         * the early termination. 
         */
        for(int k = 1; k < clusters; k++)
        {
            int useSplit = clusterToSplit.poll();

            int newClusterID = k;

            dpSubC1.clear();
            dpSubC2.clear();

            //Split the data set into its two sub data sets
            for(int i = 0; i < subDesignation[useSplit].length; i++)
            {
                int origPos = originalPositions[useSplit][i];
                if(subDesignation[useSplit][i] == 0)
                {
                    dpSubC1.add(dataSet.getDataPoint(origPos));
                    continue;//We will asigng cluster '1' to be the new cluster number
                }
                dpSubC2.add(dataSet.getDataPoint(origPos));
                designations[origPos] = newClusterID;
            }

            computeSubClusterSplit(subDesignation, useSplit, dpSubC1, 
                    dataSet, designations, originalPositions, 
                    splitEvaluation, clusterToSplit, threadpool);

            computeSubClusterSplit(subDesignation, newClusterID, dpSubC2, 
                    dataSet, designations, originalPositions, 
                    splitEvaluation, clusterToSplit, threadpool);   
        }
        
        return designations;
    }

    /**
     * Takes the data set and computes the clustering of a sub cluster, and 
     * stores its information, and places the result in the queue
     * 
     * @param subDesignation the array of arrays to store the designation array 
     * for the sub clustering
     * @param originalCluster the originalCluster that we want to store the 
     * split information of
     * @param listOfDataPointsInCluster the list of all data points that belong 
     * to <tt>originalCluster</tt>
     * @param fullDataSet the full original data set
     * @param fullDesignations the designation array for the full original data 
     * set
     * @param originalPositions the array of arrays to store the map from and 
     * index in <tt>listOfDataPointsInCluster</tt> to its index in the full data
     * set. 
     * @param splitEvaluation the array to store the cluster evaluation of the
     * data set 
     * @param clusterToSplit the priority queue that stores the cluster id and 
     * sorts based on how good the sub splits were. 
     */
    private void computeSubClusterSplit(final int[][] subDesignation, 
            int originalCluster, List<DataPoint> listOfDataPointsInCluster, DataSet fullDataSet,
            int[] fullDesignations, final int[][] originalPositions, 
            final double[] splitEvaluation, 
            PriorityQueue<Integer> clusterToSplit, ExecutorService threadpool) 
    {
        subDesignation[originalCluster] = new int[listOfDataPointsInCluster.size()];
        int pos = 0;
        for(int i = 0; i < fullDataSet.getSampleSize(); i++)
        {
            if(fullDesignations[i] != originalCluster)
                continue;
            originalPositions[originalCluster][pos++] = i;
        }
        //Cluster the sub cluster
        SimpleDataSet dpSubC1DataSet = new SimpleDataSet(listOfDataPointsInCluster);
        try
        {
            if (threadpool == null)
                baseClusterer.cluster(dpSubC1DataSet, 2, subDesignation[originalCluster]);
            else
                baseClusterer.cluster(dpSubC1DataSet, 2, threadpool, subDesignation[originalCluster]);
            splitEvaluation[originalCluster] = clusterEvaluation.evaluate(subDesignation[originalCluster], dpSubC1DataSet);
            clusterToSplit.add(originalCluster);
        }
        catch (ClusterFailureException ex)
        {
            splitEvaluation[originalCluster] = Double.POSITIVE_INFINITY;
        }


    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, int[] designations)
    {
        return cluster(dataSet, clusters, null, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, lowK, threadpool, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        return cluster(dataSet, lowK, highK, null, designations);
    }

    @Override
    public DivisiveLocalClusterer clone()
    {
        return new DivisiveLocalClusterer(this);
    }
}
