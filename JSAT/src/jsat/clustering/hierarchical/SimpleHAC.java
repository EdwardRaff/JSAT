package jsat.clustering.hierarchical;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.clustering.KClustererBase;
import static jsat.clustering.dissimilarity.AbstractClusterDissimilarity.createDistanceMatrix;
import jsat.clustering.dissimilarity.ClusterDissimilarity;
import jsat.math.OnLineStatistics;
import jsat.utils.IntSet;

/**
 * Provides a naive implementation of hierarchical agglomerative clustering 
 * (HAC). This means the clustering is built from the bottom up, merging points 
 * into clusters. HAC clustering is deterministic. The naive implementation runs
 * in O(n<sup>3</sup>) time. <br>
 * <br>
 * NOTE: This implementation does not currently support parallel clustering. 
 *
 * 
 * @author Edward Raff
 */
public class SimpleHAC extends KClustererBase
{

	private static final long serialVersionUID = 7138073766768205530L;
	/**
     * notion behind the large stnd devs is that as the clustering progresses, 
     * the min value is (usually) monotonically rising, so we would like a 
     * bigger jump
     */
    private double stndDevs = 3.5;
    private final ClusterDissimilarity dissMeasure;

    public SimpleHAC(final ClusterDissimilarity disMeasure)
    {
        this.dissMeasure = disMeasure;
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public SimpleHAC(final SimpleHAC toCopy)
    {
        this(toCopy.dissMeasure.clone());
        this.stndDevs = toCopy.stndDevs;
    }
    
    @Override
    public int[] cluster(final DataSet dataSet, final int[] designations)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()), designations);
    }

    @Override
    public int[] cluster(final DataSet dataSet, final ExecutorService threadpool, final int[] designations)
    {
        return cluster(dataSet, designations);
    }

    @Override
    public int[] cluster(final DataSet dataSet, final int clusters, final ExecutorService threadpool, final int[] designations)
    {
        return cluster(dataSet, clusters, designations);
    }

    @Override
    public int[] cluster(final DataSet dataSet, final int clusters, final int[] designations)
    {
        return cluster(dataSet, clusters, clusters, designations);
    }

    @Override
    public int[] cluster(final DataSet dataSet, final int lowK, final int highK, final ExecutorService threadpool, final int[] designations)
    {
        return cluster(dataSet, lowK, highK, designations);
    }

    @Override
    public int[] cluster(final DataSet dataSet, final int lowK, final int highK, int[] designations)
    {
        if(designations == null) {
          designations = new int[dataSet.getSampleSize()];
        }
        
        //Keep track of the average dis when merging, stop when it becomes abnormaly large
        final OnLineStatistics disChange = new OnLineStatistics();
        
        //Represent each cluster by a set of indices, intialy each data point is its own cluster
        final List<Set<Integer>> clusters = new ArrayList<Set<Integer>>(dataSet.getSampleSize());
        for(int i =0; i < dataSet.getSampleSize(); i++)
        {
            final Set<Integer> set = new IntSet();
            set.add(i);
            clusters.add(set);
        }
        
        final double[][] distanceMatrix = createDistanceMatrix(dataSet, dissMeasure);
        
        while( clusters.size() > lowK)
        {
            double lowestDiss = Double.MAX_VALUE, tmp;
            int a = 0, b = 1;
            //N^2 search for the most similar pairing of clusters
            for(int i = 0; i < clusters.size(); i++) {
              for(int j = i+1; j < clusters.size(); j++)
              {
                if( (tmp = dissMeasure.dissimilarity(clusters.get(i), clusters.get(j), distanceMatrix)) < lowestDiss)
                {
                  lowestDiss = tmp;
                  a = i;
                  b = j;
                }
              }
            }
            
            if(clusters.size() <= highK)//Then we check if we should stop early
            {
                if(disChange.getMean() + disChange.getStandardDeviation() * stndDevs < lowestDiss) {
                  break;//Abnormaly large difference, we assume we are forcing two real & sperate clusters into one group
                }
            }
            
            disChange.add(lowestDiss);
            
            //Merge clusters , a < b is gaurenteed by the loop structure
            clusters.get(a).addAll(clusters.remove(b));
        }
        
        //Now that we have the assigments, we must set them
        int curClusterID = 0;
        for(final Set<Integer> clustering : clusters)
        {
            for(final int index : clustering) {
              designations[index] = curClusterID;
            }
            curClusterID++;
        }
        
        return designations;
    }

    @Override
    public SimpleHAC clone()
    {
        return new SimpleHAC(this);
    }
    
}
