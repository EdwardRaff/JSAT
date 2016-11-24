
package jsat.clustering;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;

/**
 * A base foundation that provides an implementation of {@link #cluster(jsat.DataSet) } 
 * and {@link #cluster(jsat.DataSet, java.util.concurrent.ExecutorService) } using 
 * their int array counterparts. <br>
 * <br>
 * By default it is assumed that a cluster does not support weighted data. If
 * this is incorrect, you need to overwrite the {@link #supportsWeightedData() }
 * method.
 *
 * @author Edward Raff
 */
public abstract class ClustererBase implements Clusterer
{

    private static final long serialVersionUID = 4359554809306681680L;

    @Override
    public List<List<DataPoint>> cluster(DataSet dataSet)
    {
        int[] assignments = cluster(dataSet, (int[]) null);
        
        return createClusterListFromAssignmentArray(assignments, dataSet);
    }

    @Override
    public List<List<DataPoint>> cluster(DataSet dataSet, ExecutorService threadpool)
    {
        int[] assignments = cluster(dataSet, threadpool, (int[]) null);
        return createClusterListFromAssignmentArray(assignments, dataSet);
    }

    /**
     * Convenient helper method. A list of lists to represent a cluster may be desirable. In 
     * such a case, this method will take in an array of cluster assignments, and return a 
     * list of lists. 
     * 
     * @param assignments the array containing cluster assignments
     * @param dataSet the original data set, with data in the same order as was used to create the assignments array
     * @return a List of lists where each list contains the data points for one cluster, and the lists are in order by cluster id. 
     */
    public static List<List<DataPoint>> createClusterListFromAssignmentArray(int[] assignments, DataSet dataSet)
    {
        List<List<DataPoint>> clusterings = new ArrayList<List<DataPoint>>();
        
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            while(clusterings.size() <= assignments[i])
                clusterings.add(new ArrayList<DataPoint>());
            if(assignments[i] >= 0)
                clusterings.get(assignments[i]).add(dataSet.getDataPoint(i));
        }
        
        return clusterings;
    }
    
    /**
     * Gets a list of the datapoints in a data set that belong to the indicated cluster
     * @param c the cluster ID to get the datapoints for
     * @param assignments the array containing cluster assignments
     * @param dataSet the data set to get the points from
     * @param indexFrom stores the index from the original dataset that the 
     * datapoint is from, such that the item at index {@code i} in the returned 
     * list can be found in the original dataset at index {@code indexFrom[i]}. 
     * May be {@code null}
     * @return a list of datapoints that were assignment to the designated cluster
     */
    public static List<DataPoint> getDatapointsFromCluster(int c, int[] assignments, DataSet dataSet, int[] indexFrom)
    {
        List<DataPoint> list = new ArrayList<DataPoint>();
        int pos = 0;
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            if(assignments[i] == c)
            {
                list.add(dataSet.getDataPoint(i));
                if(indexFrom != null)
                    indexFrom[pos++] = i;
            }
        return list;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    abstract public Clusterer clone();
    
}
