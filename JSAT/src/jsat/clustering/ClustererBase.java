
package jsat.clustering;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;

/**
 * A base foundation that provides an implementation of {@link #cluster(jsat.DataSet) } 
 * and {@link #cluster(jsat.DataSet, java.util.concurrent.ExecutorService) } using 
 * their int array counterparts. 
 * 
 * @author Edward Raff
 */
public abstract class ClustererBase implements Clusterer
{

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
    
}
