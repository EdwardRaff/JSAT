package jsat.clustering;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;

/**
 * A base foundation that provides an implementation of the methods that return a list of lists for the clusterings using 
 * their int array counterparts. 
 * @author Edward Raff
 */
public abstract class KClustererBase extends ClustererBase implements KClusterer
{

	private static final long serialVersionUID = 2542432122353325407L;

	@Override
    public List<List<DataPoint>> cluster(DataSet dataSet, int clusters, ExecutorService threadpool)
    {
        int[] assignments = cluster(dataSet, clusters, threadpool, (int[]) null);
        return createClusterListFromAssignmentArray(assignments, dataSet);
    }

    @Override
    public List<List<DataPoint>> cluster(DataSet dataSet, int clusters)
    {
        int[] assignments = cluster(dataSet, clusters, (int[]) null);
        return createClusterListFromAssignmentArray(assignments, dataSet);
    }


    @Override
    public List<List<DataPoint>> cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool)
    {
        int[] assignments = cluster(dataSet, lowK, highK, threadpool, (int[]) null);
        return createClusterListFromAssignmentArray(assignments, dataSet);
    }

    @Override
    public List<List<DataPoint>> cluster(DataSet dataSet, int lowK, int highK)
    {
        int[] assignments = cluster(dataSet, lowK, highK, (int[]) null);
        return createClusterListFromAssignmentArray(assignments, dataSet);
    }
    
    @Override
    abstract public KClusterer clone();
}
