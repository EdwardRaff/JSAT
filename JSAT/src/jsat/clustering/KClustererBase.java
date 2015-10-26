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
    public List<List<DataPoint>> cluster(final DataSet dataSet, final int clusters, final ExecutorService threadpool)
    {
        final int[] assignments = cluster(dataSet, clusters, threadpool, (int[]) null);
        return createClusterListFromAssignmentArray(assignments, dataSet);
    }

    @Override
    public List<List<DataPoint>> cluster(final DataSet dataSet, final int clusters)
    {
        final int[] assignments = cluster(dataSet, clusters, (int[]) null);
        return createClusterListFromAssignmentArray(assignments, dataSet);
    }


    @Override
    public List<List<DataPoint>> cluster(final DataSet dataSet, final int lowK, final int highK, final ExecutorService threadpool)
    {
        final int[] assignments = cluster(dataSet, lowK, highK, threadpool, (int[]) null);
        return createClusterListFromAssignmentArray(assignments, dataSet);
    }

    @Override
    public List<List<DataPoint>> cluster(final DataSet dataSet, final int lowK, final int highK)
    {
        final int[] assignments = cluster(dataSet, lowK, highK, (int[]) null);
        return createClusterListFromAssignmentArray(assignments, dataSet);
    }
    
    @Override
    abstract public KClusterer clone();
}
