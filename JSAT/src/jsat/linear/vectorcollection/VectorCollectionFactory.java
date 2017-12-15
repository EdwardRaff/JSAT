
package jsat.linear.vectorcollection;

import java.io.Serializable;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.utils.concurrent.ParallelUtils;

/**
 * A factory interface for the creation of {@link VectorCollection} objects. 
 * 
 * @author Edward Raff
 */
public interface VectorCollectionFactory<V extends Vec> extends Cloneable, Serializable
{
    /**
     * Creates a new Vector Collection from the given source using the provided metric. 
     * 
     * @param source the list of vectors to put into the collection. 
     * @param distanceMetric the distance measure to use for the space
     * @return a new vector collection
     */
    public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric);
    
    /**
     * Creates a new Vector Collection from the given source using the provided metric. 
     * 
     * @param source the list of vectors to put into the collection. 
     * @param distanceMetric the distance measure to use for the space
     * @param threadpool the source for threads for multi threaded execution 
     * @return  a new vector collection
     */
    public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric, ExecutorService threadpool);
    
    default public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric, boolean parallel)
    {
        //TODO this will be changed, but using this for now to make refactoring easier
        if(parallel)
        {
            ExecutorService threadpool = ParallelUtils.getNewExecutor(parallel);
            try
            {
                return getVectorCollection(source, distanceMetric, threadpool);
            }
            finally
            {
                threadpool.shutdownNow();
            }
        }
        else
            return getVectorCollection(source, distanceMetric);
    }
    
    public VectorCollectionFactory<V> clone();
}
