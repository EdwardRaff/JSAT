
package jsat.linear.vectorcollection;

import java.io.Serializable;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;

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
    
    public VectorCollectionFactory<V> clone();
}
