
package jsat.linear.vectorcollection;

import java.util.List;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;

/**
 *
 * @author Edward Raff
 */
public interface VectorCollectionFactory<V extends Vec>
{
    public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric);
}
