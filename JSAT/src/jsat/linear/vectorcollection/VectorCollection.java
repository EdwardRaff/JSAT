
package jsat.linear.vectorcollection;

import java.io.Serializable;
import java.util.List;
import jsat.linear.Vec;
import jsat.linear.VecPaired;

/**
 * A Vector Collection is a collection of vectors that is meant to be used to 
 * obtain a subset of the collection via a query vector. A query can be for 
 * the nearest neighbors, or for all vectors within a given range. 
 * <br>
 * Different vector collections have different performance properties for both training and execution time. 
 * 
 * @author Edward Raff
 */
public interface VectorCollection<V extends Vec> extends Cloneable, Serializable
{

    /**
     * Searches the space for all vectors that are within  a given range of the query vector. 
     * @param query the vector we want to find others near
     * @param range the search range around our query
     * @return the list of all vectors within the range of our query. The paired value contains the distance to the query vector. 
     */
    public List<? extends VecPaired<V, Double>> search(Vec query, double range);
    
    /**
     * Searches the space for the k neighbors that are closest to the given query vector
     * @param query the vector we want to find neighbors of
     * @param neighbors the maximum number of neighbors to return
     * @return the list the k nearest neighbors, in sorted order from closest to farthest. The paired value contains the distance to the query vector
     */
    public List<? extends VecPaired<V, Double>> search(Vec query, int neighbors);
    
    /**
     * Returns the number of vectors stored in the collection
     * @return the size of the collection
     */
    public int size();
    
    public VectorCollection<V> clone();
}
