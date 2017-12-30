
package jsat.linear.vectorcollection;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.utils.DoubleList;
import jsat.utils.IntList;

/**
 * A Vector Collection is a collection of vectors that is meant to be used to 
 * obtain a subset of the collection via a query vector. A query can be for 
 * the nearest neighbors, or for all vectors within a given range. 
 * <br>
 * Different vector collections have different performance properties for both training and execution time. 
 * 
 * @author Edward Raff
 * @param <V>
 */
public interface VectorCollection<V extends Vec> extends Cloneable, Serializable
{

    /**
     * Searches the space for all vectors that are within  a given range of the query vector. 
     * @param query the vector we want to find others near
     * @param range the search range around our query
     * @return the list of all vectors within the range of our query. The paired value contains the distance to the query vector. 
     * @deprecated This API is from the original JSAT interface. It will be removed in the future. 
     */
    default public List<? extends VecPaired<V, Double>> search(Vec query, double range)
    {
        IntList neighbors = new IntList();
        DoubleList distances = new DoubleList();
        search(query, range, neighbors, distances);
        List<VecPaired<V, Double>> toRet = new ArrayList<>();
        for(int i = 0; i < neighbors.size(); i++)
            toRet.add(new VecPaired<>(get(neighbors.getI(i)), distances.getD(i)));
        return toRet;
    }
    
    /**
     * Searches the space for the k neighbors that are closest to the given query vector
     * @param query the vector we want to find neighbors of
     * @param num_neighbors the maximum number of neighbors to return
     * @return the list the k nearest neighbors, in sorted order from closest to farthest. The paired value contains the distance to the query vector
     * @deprecated This API is from the original JSAT interface. It will be removed in the future. 
     */
    default public List<? extends VecPaired<V, Double>> search(Vec query, int num_neighbors)
    {
        IntList neighbors = new IntList();
        DoubleList distances = new DoubleList();
        search(query, num_neighbors, neighbors, distances);
        List<VecPaired<V, Double>> toRet = new ArrayList<>();
        for(int i = 0; i < neighbors.size(); i++)
            toRet.add(new VecPaired<>(get(neighbors.getI(i)), distances.getD(i)));
        return toRet;
    }

    /**
     * Performs a range search of the current collection. The index and distance
     * of each found neighbor will be placed into the given Lists.
     *
     * @param query the point to search for the neighbors within a given radius.
     * @param range the radius to search for all the neighbors with a distance
     * &le; range.
     * @param neighbors the list to store the index of the neighbors in. Will be
     * sorted by distance to the query, and paired with the values in
     * <tt>distances</tt>.
     * @param distances the list to store the distance of the neighbors to the
     * query in. Will be sorted, and paired with the values in
     * <tt>neighbors</tt>.
     */
    public void search(Vec query, double range, List<Integer> neighbors, List<Double> distances);

    /**
     * Performs k-Nearest Neighbor search of the current collection. The index
     * and distance of each found neighbor will be placed into the given Lists.
     *
     * @param query the point to search for the k-nearest neighbors of
     * @param numNeighbors the number of neighbors <i>k</i> to search for.
     * @param neighbors the list to store the index of the neighbors in. Will be
     * sorted by distance to the query, and paired with the values in
     * <tt>distances</tt>.
     * @param distances the list to store the distance of the neighbors to the
     * query in. Will be sorted, and paired with the values in
     * <tt>neighbors</tt>.
     */
    public void search(Vec query, int numNeighbors, List<Integer> neighbors, List<Double> distances);
    
    /**
     * Accesses a vector from this collection via index. 
     * @param indx the index in [0, {@link #size() }) of the vector to access
     * @return the vector from the collection 
     */
    public V get(int indx);
    
    /**
     * Returns the number of vectors stored in the collection
     * @return the size of the collection
     */
    public int size();
    
    public VectorCollection<V> clone();
}
