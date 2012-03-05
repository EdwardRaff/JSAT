
package jsat.linear.vectorcollection;

import jsat.math.OnLineStatistics;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.utils.ListUtils;
import static jsat.utils.SystemInfo.*;

/**
 * A collection of common utility methods to perform on a {@link VectorCollection}
 * 
 * @author Edward Raff
 */
public class VectorCollectionUtils
{
    /**
     * Searches the given collection for the <tt>k</tt> nearest neighbors for every data point in the given search list. 
     * @param <V> the vector type
     * @param collection the collection to search from
     * @param search the vectors to search for
     * @param k the number of nearest neighbors
     * @return The list of lists for all nearest neighbors 
     */
    public static <V extends Vec> List<List<VecPaired<Double, V>>> allNearestNeighbors(VectorCollection<V> collection, List<Vec> search, int k)
    {
        List<List<VecPaired<Double, V>>> results = new ArrayList<List<VecPaired<Double, V>>>(search.size());
        for(Vec v : search)
            results.add(collection.search(v, k));
        return results;
    }
    
    /**
     * Searches the given collection for the <tt>k</tt> nearest neighbors for every data point in the given search list. 
     * @param <V> the vector type
     * @param collection the collection to search from
     * @param search the vectors to search for
     * @param k the number of nearest neighbors
     * @param threadpool the source of threads to perform the computation in parallel 
     * @return The list of lists for all nearest neighbors 
     */
    public static <V extends Vec> List<List<VecPaired<Double, V>>> allNearestNeighbors(final VectorCollection<V> collection, List<Vec> search, final int k, ExecutorService threadpool)
             throws InterruptedException, ExecutionException
    {
        List<List<VecPaired<Double, V>>> results = new ArrayList<List<VecPaired<Double, V>>>(search.size());
        List<Future<List<List<VecPaired<Double, V>>>>> subResults = new ArrayList<Future<List<List<VecPaired<Double, V>>>>>(LogicalCores);
        
        for(final List<Vec> subSearch : ListUtils.splitList(search, LogicalCores))
        {
            subResults.add(threadpool.submit(new Callable<List<List<VecPaired<Double, V>>>>() {

                public List<List<VecPaired<Double, V>>> call() throws Exception
                {
                    List<List<VecPaired<Double, V>>> subResult = new ArrayList<List<VecPaired<Double, V>>>(subSearch.size());
                    
                    for(Vec v : subSearch )
                        subResult.add(collection.search(v, k));
                    
                    return subResult;
                }
            }));
        }

        for (List<List<VecPaired<Double, V>>> subResult : ListUtils.collectFutures(subResults))
            results.addAll(subResult);

        return results;
    }

    /**
     * Computes statistics about the distance of the k'th nearest neighbor for each data point in the <tt>search</tt> list. 
     * 
     * @param <V> the type of vector in the collection 
     * @param collection the collection of vectors to query from
     * @param search the list of vectors to search for
     * @param k the nearest neighbor to use
     * @return the statistics for the distance of the k'th nearest neighbor from the query point
     */
    public static <V extends Vec> OnLineStatistics getKthNeighborStats(VectorCollection<V> collection, List<Vec> search, int k)
    {
        OnLineStatistics stats = new OnLineStatistics();
        for(Vec v : search)
            stats.add(collection.search(v, k).get(k-1).getPair());
        
        return stats;
    }
    
    /**
     * Computes statistics about the distance of the k'th nearest neighbor for each data point in the <tt>search</tt> list. 
     * 
     * @param <V> the type of vector in the collection 
     * @param collection the collection of vectors to query from
     * @param search the list of vectors to search for
     * @param k the nearest neighbor to use
     * @param threadpool the source of threads to perform the computation in parallel 
     * @return the statistics for the distance of the k'th nearest neighbor from the query point
     */
    public static <V extends Vec> OnLineStatistics getKthNeighborStats(final VectorCollection<V> collection, List<Vec> search, final int k, ExecutorService threadpool)
            throws InterruptedException, ExecutionException
    {
        List<Future<OnLineStatistics>> futureStats = new ArrayList<Future<OnLineStatistics>>(LogicalCores);
        
        for(final List<Vec> subSearch : ListUtils.splitList(search, LogicalCores))
        {
            futureStats.add(threadpool.submit(new Callable<OnLineStatistics>() {

                public OnLineStatistics call() throws Exception
                {
                    OnLineStatistics stats = new OnLineStatistics();
                    
                    for(Vec v: subSearch)
                        stats.add(collection.search(v, k).get(k-1).getPair());
                    
                    return stats;
                }
            }));
        }

        OnLineStatistics stats = new OnLineStatistics();
        for (OnLineStatistics subResult : ListUtils.collectFutures(futureStats))
            stats = OnLineStatistics.add(stats, subResult);

        return stats;
    }
}
