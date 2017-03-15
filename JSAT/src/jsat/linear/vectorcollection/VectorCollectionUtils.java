
package jsat.linear.vectorcollection;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.math.OnLineStatistics;
import jsat.utils.ListUtils;
import static jsat.utils.SystemInfo.LogicalCores;

/**
 * A collection of common utility methods to perform on a {@link VectorCollection}
 * 
 * @author Edward Raff
 */
public class VectorCollectionUtils
{
    /**
     * Searches the given collection for the <tt>k</tt> nearest neighbors for every data point in the given search list. 
     * @param <V0> the vector type in the collection
     * @param <V1> the type of vector in the search collection 
     * @param collection the collection to search from
     * @param search the vectors to search for
     * @param k the number of nearest neighbors
     * @return The list of lists for all nearest neighbors 
     */
    public static <V0 extends Vec, V1 extends Vec> List<List<? extends VecPaired<V0, Double>>> allNearestNeighbors(VectorCollection<V0> collection, List<V1> search, int k)
    {
        List<List<? extends VecPaired<V0, Double>>> results = new ArrayList<List<? extends VecPaired<V0, Double>>>(search.size());
        for(Vec v : search)
            results.add(collection.search(v, k));
        return results;
    }
    
    /**
     * Searches the given collection for the <tt>k</tt> nearest neighbors for every data point in the given search list. 
     * @param <V0> the vector type in the collection
     * @param <V1> the type of vector in the search array 
     * @param collection the collection to search from
     * @param search the vectors to search for
     * @param k the number of nearest neighbors
     * @return The list of lists for all nearest neighbors 
     */
    public static <V0 extends Vec, V1 extends Vec> List<List<? extends VecPaired<V0, Double>>> allNearestNeighbors(VectorCollection<V0> collection, V1[] search, int k)
    {
        return allNearestNeighbors(collection, Arrays.asList(search), k);
    }
    
    /**
     * Searches the given collection for the <tt>k</tt> nearest neighbors for every data point in the given search list. 
     * @param <V0> the vector type in the collection
     * @param <V1> the type of vector in the search collection 
     * @param collection the collection to search from
     * @param search the vectors to search for
     * @param k the number of nearest neighbors
     * @param threadpool the source of threads to perform the computation in parallel 
     * @return The list of lists for all nearest neighbors 
     */
    public static <V0 extends Vec, V1 extends Vec> List<List<? extends VecPaired<V0, Double>>> allNearestNeighbors(final VectorCollection<V0> collection, List<V1> search, final int k, ExecutorService threadpool)
    {
        List<List<? extends VecPaired<V0, Double>>> results = new ArrayList<List<? extends VecPaired<V0, Double>>>(search.size());
        List<Future<List<List<? extends VecPaired<V0, Double>>>>> subResults = new ArrayList<Future<List<List<? extends VecPaired<V0, Double>>>>>(LogicalCores);
        
        for(final List<V1> subSearch : ListUtils.splitList(search, LogicalCores))
        {
            subResults.add(threadpool.submit(new Callable<List<List<? extends VecPaired<V0, Double>>>>() {

                @Override
                public List<List<? extends VecPaired<V0, Double>>> call() throws Exception
                {
                    List<List<? extends VecPaired<V0, Double>>> subResult = new ArrayList<List<? extends VecPaired<V0, Double>>>(subSearch.size());
                    
                    for(Vec v : subSearch )
                        subResult.add(collection.search(v, k));
                    
                    return subResult;
                }
            }));
        }

        try
        {
            for (List<List<? extends VecPaired<V0, Double>>> subResult : ListUtils.collectFutures(subResults))
                results.addAll(subResult);
        }
        catch (ExecutionException ex)
        {
            Logger.getLogger(VectorCollectionUtils.class.getName()).log(Level.SEVERE, null, ex);
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(VectorCollectionUtils.class.getName()).log(Level.SEVERE, null, ex);
        }

        return results;
    }
    
    /**
     * Searches the given collection for all the neighbors within a distance of <tt>radius</tt> for every data point in the given search list. 
     * @param <V0> the vector type in the collection
     * @param <V1> the type of vector in the search collection 
     * @param collection the collection to search from
     * @param search the vectors to search for
     * @param radius the distance to search for neighbors 
     * @param threadpool the source of threads to perform the computation in parallel 
     * @return The list of lists for all nearest neighbors 
     */
    public static <V0 extends Vec, V1 extends Vec> List<List<? extends VecPaired<V0, Double>>> allEpsNeighbors(final VectorCollection<V0> collection, List<V1> search, final double radius, ExecutorService threadpool)
    {
        List<List<? extends VecPaired<V0, Double>>> results = new ArrayList<List<? extends VecPaired<V0, Double>>>(search.size());
        List<Future<List<List<? extends VecPaired<V0, Double>>>>> subResults = new ArrayList<Future<List<List<? extends VecPaired<V0, Double>>>>>(LogicalCores);
        
        for(final List<V1> subSearch : ListUtils.splitList(search, LogicalCores))
        {
            subResults.add(threadpool.submit(new Callable<List<List<? extends VecPaired<V0, Double>>>>() {

                @Override
                public List<List<? extends VecPaired<V0, Double>>> call() throws Exception
                {
                    List<List<? extends VecPaired<V0, Double>>> subResult = new ArrayList<List<? extends VecPaired<V0, Double>>>(subSearch.size());
                    
                    for(Vec v : subSearch )
                        subResult.add(collection.search(v, radius));
                    
                    return subResult;
                }
            }));
        }

        try
        {
            for (List<List<? extends VecPaired<V0, Double>>> subResult : ListUtils.collectFutures(subResults))
                results.addAll(subResult);
        }
        catch (ExecutionException ex)
        {
            Logger.getLogger(VectorCollectionUtils.class.getName()).log(Level.SEVERE, null, ex);
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(VectorCollectionUtils.class.getName()).log(Level.SEVERE, null, ex);
        }

        return results;
    }
    
    /**
     * Searches the given collection for the <tt>k</tt> nearest neighbors for every data point in the given search list. 
     * @param <V0> the vector type in the collection
     * @param <V1> the type of vector in the search collection 
     * @param collection the collection to search from
     * @param search the vectors to search for
     * @param k the number of nearest neighbors
     * @param threadpool the source of threads to perform the computation in parallel 
     * @return The list of lists for all nearest neighbors 
     */
    public static <V0 extends Vec, V1 extends Vec> List<List<? extends VecPaired<V0, Double>>> allNearestNeighbors(final VectorCollection<V0> collection, V1[] search, final int k, ExecutorService threadpool)
    {
        return allNearestNeighbors(collection, Arrays.asList(search), k, threadpool);
    }

    /**
     * Computes statistics about the distance of the k'th nearest neighbor for each data point in the <tt>search</tt> list. 
     * 
     * @param <V0> the type of vector in the collection 
     * @param <V1> the type of vector in the search collection 
     * @param collection the collection of vectors to query from
     * @param search the list of vectors to search for
     * @param k the nearest neighbor to use
     * @return the statistics for the distance of the k'th nearest neighbor from the query point
     */
    public static <V0 extends Vec, V1 extends Vec> OnLineStatistics getKthNeighborStats(VectorCollection<V0> collection, List<V1> search, int k)
    {
        OnLineStatistics stats = new OnLineStatistics();
        for(Vec v : search)
            stats.add(collection.search(v, k).get(k-1).getPair());
        
        return stats;
    }
    
    /**
     * Computes statistics about the distance of the k'th nearest neighbor for each data point in the <tt>search</tt> list. 
     * 
     * @param <V0> the type of vector in the collection 
     * @param <V1> the type of vector in the search array 
     * @param collection the collection of vectors to query from
     * @param search the array of vectors to search for
     * @param k the nearest neighbor to use
     * @return the statistics for the distance of the k'th nearest neighbor from the query point
     */
    public static <V0 extends Vec, V1 extends Vec> OnLineStatistics getKthNeighborStats(VectorCollection<V0> collection, V1[] search, int k)
    {
        return getKthNeighborStats(collection, Arrays.asList(search), k);
    }
    
    /**
     * Computes statistics about the distance of the k'th nearest neighbor for each data point in the <tt>search</tt> list. 
     * 
     * @param <V0> the type of vector in the collection 
     * @param <V1> the type of vector in the search collection 
     * @param collection the collection of vectors to query from
     * @param search the list of vectors to search for
     * @param k the nearest neighbor to use
     * @param threadpool the source of threads to perform the computation in parallel 
     * @return the statistics for the distance of the k'th nearest neighbor from the query point
     */
    public static <V0 extends Vec, V1 extends Vec> OnLineStatistics getKthNeighborStats(final VectorCollection<V0> collection, List<V1> search, final int k, ExecutorService threadpool)
    {
        List<Future<OnLineStatistics>> futureStats = new ArrayList<Future<OnLineStatistics>>(LogicalCores);
        
        for(final List<V1> subSearch : ListUtils.splitList(search, LogicalCores))
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
        try
        {
            for (OnLineStatistics subResult : ListUtils.collectFutures(futureStats))
                stats = OnLineStatistics.add(stats, subResult);
        }
        catch (ExecutionException ex)
        {
            Logger.getLogger(VectorCollectionUtils.class.getName()).log(Level.SEVERE, null, ex);
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(VectorCollectionUtils.class.getName()).log(Level.SEVERE, null, ex);
        }

        return stats;
    }
    
    /**
     * Computes statistics about the distance of the k'th nearest neighbor for each data point in the <tt>search</tt> list. 
     * 
     * @param <V0> the type of vector in the collection 
     * @param <V1> the type of vector in the search array 
     * @param collection the collection of vectors to query from
     * @param search the array of vectors to search for
     * @param k the nearest neighbor to use
     * @param threadpool the source of threads to perform the computation in parallel 
     * @return the statistics for the distance of the k'th nearest neighbor from the query point
     */
    public static <V0 extends Vec, V1 extends Vec> OnLineStatistics getKthNeighborStats(final VectorCollection<V0> collection, V1[] search, final int k, ExecutorService threadpool)
    {
        return getKthNeighborStats(collection, Arrays.asList(search), k, threadpool);
    }
}
