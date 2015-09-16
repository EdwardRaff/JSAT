package jsat.linear.vectorcollection;

import static jsat.utils.SystemInfo.LogicalCores;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.math.OnLineStatistics;
import jsat.utils.ListUtils;

/**
 * A collection of common utility methods to perform on a
 * {@link VectorCollection}
 *
 * @author Edward Raff
 */
public class VectorCollectionUtils {

  /**
   * Searches the given collection for the <tt>k</tt> nearest neighbors for
   * every data point in the given search list.
   *
   * @param <V0>
   *          the vector type in the collection
   * @param <V1>
   *          the type of vector in the search collection
   * @param collection
   *          the collection to search from
   * @param search
   *          the vectors to search for
   * @param k
   *          the number of nearest neighbors
   * @return The list of lists for all nearest neighbors
   */
  public static <V0 extends Vec, V1 extends Vec> List<List<? extends VecPaired<V0, Double>>> allNearestNeighbors(
      final VectorCollection<V0> collection, final List<V1> search, final int k) {
    final List<List<? extends VecPaired<V0, Double>>> results = new ArrayList<List<? extends VecPaired<V0, Double>>>(
        search.size());
    for (final Vec v : search) {
      results.add(collection.search(v, k));
    }
    return results;
  }

  /**
   * Searches the given collection for the <tt>k</tt> nearest neighbors for
   * every data point in the given search list.
   *
   * @param <V0>
   *          the vector type in the collection
   * @param <V1>
   *          the type of vector in the search collection
   * @param collection
   *          the collection to search from
   * @param search
   *          the vectors to search for
   * @param k
   *          the number of nearest neighbors
   * @param threadpool
   *          the source of threads to perform the computation in parallel
   * @return The list of lists for all nearest neighbors
   */
  public static <V0 extends Vec, V1 extends Vec> List<List<? extends VecPaired<V0, Double>>> allNearestNeighbors(
      final VectorCollection<V0> collection, final List<V1> search, final int k, final ExecutorService threadpool)
          throws InterruptedException, ExecutionException {
    final List<List<? extends VecPaired<V0, Double>>> results = new ArrayList<List<? extends VecPaired<V0, Double>>>(
        search.size());
    final List<Future<List<List<? extends VecPaired<V0, Double>>>>> subResults = new ArrayList<Future<List<List<? extends VecPaired<V0, Double>>>>>(
        LogicalCores);

    for (final List<V1> subSearch : ListUtils.splitList(search, LogicalCores)) {
      subResults.add(threadpool.submit(new Callable<List<List<? extends VecPaired<V0, Double>>>>() {

        @Override
        public List<List<? extends VecPaired<V0, Double>>> call() throws Exception {
          final List<List<? extends VecPaired<V0, Double>>> subResult = new ArrayList<List<? extends VecPaired<V0, Double>>>(
              subSearch.size());

          for (final Vec v : subSearch) {
            subResult.add(collection.search(v, k));
          }

          return subResult;
        }
      }));
    }

    for (final List<List<? extends VecPaired<V0, Double>>> subResult : ListUtils.collectFutures(subResults)) {
      results.addAll(subResult);
    }

    return results;
  }

  /**
   * Searches the given collection for the <tt>k</tt> nearest neighbors for
   * every data point in the given search list.
   *
   * @param <V0>
   *          the vector type in the collection
   * @param <V1>
   *          the type of vector in the search array
   * @param collection
   *          the collection to search from
   * @param search
   *          the vectors to search for
   * @param k
   *          the number of nearest neighbors
   * @return The list of lists for all nearest neighbors
   */
  public static <V0 extends Vec, V1 extends Vec> List<List<? extends VecPaired<V0, Double>>> allNearestNeighbors(
      final VectorCollection<V0> collection, final V1[] search, final int k) {
    return allNearestNeighbors(collection, Arrays.asList(search), k);
  }

  /**
   * Searches the given collection for the <tt>k</tt> nearest neighbors for
   * every data point in the given search list.
   *
   * @param <V0>
   *          the vector type in the collection
   * @param <V1>
   *          the type of vector in the search collection
   * @param collection
   *          the collection to search from
   * @param search
   *          the vectors to search for
   * @param k
   *          the number of nearest neighbors
   * @param threadpool
   *          the source of threads to perform the computation in parallel
   * @return The list of lists for all nearest neighbors
   */
  public static <V0 extends Vec, V1 extends Vec> List<List<? extends VecPaired<V0, Double>>> allNearestNeighbors(
      final VectorCollection<V0> collection, final V1[] search, final int k, final ExecutorService threadpool)
          throws InterruptedException, ExecutionException {
    return allNearestNeighbors(collection, Arrays.asList(search), k, threadpool);
  }

  /**
   * Computes statistics about the distance of the k'th nearest neighbor for
   * each data point in the <tt>search</tt> list.
   *
   * @param <V0>
   *          the type of vector in the collection
   * @param <V1>
   *          the type of vector in the search collection
   * @param collection
   *          the collection of vectors to query from
   * @param search
   *          the list of vectors to search for
   * @param k
   *          the nearest neighbor to use
   * @return the statistics for the distance of the k'th nearest neighbor from
   *         the query point
   */
  public static <V0 extends Vec, V1 extends Vec> OnLineStatistics getKthNeighborStats(
      final VectorCollection<V0> collection, final List<V1> search, final int k) {
    final OnLineStatistics stats = new OnLineStatistics();
    for (final Vec v : search) {
      stats.add(collection.search(v, k).get(k - 1).getPair());
    }

    return stats;
  }

  /**
   * Computes statistics about the distance of the k'th nearest neighbor for
   * each data point in the <tt>search</tt> list.
   *
   * @param <V0>
   *          the type of vector in the collection
   * @param <V1>
   *          the type of vector in the search collection
   * @param collection
   *          the collection of vectors to query from
   * @param search
   *          the list of vectors to search for
   * @param k
   *          the nearest neighbor to use
   * @param threadpool
   *          the source of threads to perform the computation in parallel
   * @return the statistics for the distance of the k'th nearest neighbor from
   *         the query point
   */
  public static <V0 extends Vec, V1 extends Vec> OnLineStatistics getKthNeighborStats(
      final VectorCollection<V0> collection, final List<V1> search, final int k, final ExecutorService threadpool)
          throws InterruptedException, ExecutionException {
    final List<Future<OnLineStatistics>> futureStats = new ArrayList<Future<OnLineStatistics>>(LogicalCores);

    for (final List<V1> subSearch : ListUtils.splitList(search, LogicalCores)) {
      futureStats.add(threadpool.submit(new Callable<OnLineStatistics>() {

        @Override
        public OnLineStatistics call() throws Exception {
          final OnLineStatistics stats = new OnLineStatistics();

          for (final Vec v : subSearch) {
            stats.add(collection.search(v, k).get(k - 1).getPair());
          }

          return stats;
        }
      }));
    }

    OnLineStatistics stats = new OnLineStatistics();
    for (final OnLineStatistics subResult : ListUtils.collectFutures(futureStats)) {
      stats = OnLineStatistics.add(stats, subResult);
    }

    return stats;
  }

  /**
   * Computes statistics about the distance of the k'th nearest neighbor for
   * each data point in the <tt>search</tt> list.
   *
   * @param <V0>
   *          the type of vector in the collection
   * @param <V1>
   *          the type of vector in the search array
   * @param collection
   *          the collection of vectors to query from
   * @param search
   *          the array of vectors to search for
   * @param k
   *          the nearest neighbor to use
   * @return the statistics for the distance of the k'th nearest neighbor from
   *         the query point
   */
  public static <V0 extends Vec, V1 extends Vec> OnLineStatistics getKthNeighborStats(
      final VectorCollection<V0> collection, final V1[] search, final int k) {
    return getKthNeighborStats(collection, Arrays.asList(search), k);
  }

  /**
   * Computes statistics about the distance of the k'th nearest neighbor for
   * each data point in the <tt>search</tt> list.
   *
   * @param <V0>
   *          the type of vector in the collection
   * @param <V1>
   *          the type of vector in the search array
   * @param collection
   *          the collection of vectors to query from
   * @param search
   *          the array of vectors to search for
   * @param k
   *          the nearest neighbor to use
   * @param threadpool
   *          the source of threads to perform the computation in parallel
   * @return the statistics for the distance of the k'th nearest neighbor from
   *         the query point
   */
  public static <V0 extends Vec, V1 extends Vec> OnLineStatistics getKthNeighborStats(
      final VectorCollection<V0> collection, final V1[] search, final int k, final ExecutorService threadpool)
          throws InterruptedException, ExecutionException {
    return getKthNeighborStats(collection, Arrays.asList(search), k, threadpool);
  }
}
