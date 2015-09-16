package jsat.clustering;

import static jsat.utils.SystemInfo.LogicalCores;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.linear.MatrixStatistics;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.FakeExecutor;
import jsat.utils.IndexTable;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;
import jsat.utils.concurrent.ParallelUtils;

/**
 * This class provides methods for sampling a data set for a set of initial
 * points to act as the seeds for a clustering algorithm.
 *
 * @author Edward Raff
 */
public class SeedSelectionMethods {

  static public enum SeedSelection {
    /**
     * The seed values will be randomly selected from the data set
     */
    RANDOM, /**
             * The k-means++ seeding algo: <br>
             * The seed values will be probabilistically selected from the data
             * set. <br>
             * The solution is O(log(k)) competitive with the optimal k
             * clustering when using {@link EuclideanDistance}. <br>
             * <br>
             * See k-means++: The Advantages of Careful Seeding
             */
    KPP, /**
          * The first seed is chosen randomly, and then all others are chosen to
          * be the farthest away from all other seeds
          */
    FARTHEST_FIRST, /**
                     * Selects the seeds in one pass by selecting points as
                     * evenly distributed quantiles for the distance of each
                     * point from the mean of the whole data set. This makes the
                     * seed selection deterministic <br>
                     * <br>
                     * See: J. A. Hartigan and M. A. Wong,
                     * "A k-means clustering algorithm", Applied Statistics,
                     * vol. 28, pp. 100–108, 1979.
                     */
    MEAN_QUANTILES
  }

  private static void ffSelection(final int[] indices, final Random rand, final DataSet d, final int k,
      final DistanceMetric dm, final List<Double> accelCache, final ExecutorService threadpool)
          throws InterruptedException, ExecutionException {
    // Initial random point
    indices[0] = rand.nextInt(d.getSampleSize());

    final double[] closestDist = new double[d.getSampleSize()];
    Arrays.fill(closestDist, Double.POSITIVE_INFINITY);
    final List<Vec> X = d.getDataVectors();

    // Each future will return the local chance to the overal sqared distance.
    final List<Future<Integer>> futures = new ArrayList<Future<Integer>>(LogicalCores);

    for (int j = 1; j < k; j++) {
      // Compute the distance from each data point to the closest mean
      final int newMeanIndx = indices[j - 1];// Only the most recently added
                                             // mean needs to get distances
                                             // computed.
      futures.clear();

      final int blockSize = d.getSampleSize() / LogicalCores;
      int extra = d.getSampleSize() % LogicalCores;
      int pos = 0;
      while (pos < d.getSampleSize()) {
        final int from = pos;
        final int to = Math.min(pos + blockSize + (extra-- > 0 ? 1 : 0), d.getSampleSize());
        pos = to;
        final Future<Integer> future = threadpool.submit(new Callable<Integer>() {

          @Override
          public Integer call() throws Exception {
            double maxDist = Double.NEGATIVE_INFINITY;
            int max = -1;
            for (int i = from; i < to; i++) {
              final double newDist = dm.dist(newMeanIndx, i, X, accelCache);
              closestDist[i] = Math.min(newDist, closestDist[i]);

              if (closestDist[i] > maxDist) {
                maxDist = closestDist[i];
                max = i;
              }
            }

            return max;
          }
        });

        futures.add(future);
      }

      int max = -1;
      double maxDist = Double.NEGATIVE_INFINITY;
      for (final Integer localMax : ListUtils.collectFutures(futures)) {
        if (closestDist[localMax] > maxDist) {
          max = localMax;
          maxDist = closestDist[localMax];
        }
      }

      indices[j] = max;
    }
  };

  private static void kppSelection(final int[] indices, final Random rand, final DataSet d, final int k,
      final DistanceMetric dm, final List<Double> accelCache) {
    /*
     * http://www.stanford.edu/~darthur/kMeansPlusPlus.pdf : k-means++: The
     * Advantages of Careful Seeding
     *
     */
    // Initial random point
    indices[0] = rand.nextInt(d.getSampleSize());

    final double[] closestDist = new double[d.getSampleSize()];
    double sqrdDistSum = 0.0;
    double newDist;

    final List<Vec> vecs = d.getDataVectors();

    for (int j = 1; j < k; j++) {
      // Compute the distance from each data point to the closest mean
      final int newMeanIndx = indices[j - 1];// Only the most recently added
                                             // mean needs to get distances
                                             // computed.
      for (int i = 0; i < d.getSampleSize(); i++) {
        newDist = dm.dist(newMeanIndx, i, vecs, accelCache);

        newDist *= newDist;
        if (newDist < closestDist[i] || j == 1) {
          sqrdDistSum -= closestDist[i];// on inital, -= 0 changes nothing. on
                                        // others, removed the old value
          sqrdDistSum += newDist;
          closestDist[i] = newDist;
        }
      }

      if (sqrdDistSum <= 1e-6) // everyone is too close, randomly fill rest
      {
        final Set<Integer> ind = new IntSet();
        for (int i = 0; i < j; i++) {
          ind.add(indices[i]);
        }
        while (ind.size() < k) {
          ind.add(rand.nextInt(closestDist.length));
        }
        int pos = 0;
        for (final int i : ind) {
          indices[pos++] = i;
        }
        return;
      }

      // Choose new x as weighted probablity by the squared distances
      final double rndX = rand.nextDouble() * sqrdDistSum;
      double searchSum = closestDist[0];
      int i = 0;
      while (searchSum < rndX && i < d.getSampleSize() - 1) {
        searchSum += closestDist[++i];
      }

      indices[j] = i;
    }
  }

  private static void kppSelection(final int[] indices, final Random rand, final DataSet d, final int k,
      final DistanceMetric dm, final List<Double> accelCache, final ExecutorService threadpool)
          throws InterruptedException, ExecutionException {
    /*
     * http://www.stanford.edu/~darthur/kMeansPlusPlus.pdf : k-means++: The
     * Advantages of Careful Seeding
     *
     */
    // Initial random point
    indices[0] = rand.nextInt(d.getSampleSize());

    final double[] closestDist = new double[d.getSampleSize()];
    double sqrdDistSum = 0.0;
    final List<Vec> X = d.getDataVectors();

    // Each future will return the local chance to the overal sqared distance.
    final List<Future<Double>> futureChanges = new ArrayList<Future<Double>>(LogicalCores);

    for (int j = 1; j < k; j++) {
      // Compute the distance from each data point to the closest mean
      final int newMeanIndx = indices[j - 1];// Only the most recently added
                                             // mean needs to get distances
                                             // computed.
      futureChanges.clear();

      for (int id = 0; id < LogicalCores; id++) {
        final int from = ParallelUtils.getStartBlock(X.size(), id, LogicalCores);
        final int to = ParallelUtils.getEndBlock(X.size(), id, LogicalCores);
        final boolean forceCompute = j == 1;
        final Future<Double> future = threadpool.submit(new Callable<Double>() {

          @Override
          public Double call() throws Exception {
            double sqrdDistChanges = 0.0;
            for (int i = from; i < to; i++) {
              double newDist = dm.dist(newMeanIndx, i, X, accelCache);

              newDist *= newDist;
              if (newDist < closestDist[i] || forceCompute) {
                sqrdDistChanges -= closestDist[i];// on inital, -= 0 changes
                                                  // nothing. on others, removed
                                                  // the old value
                sqrdDistChanges += newDist;
                closestDist[i] = newDist;
              }
            }

            return sqrdDistChanges;
          }
        });

        futureChanges.add(future);
      }

      for (final Double change : ListUtils.collectFutures(futureChanges)) {
        sqrdDistSum += change;
      }

      if (sqrdDistSum <= 1e-6) // everyone is too close, randomly fill rest
      {
        final Set<Integer> ind = new IntSet();
        for (int i = 0; i < j; i++) {
          ind.add(indices[i]);
        }
        while (ind.size() < k) {
          ind.add(rand.nextInt(closestDist.length));
        }
        int pos = 0;
        for (final int i : ind) {
          indices[pos++] = i;
        }
        return;
      }

      // Choose new x as weighted probablity by the squared distances
      final double rndX = rand.nextDouble() * sqrdDistSum;
      double searchSum = closestDist[0];
      int i = 0;
      while (searchSum < rndX && i < d.getSampleSize() - 1) {
        searchSum += closestDist[++i];
      }

      indices[j] = i;
    }
  }

  private static void mqSelection(final int[] indices, final DataSet d, final int k, final DistanceMetric dm,
      final List<Double> accelCache, final ExecutorService threadpool) throws InterruptedException, ExecutionException {
    final double[] meanDist = new double[d.getSampleSize()];

    // Compute the distance from each data point to the closest mean
    final Vec newMean = MatrixStatistics.meanVector(d);
    final List<Double> meanQI = dm.getQueryInfo(newMean);
    final List<Vec> X = d.getDataVectors();

    final CountDownLatch latch = new CountDownLatch(LogicalCores);
    final int blockSize = d.getSampleSize() / LogicalCores;
    int extra = d.getSampleSize() % LogicalCores;
    int pos = 0;
    while (pos < d.getSampleSize()) {
      final int from = pos;
      final int to = Math.min(pos + blockSize + (extra-- > 0 ? 1 : 0), d.getSampleSize());
      pos = to;
      threadpool.submit(new Runnable() {
        @Override
        public void run() {
          for (int i = from; i < to; i++) {
            meanDist[i] = dm.dist(i, newMean, meanQI, X, accelCache);
          }
          latch.countDown();
        }
      });
    }

    latch.await();

    final IndexTable indxTbl = new IndexTable(meanDist);
    for (int l = 0; l < k; l++) {
      indices[l] = indxTbl.index(l * d.getSampleSize() / k);
    }
  }

  /**
   *
   * @param d
   *          the data set to perform select from
   * @param k
   *          the number of seeds to choose
   * @param dm
   *          the distance metric to used when selecting points
   * @param accelCache
   *          the cache of pre-generated acceleration information for the
   *          distance metric. May be null
   * @param rand
   *          a source of randomness
   * @param selectionMethod
   *          The method of seed selection to use.
   * @return a list of the copies of the chosen vectors.
   */
  static public List<Vec> selectIntialPoints(final DataSet d, final int k, final DistanceMetric dm,
      final List<Double> accelCache, final Random rand, final SeedSelection selectionMethod) {
    final int[] indicies = new int[k];
    selectIntialPoints(d, indicies, dm, accelCache, rand, selectionMethod, null);
    final List<Vec> vecs = new ArrayList<Vec>(k);
    for (final Integer i : indicies) {
      vecs.add(d.getDataPoint(i).getNumericalValues().clone());
    }
    return vecs;
  }

  /**
   * Selects seeds from a data set to use for a clustering algorithm. Copies of
   * the vectors chosen will be returned.
   *
   * @param d
   *          the data set to perform select from
   * @param k
   *          the number of seeds to choose
   * @param dm
   *          the distance metric to used when selecting points
   * @param accelCache
   *          the cache of pre-generated acceleration information for the
   *          distance metric. May be null
   * @param rand
   *          a source of randomness
   * @param selectionMethod
   *          The method of seed selection to use.
   * @param threadpool
   *          the source of threads for parallel computation
   * @return a list of the copies of the chosen vectors.
   */
  static public List<Vec> selectIntialPoints(final DataSet d, final int k, final DistanceMetric dm,
      final List<Double> accelCache, final Random rand, final SeedSelection selectionMethod,
      final ExecutorService threadpool) {
    final int[] indicies = new int[k];
    selectIntialPoints(d, indicies, dm, accelCache, rand, selectionMethod, threadpool);
    final List<Vec> vecs = new ArrayList<Vec>(k);
    for (final Integer i : indicies) {
      vecs.add(d.getDataPoint(i).getNumericalValues().clone());
    }
    return vecs;
  }

  /**
   * Selects seeds from a data set to use for a clustering algorithm. Copies of
   * the vectors chosen will be returned.
   *
   * @param d
   *          the data set to perform select from
   * @param k
   *          the number of seeds to choose
   * @param dm
   *          the distance metric to used when selecting points
   * @param rand
   *          a source of randomness
   * @param selectionMethod
   *          The method of seed selection to use.
   * @return a list of the copies of the chosen vectors.
   */
  static public List<Vec> selectIntialPoints(final DataSet d, final int k, final DistanceMetric dm, final Random rand,
      final SeedSelection selectionMethod) {
    return selectIntialPoints(d, k, dm, null, rand, selectionMethod);
  }

  /**
   * Selects seeds from a data set to use for a clustering algorithm. Copies of
   * the vectors chosen will be returned.
   *
   * @param d
   *          the data set to perform select from
   * @param k
   *          the number of seeds to choose
   * @param dm
   *          the distance metric to used when selecting points
   * @param rand
   *          a source of randomness
   * @param selectionMethod
   *          The method of seed selection to use.
   * @param threadpool
   *          the source of threads for parallel computation
   * @return a list of the copies of the chosen vectors.
   */
  static public List<Vec> selectIntialPoints(final DataSet d, final int k, final DistanceMetric dm, final Random rand,
      final SeedSelection selectionMethod, final ExecutorService threadpool) {
    return selectIntialPoints(d, k, dm, null, rand, selectionMethod, threadpool);
  }

  /**
   * Selects seeds from a data set to use for a clustering algorithm. The
   * indices of the chosen points will be placed in the <tt>indices</tt> array.
   *
   * @param d
   *          the data set to perform select from
   * @param indices
   *          a storage place to note the indices that were chosen as seed. The
   *          length of the array indicates how many seeds to select.
   * @param dm
   *          the distance metric to used when selecting points
   * @param accelCache
   *          the cache of pre-generated acceleration information for the
   *          distance metric. May be null
   * @param rand
   *          a source of randomness
   * @param selectionMethod
   *          The method of seed selection to use.
   */
  static public void selectIntialPoints(final DataSet d, final int[] indices, final DistanceMetric dm,
      final List<Double> accelCache, final Random rand, final SeedSelection selectionMethod) {
    selectIntialPoints(d, indices, dm, accelCache, rand, selectionMethod, null);
  }

  /**
   * Selects seeds from a data set to use for a clustering algorithm. The
   * indices of the chosen points will be placed in the <tt>indices</tt> array.
   *
   * @param d
   *          the data set to perform select from
   * @param indices
   *          a storage place to note the indices that were chosen as seed. The
   *          length of the array indicates how many seeds to select.
   * @param dm
   *          the distance metric to used when selecting points
   * @param accelCache
   *          the cache of pre-generated acceleration information for the
   *          distance metric. May be null
   * @param rand
   *          a source of randomness
   * @param selectionMethod
   *          The method of seed selection to use.
   * @param threadpool
   *          the source of threads for parallel computation
   */
  static public void selectIntialPoints(final DataSet d, final int[] indices, final DistanceMetric dm,
      final List<Double> accelCache, final Random rand, final SeedSelection selectionMethod,
      final ExecutorService threadpool) {
    try {
      final int k = indices.length;

      if (selectionMethod == SeedSelection.RANDOM) {
        final Set<Integer> indecies = new IntSet(k);

        while (indecies.size() != k) {
          indecies.add(rand.nextInt(d.getSampleSize()));// TODO create method to
                                                        // do uniform sampleling
                                                        // for a select range
        }

        int j = 0;
        for (final Integer i : indecies) {
          indices[j++] = i;
        }
      } else if (selectionMethod == SeedSelection.KPP) {
        if (threadpool == null || threadpool instanceof FakeExecutor) {
          kppSelection(indices, rand, d, k, dm, accelCache);
        } else {
          kppSelection(indices, rand, d, k, dm, accelCache, threadpool);
        }
      } else if (selectionMethod == SeedSelection.FARTHEST_FIRST) {
        if (threadpool == null) {
          ffSelection(indices, rand, d, k, dm, accelCache, new FakeExecutor());
        } else {
          ffSelection(indices, rand, d, k, dm, accelCache, threadpool);
        }
      } else if (selectionMethod == SeedSelection.MEAN_QUANTILES) {
        if (threadpool == null) {
          mqSelection(indices, d, k, dm, accelCache, new FakeExecutor());
        } else {
          mqSelection(indices, d, k, dm, accelCache, threadpool);
        }
      }
    } catch (final InterruptedException ex) {
      Logger.getLogger(SeedSelectionMethods.class.getName()).log(Level.SEVERE, null, ex);
    } catch (final ExecutionException ex) {
      Logger.getLogger(SeedSelectionMethods.class.getName()).log(Level.SEVERE, null, ex);
    }
  }

  /**
   * Selects seeds from a data set to use for a clustering algorithm. The
   * indices of the chosen points will be placed in the <tt>indices</tt> array.
   *
   * @param d
   *          the data set to perform select from
   * @param indices
   *          a storage place to note the indices that were chosen as seed. The
   *          length of the array indicates how many seeds to select.
   * @param dm
   *          the distance metric to used when selecting points
   * @param rand
   *          a source of randomness
   * @param selectionMethod
   *          The method of seed selection to use.
   */
  static public void selectIntialPoints(final DataSet d, final int[] indices, final DistanceMetric dm,
      final Random rand, final SeedSelection selectionMethod) {
    selectIntialPoints(d, indices, dm, null, rand, selectionMethod);
  }

  /**
   * Selects seeds from a data set to use for a clustering algorithm. The
   * indices of the chosen points will be placed in the <tt>indices</tt> array.
   *
   * @param d
   *          the data set to perform select from
   * @param indices
   *          a storage place to note the indices that were chosen as seed. The
   *          length of the array indicates how many seeds to select.
   * @param dm
   *          the distance metric to used when selecting points
   * @param rand
   *          a source of randomness
   * @param selectionMethod
   *          The method of seed selection to use.
   * @param threadpool
   *          the source of threads for parallel computation
   */
  static public void selectIntialPoints(final DataSet d, final int[] indices, final DistanceMetric dm,
      final Random rand, final SeedSelection selectionMethod, final ExecutorService threadpool) {
    selectIntialPoints(d, indices, dm, null, rand, selectionMethod, threadpool);
  }

  private SeedSelectionMethods() {
  }

}
