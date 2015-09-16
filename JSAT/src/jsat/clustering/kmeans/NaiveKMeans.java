package jsat.clustering.kmeans;

import static jsat.clustering.SeedSelectionMethods.selectIntialPoints;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.clustering.SeedSelectionMethods;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

/**
 * An implementation of Lloyd's K-Means clustering algorithm using the naive
 * algorithm. This implementation exists mostly for comparison as a base line
 * and educational reasons. For efficient exact k-Means, use {@link ElkanKMeans}
 * <br>
 * <br>
 * This implementation is parallel, but does not support any of the clustering
 * methods that do not specify the number of clusters.
 *
 * @author Edward Raff
 */
public class NaiveKMeans extends KMeans {

  private static final long serialVersionUID = 6164910874898843069L;

  /**
   * Creates a new naive k-Means cluster using {@link SeedSelection#KPP
   * k-means++} for the seed selection and the {@link EuclideanDistance}
   */
  public NaiveKMeans() {
    this(new EuclideanDistance());
  }

  /**
   * Creates a new naive k-Means cluster using {@link SeedSelection#KPP
   * k-means++} for the seed selection.
   *
   * @param dm
   *          the distance function to use
   */
  public NaiveKMeans(final DistanceMetric dm) {
    this(dm, SeedSelectionMethods.SeedSelection.KPP);
  }

  /**
   * Creates a new naive k-Means cluster
   *
   * @param dm
   *          the distance function to use
   * @param seedSelection
   *          the method of selecting the initial seeds
   */
  public NaiveKMeans(final DistanceMetric dm, final SeedSelection seedSelection) {
    this(dm, seedSelection, new XORWOW());
  }

  /**
   * Creates a new naive k-Means cluster
   *
   * @param dm
   *          the distance function to use
   * @param seedSelection
   *          the method of selecting the initial seeds
   * @param rand
   *          the source of randomness to use
   */
  public NaiveKMeans(final DistanceMetric dm, final SeedSelection seedSelection, final Random rand) {
    super(dm, seedSelection, rand);
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public NaiveKMeans(final NaiveKMeans toCopy) {
    super(toCopy);
  }

  @Override
  public NaiveKMeans clone() {
    return new NaiveKMeans(this);
  }

  @Override
  protected double cluster(final DataSet dataSet, final List<Double> accelCacheInit, final int k, final List<Vec> means,
      final int[] assignment, final boolean exactTotal, ExecutorService threadpool, final boolean returnError) {
    TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadpool);

    if (threadpool == null) {
      threadpool = new FakeExecutor();
    }

    final int blockSize = dataSet.getSampleSize() / SystemInfo.LogicalCores;
    final List<Vec> X = dataSet.getDataVectors();
    // done a wonky way b/c we want this as a final object for convinence,
    // otherwise we may be stuck with null accel when we dont need to be
    final List<Double> accelCache;
    if (accelCacheInit == null) {
      if (threadpool instanceof FakeExecutor) {
        accelCache = dm.getAccelerationCache(X);
      } else {
        accelCache = dm.getAccelerationCache(X, threadpool);
      }
    } else {
      accelCache = accelCacheInit;
    }

    if (means.size() != k) {
      means.clear();
      if (threadpool instanceof FakeExecutor) {
        means.addAll(selectIntialPoints(dataSet, k, dm, accelCache, rand, seedSelection));
      } else {
        means.addAll(selectIntialPoints(dataSet, k, dm, accelCache, rand, seedSelection, threadpool));
      }
    }

    final List<List<Double>> meanQIs = new ArrayList<List<Double>>(k);

    // Use dense mean objects
    for (int i = 0; i < means.size(); i++) {
      if (dm.supportsAcceleration()) {
        meanQIs.add(dm.getQueryInfo(means.get(i)));
      } else {
        meanQIs.add(Collections.EMPTY_LIST);
      }

      if (means.get(i).isSparse()) {
        means.set(i, new DenseVector(means.get(i)));
      }
    }

    final List<Vec> meanSum = new ArrayList<Vec>(means.size());
    final AtomicIntegerArray meanCounts = new AtomicIntegerArray(means.size());
    for (int i = 0; i < k; i++) {
      meanSum.add(new DenseVector(means.get(0).length()));
    }
    final AtomicInteger changes = new AtomicInteger();

    // used to store local changes to the means and accumulated at the end
    final ThreadLocal<Vec[]> localMeanDeltas = new ThreadLocal<Vec[]>() {
      @Override
      protected Vec[] initialValue() {
        final Vec[] deltas = new Vec[k];
        for (int i = 0; i < k; i++) {
          deltas[i] = new DenseVector(means.get(0).length());
        }
        return deltas;
      }
    };

    Arrays.fill(assignment, -1);
    do {
      changes.set(0);
      int extra = dataSet.getSampleSize() % SystemInfo.LogicalCores;
      int start = 0;
      final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
      while (start < dataSet.getSampleSize()) {
        final int s = start;
        final int end = start + blockSize + (extra-- > 0 ? 1 : 0);
        threadpool.submit(new Runnable() {
          @Override
          public void run() {
            final Vec[] deltas = localMeanDeltas.get();
            double tmp;
            for (int i = s; i < end; i++) {
              final Vec x = X.get(i);
              double minDist = Double.POSITIVE_INFINITY;
              int min = -1;
              for (int j = 0; j < means.size(); j++) {
                tmp = dm.dist(i, means.get(j), meanQIs.get(j), X, accelCache);
                if (tmp < minDist) {
                  minDist = tmp;
                  min = j;
                }
              }
              if (assignment[i] == min) {
                continue;
              }

              // add change
              deltas[min].mutableAdd(x);
              meanCounts.incrementAndGet(min);
              // remove from prev owner
              if (assignment[i] >= 0) {
                deltas[assignment[i]].mutableSubtract(x);
                meanCounts.getAndDecrement(assignment[i]);
              }
              assignment[i] = min;
              changes.incrementAndGet();
            }

            // accumulate deltas into globals
            for (int i = 0; i < deltas.length; i++) {
              synchronized (meanSum.get(i)) {
                meanSum.get(i).mutableAdd(deltas[i]);
                deltas[i].zeroOut();
              }
            }

            latch.countDown();
          }
        });

        start = end;
      }

      try {
        latch.await();
        if (changes.get() == 0) {
          break;
        }
        for (int i = 0; i < k; i++) {
          meanSum.get(i).copyTo(means.get(i));
          means.get(i).mutableDivide(meanCounts.get(i));
          if (dm.supportsAcceleration()) {
            meanQIs.set(i, dm.getQueryInfo(means.get(i)));
          }
        }
      } catch (final InterruptedException ex) {
        Logger.getLogger(NaiveKMeans.class.getName()).log(Level.SEVERE, null, ex);
      }
    } while (changes.get() > 0);

    if (returnError) {
      double totalDistance = 0;
      if (saveCentroidDistance) {
        nearestCentroidDist = new double[X.size()];
      } else {
        nearestCentroidDist = null;
      }

      for (int i = 0; i < dataSet.getSampleSize(); i++) {
        final double dist = dm.dist(i, means.get(assignment[i]), meanQIs.get(assignment[i]), X, accelCache);
        totalDistance += Math.pow(dist, 2);
        if (saveCentroidDistance) {
          nearestCentroidDist[i] = dist;
        }
      }

      return totalDistance;
    } else {
      return 0;// who cares
    }
  }

}
