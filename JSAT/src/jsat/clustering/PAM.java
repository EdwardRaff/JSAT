package jsat.clustering;

import static jsat.clustering.SeedSelectionMethods.selectIntialPoints;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.math.OnLineStatistics;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class PAM extends KClustererBase {

  private class ClusterWorker implements Runnable {

    private final DataSet dataSet;
    private int[] assignments;
    private int[] medioids;
    private volatile double result = -1;
    private volatile BlockingQueue<ClusterWorker> putSelf;

    public ClusterWorker(final DataSet dataSet, final BlockingQueue<ClusterWorker> putSelf) {
      this.dataSet = dataSet;
      assignments = new int[dataSet.getSampleSize()];
      // XXX useless assignment
      // this.medioids = medioids;
      this.putSelf = putSelf;
    }

    public int getK() {
      return medioids.length;
    }

    public double getResult() {
      return result;
    }

    @Override
    public void run() {
      result = cluster(dataSet, true, medioids, assignments, null);
      putSelf.add(this);
    }

    @SuppressWarnings("unused")
    public void setAssignments(final int[] assignments) {
      this.assignments = assignments;
    }

    public void setMedioids(final int[] medioids) {
      this.medioids = medioids;
    }

  }

  private static final long serialVersionUID = 4787649180692115514L;
  protected DistanceMetric dm;
  protected Random rand;
  protected SeedSelection seedSelection;
  protected int repeats = 1;

  protected int iterLimit = 100;
  protected int[] medoids;

  protected boolean storeMedoids = true;

  public PAM() {
    this(new EuclideanDistance());
  }

  public PAM(final DistanceMetric dm) {
    this(dm, new Random());
  }

  public PAM(final DistanceMetric dm, final Random rand) {
    this(dm, rand, SeedSelection.KPP);
  }

  public PAM(final DistanceMetric dm, final Random rand, final SeedSelection seedSelection) {
    this.dm = dm;
    this.rand = rand;
    this.seedSelection = seedSelection;
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public PAM(final PAM toCopy) {
    dm = toCopy.dm.clone();
    rand = new XORWOW();
    seedSelection = toCopy.seedSelection;
    if (toCopy.medoids != null) {
      medoids = Arrays.copyOf(toCopy.medoids, toCopy.medoids.length);
    }
    storeMedoids = toCopy.storeMedoids;
    iterLimit = toCopy.iterLimit;
    repeats = toCopy.repeats;
  }

  @Override
  public PAM clone() {
    return new PAM(this);
  }

  /**
   * Performs the actual work of PAM.
   *
   * @param data
   *          the data set to apply PAM to
   * @param doInit
   *          {@code true} if the initialization procedure of training the
   *          distance metric, initiating its cache, and selecting he seeds,
   *          should be done.
   * @param medioids
   *          the array to store the indices that get chosen as the medoids. The
   *          length of the array indicates how many medoids should be obtained.
   * @param assignments
   *          an array of the same length as <tt>data</tt>, each value
   *          indicating what cluster that point belongs to.
   * @param cacheAccel
   *          the pre-computed distance acceleration cache. May be {@code null}.
   * @return the sum of the squared distance from each point to its closest
   *         medoids
   */
  protected double cluster(final DataSet data, final boolean doInit, final int[] medioids, final int[] assignments,
      List<Double> cacheAccel) {
    double totalDistance = 0;
    int changes = -1;
    Arrays.fill(assignments, -1);// -1, invalid category!

    final int[] bestMedCand = new int[medioids.length];
    final double[] bestMedCandDist = new double[medioids.length];
    final List<Vec> X = data.getDataVectors();

    if (doInit) {
      TrainableDistanceMetric.trainIfNeeded(dm, data);
      cacheAccel = dm.getAccelerationCache(X);
      selectIntialPoints(data, medoids, dm, cacheAccel, rand, seedSelection);
    }

    int iter = 0;
    do {
      changes = 0;
      totalDistance = 0.0;

      for (int i = 0; i < data.getSampleSize(); i++) {
        final Vec dpVec = data.getDataPoint(i).getNumericalValues();
        int assignment = 0;
        double minDist = dm.dist(medioids[0], i, X, cacheAccel);

        for (int k = 1; k < medioids.length; k++) {
          final double dist = dm.dist(medioids[k], i, X, cacheAccel);
          if (dist < minDist) {
            minDist = dist;
            assignment = k;
          }
        }

        // Update which cluster it is in
        if (assignments[i] != assignment) {
          changes++;
          assignments[i] = assignment;
        }
        totalDistance += minDist * minDist;

      }

      // TODO this update may be faster by using more memory, and actually
      // moiving people about in the assignment loop above
      // Update the medoids
      Arrays.fill(bestMedCandDist, Double.MAX_VALUE);
      for (int i = 0; i < data.getSampleSize(); i++) {
        double thisCandidateDistance = 0.0;
        final int clusterID = assignments[i];
        final int medCandadate = i;
        for (int j = 0; j < data.getSampleSize(); j++) {
          if (j == i || assignments[j] != clusterID) {
            continue;
          }
          thisCandidateDistance += Math.pow(dm.dist(medCandadate, j, X, cacheAccel), 2);
        }

        if (thisCandidateDistance < bestMedCandDist[clusterID]) {
          bestMedCand[clusterID] = i;
          bestMedCandDist[clusterID] = thisCandidateDistance;
        }
      }
      System.arraycopy(bestMedCand, 0, medioids, 0, medioids.length);
    } while (changes > 0 && iter++ < iterLimit);

    return totalDistance;
  }

  @Override
  public int[] cluster(final DataSet dataSet, final ExecutorService threadpool, final int[] designations) {
    return cluster(dataSet, 2, (int) Math.sqrt(dataSet.getSampleSize() / 2), threadpool, designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int clusters, final ExecutorService threadpool,
      final int[] designations) {
    return cluster(dataSet, clusters, designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int lowK, final int highK, final ExecutorService threadpool,
      final int[] designations) {
    final double[] totDistances = new double[highK - lowK + 1];

    final BlockingQueue<ClusterWorker> workerQue = new ArrayBlockingQueue<ClusterWorker>(SystemInfo.LogicalCores);
    for (int i = 0; i < SystemInfo.LogicalCores; i++) {
      workerQue.add(new ClusterWorker(dataSet, workerQue));
    }

    int k = lowK;
    int received = 0;
    while (received < totDistances.length) {
      try {
        final ClusterWorker worker = workerQue.take();
        if (worker.getResult() != -1) // -1 means not really in use
        {
          totDistances[worker.getK() - lowK] = worker.getResult();
          received++;
        }
        if (k <= highK) {
          worker.setMedioids(new int[k++]);
          threadpool.submit(worker);
        }
      } catch (final InterruptedException ex) {
        Logger.getLogger(PAM.class.getName()).log(Level.SEVERE, null, ex);
      }
    }

    // Now we process the distance changes
    /**
     * Keep track of the changes
     */
    final OnLineStatistics stats = new OnLineStatistics();

    double maxChange = Double.MIN_VALUE;
    int maxChangeK = lowK;

    for (int i = 1; i < totDistances.length; i++) {
      final double change = Math.abs(totDistances[i] - totDistances[i - 1]);
      stats.add(change);
      if (change > maxChange) {
        maxChange = change;
        maxChangeK = i + lowK;
      }
    }

    if (maxChange < stats.getStandardDeviation() * 2 + stats.getMean()) {
      maxChangeK = lowK;
    }

    return cluster(dataSet, maxChangeK, designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int lowK, final int highK, final int[] designations) {
    return cluster(dataSet, lowK, highK, new FakeExecutor(), designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int clusters, int[] designations) {
    if (designations == null) {
      designations = new int[dataSet.getSampleSize()];
    }
    medoids = new int[clusters];

    cluster(dataSet, true, medoids, designations, null);

    if (!storeMedoids) {
      medoids = null;
    }

    return designations;
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int[] designations) {
    return cluster(dataSet, 2, (int) Math.sqrt(dataSet.getSampleSize() / 2), designations);
  }

  /**
   * Returns the raw array of indices that indicate which data point acted as
   * the center for each cluster.
   *
   * @return the array of medeoid indices
   */
  public int[] getMedoids() {
    return medoids;
  }

  /**
   *
   * @return the method of seed selection used by this algorithm
   */
  public SeedSelection getSeedSelection() {
    return seedSelection;
  }

  /**
   * Sets the method of seed selection used by this algorithm
   *
   * @param seedSelection
   *          the method of seed selection to used
   */
  public void setSeedSelection(final SeedSelection seedSelection) {
    this.seedSelection = seedSelection;
  }

  /**
   * If set to {@code true} the computed medoids will be stored after clustering
   * is completed, and can then be retrieved using {@link #getMedoids() }.
   *
   * @param storeMedoids
   *          {@code true} if the medoids should be stored for later,
   *          {@code false} to discard them once clustering is complete.
   */
  public void setStoreMedoids(final boolean storeMedoids) {
    this.storeMedoids = storeMedoids;
  }
}
