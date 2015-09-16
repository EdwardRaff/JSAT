package jsat.clustering.kmeans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.clustering.KClustererBase;
import jsat.distributions.kernels.KernelTrick;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.random.XOR96;

/**
 * Base class for various Kernel K Means implementations. Because the Kernelized
 * version is more computationally expensive, only the clustering methods where
 * the number of clusters is specified apriori are supported. <br>
 * <br>
 * KernelKMeans keeps a reference to the data passed in for clustering so that
 * queries can be conveniently answered, such as getting
 * {@link #findClosestCluster(jsat.linear.Vec) the closest cluster} or finding
 * the {@link #meanToMeanDistance(int, int) distance between means}
 *
 * @author Edward Raff
 */
public abstract class KernelKMeans extends KClustererBase implements Parameterized {

  private static final long serialVersionUID = -5394680202634779440L;

  /**
   * The kernel trick to use
   */
  @ParameterHolder
  protected KernelTrick kernel;

  /**
   * The list of data points that this was trained on
   */
  protected List<Vec> X;
  /**
   * THe acceleration cache for the kernel
   */
  protected List<Double> accel;
  /**
   * The value of k(x,x) for every point in {@link #X}
   */
  protected double[] selfK;

  /**
   * The value of the un-normalized squared norm for each mean
   */
  protected double[] meanSqrdNorms;

  /**
   * The normalizing constant for each mean. General this would be 1/owned[k]
   * <sup>2</sup>
   */
  protected double[] normConsts;

  /**
   * The number of dataums owned by each mean
   */
  protected int[] ownes;

  /**
   * A temporary space for updating ownership designations for each datapoint.
   * When done, this will store the final designations for each point
   */
  protected int[] newDesignations;
  protected int maximumIterations = Integer.MAX_VALUE;

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public KernelKMeans(final KernelKMeans toCopy) {
    kernel = toCopy.kernel.clone();
    maximumIterations = toCopy.maximumIterations;
    if (toCopy.X != null) {
      X = new ArrayList<Vec>(toCopy.X.size());
      for (final Vec v : toCopy.X) {
        X.add(v.clone());
      }

    }
    if (toCopy.accel != null) {
      accel = new DoubleList(toCopy.accel);
    }
    if (toCopy.selfK != null) {
      selfK = Arrays.copyOf(toCopy.selfK, toCopy.selfK.length);
    }

    if (toCopy.meanSqrdNorms != null) {
      meanSqrdNorms = Arrays.copyOf(toCopy.meanSqrdNorms, toCopy.meanSqrdNorms.length);
    }

    if (toCopy.normConsts != null) {
      normConsts = Arrays.copyOf(toCopy.normConsts, toCopy.normConsts.length);
    }

    if (toCopy.ownes != null) {
      ownes = Arrays.copyOf(toCopy.ownes, toCopy.ownes.length);
    }

    if (toCopy.newDesignations != null) {
      newDesignations = Arrays.copyOf(toCopy.newDesignations, toCopy.newDesignations.length);
    }
  }

  /**
   *
   * @param kernel
   *          the kernel to use
   */
  public KernelKMeans(final KernelTrick kernel) {
    this.kernel = kernel;
  }

  protected void applyMeanUpdates(final double[] sqrdNorms, final int[] ownerships) {
    for (int i = 0; i < sqrdNorms.length; i++) {
      meanSqrdNorms[i] += sqrdNorms[i];
      ownes[i] += ownerships[i];
    }
  }

  @Override
  abstract public KernelKMeans clone();

  @Override
  public int[] cluster(final DataSet dataSet, final ExecutorService threadpool, final int[] designations) {
    throw new UnsupportedOperationException("Not supported.");
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int lowK, final int highK, final ExecutorService threadpool,
      final int[] designations) {
    throw new UnsupportedOperationException("Not supported.");
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int lowK, final int highK, final int[] designations) {
    throw new UnsupportedOperationException("Not supported.");
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int[] designations) {
    throw new UnsupportedOperationException("Not supported.");
  }

  /**
   * Computes the distance between one data point and a specified mean
   *
   * @param i
   *          the data point to get the distance for
   * @param k
   *          the mean index to get the distance to
   * @param designations
   *          the array if ownership designations for each cluster to use
   * @return the distance between data point {@link #X x}<sub>i</sub> and mean
   *         {@code k}
   */
  protected double distance(final int i, final int k, final int[] designations) {
    return Math
        .sqrt(Math.max(selfK[i] - 2.0 / ownes[k] * evalSumK(i, k, designations) + meanSqrdNorms[k] * normConsts[k], 0));
  }

  /**
   * Returns the distance between the given data point and the the specified
   * cluster
   *
   * @param x
   *          the data point to get the distance for
   * @param k
   *          the cluster id to get the distance to
   * @return the distance between the given data point and the specified cluster
   */
  public double distance(final Vec x, final int k) {
    return distance(x, kernel.getQueryInfo(x), k);
  }

  /**
   * Returns the distance between the given data point and the the specified
   * cluster
   *
   * @param x
   *          the data point to get the distance for
   * @param qi
   *          the query information for the given data point generated for the
   *          kernel in use. See
   *          {@link KernelTrick#getQueryInfo(jsat.linear.Vec) } @ param k the
   *          cluster id to get the distance to
   * @return the distance between the given data point and the specified cluster
   */
  public double distance(final Vec x, final List<Double> qi, final int k) {
    if (k >= meanSqrdNorms.length || k < 0) {
      throw new IndexOutOfBoundsException("Only " + meanSqrdNorms.length + " clusters. " + k + " is not a valid index");
    }
    return Math.sqrt(Math.max(kernel.eval(0, 0, Arrays.asList(x), qi)
        - 2.0 / ownes[k] * evalSumK(x, qi, k, newDesignations) + meanSqrdNorms[k] * normConsts[k], 0));
  }

  /**
   * dot product between two different clusters from one set of cluster
   * assignments
   *
   * @param k0
   *          the index of the first cluster
   * @param k1
   *          the index of the second cluster
   * @param assignment
   *          the array of assignments for cluster ownership
   * @return
   */
  private double dot(final int k0, final int k1, final int[] assignment) {
    return dot(k0, k1, assignment, assignment);
  }

  /**
   * dot product between two different clusters from different sets of cluster
   * assignments
   *
   * @param k0
   * @param k1
   * @param assignment
   * @return
   */
  private double dot(final int k0, final int k1, final int[] assignment0, final int[] assignment1) {
    double dot = 0;
    final int N = X.size();
    int a = 0, b = 0;
    /*
     * Below, unless i&amp;j are somehow in the same cluster - nothing bad will
     * happen
     */
    for (int i = 0; i < N; i++) {
      if (assignment0[i] != k0) {
        continue;
      }
      a++;
      for (int j = 0; j < N; j++) {
        if (assignment1[j] != k1) {
          continue;
        }
        dot += kernel.eval(i, j, X, accel);
      }
    }
    for (int j = 0; j < N; j++) {
      if (assignment1[j] == k1) {
        b++;
      }
    }
    return dot / (a * b);
  }

  /**
   * Computes the kernel sum of data point {@code i} against all the points in
   * cluster group {@code clusterID}.
   *
   * @param i
   *          the index of the data point to query for
   * @param clusterID
   *          the cluster index to get the sum of kernel products
   * @param d
   * @return the sum <big>&Sigma;</big>k(x<sub>i</sub>, x<sub>j</sub>), &forall;
   *         j, d[<i>j</i>] == <i>clusterID</i>
   */
  protected double evalSumK(final int i, final int clusterID, final int[] d) {
    double sum = 0;
    for (int j = 0; j < X.size(); j++) {
      if (d[j] == clusterID) {
        sum += kernel.eval(i, j, X, accel);
      }
    }
    return sum;
  }

  /**
   * Computes the kernel sum of the given data point against all the points in
   * cluster group {@code clusterID}.
   *
   * @param x
   *          the data point to get the kernel sum of
   * @param qi
   *          the query information for the given data point generated from the
   *          kernel in use. See
   *          {@link KernelTrick#getQueryInfo(jsat.linear.Vec) } @ param
   *          clusterID the cluster index to get the sum of kernel products
   * @param d
   *          the array of cluster assignments
   * @return the sum <big>&Sigma;</big>k(x<sub>i</sub>, x<sub>j</sub>), &forall;
   *         j, d[<i>j</i>] == <i>clusterID</i>
   */
  protected double evalSumK(final Vec x, final List<Double> qi, final int clusterID, final int[] d) {
    double sum = 0;
    for (int j = 0; j < X.size(); j++) {
      if (d[j] == clusterID) {
        sum += kernel.eval(j, x, qi, X, accel);
      }
    }
    return sum;
  }

  /**
   * Finds the cluster ID that is closest to the given data point
   *
   * @param x
   *          the data point to get the closest cluster for
   * @return the index of the closest cluster
   */
  public int findClosestCluster(final Vec x) {
    return findClosestCluster(x, kernel.getQueryInfo(x));
  }

  /**
   * Finds the cluster ID that is closest to the given data point
   *
   * @param x
   *          the data point to get the closest cluster for
   * @param qi
   *          the query information for the given data point generated for the
   *          kernel in use. See
   *          {@link KernelTrick#getQueryInfo(jsat.linear.Vec) } @ return the
   *          index of the closest cluster
   */
  public int findClosestCluster(final Vec x, final List<Double> qi) {
    double min = distance(x, qi, 0);
    int min_indx = 0;
    for (int i = 1; i < meanSqrdNorms.length; i++) {
      final double dist = distance(x, qi, i);
      if (dist < min) {
        min = dist;
        min_indx = i;
      }
    }

    return min_indx;
  }

  /**
   * Returns the maximum number of iterations of the KMeans algorithm that will
   * be performed.
   *
   * @return the maximum number of iterations of the KMeans algorithm that will
   *         be performed.
   */
  public int getMaximumIterations() {
    return maximumIterations;
  }

  @Override
  public Parameter getParameter(final String paramName) {
    return Parameter.toParameterMap(getParameters()).get(paramName);
  }

  @Override
  public List<Parameter> getParameters() {
    return Parameter.getParamsFromMethods(this);
  }

  /**
   * Computes the distance between two of the means in the clustering
   *
   * @param k0
   *          the index of the first mean
   * @param k1
   *          the index of the second mean
   * @return the distance between the two
   */
  public double meanToMeanDistance(final int k0, final int k1) {
    if (k0 >= meanSqrdNorms.length || k0 < 0) {
      throw new IndexOutOfBoundsException(
          "Only " + meanSqrdNorms.length + " clusters. " + k0 + " is not a valid index");
    }
    if (k1 >= meanSqrdNorms.length || k1 < 0) {
      throw new IndexOutOfBoundsException(
          "Only " + meanSqrdNorms.length + " clusters. " + k1 + " is not a valid index");
    }

    return meanToMeanDistance(k0, k1, newDesignations);
  }

  protected double meanToMeanDistance(final int k0, final int k1, final int[] assignments) {
    final double d = meanSqrdNorms[k0] * normConsts[k0] + meanSqrdNorms[k1] * normConsts[k1]
        - 2 * dot(k0, k1, assignments);
    return Math.sqrt(Math.max(0, d));// Avoid rare cases wehre 2*dot might be
                                     // slightly larger
  }

  /**
   *
   * @param k0
   *          the index of the first cluster
   * @param k1
   *          the index of the second cluster
   * @param assignments0
   *          the array of assignments to use for index k0
   * @param assignments1
   *          the array of assignments to use for index k1
   * @param k1SqrdNorm
   *          the <i>normalized</i> squared norm for the mean indicated by
   *          {@code k1}. (ie: {@link #meanSqrdNorms} multiplied by
   *          {@link #normConsts}
   * @return
   */
  protected double meanToMeanDistance(final int k0, final int k1, final int[] assignments0, final int[] assignments1,
      final double k1SqrdNorm) {
    final double d = meanSqrdNorms[k0] * normConsts[k0] + k1SqrdNorm - 2 * dot(k0, k1, assignments0, assignments1);
    return Math.sqrt(Math.max(0, d));// Avoid rare cases wehre 2*dot might be
                                     // slightly larger
  }

  /**
   * Sets the maximum number of iterations allowed
   *
   * @param iterLimit
   *          the maximum number of iterations of the KMeans algorithm
   */
  public void setMaximumIterations(final int iterLimit) {
    if (iterLimit <= 0) {
      throw new IllegalArgumentException("iterations must be a positive value, not " + iterLimit);
    }
    maximumIterations = iterLimit;
  }

  /**
   * Sets up the internal structure for KenrelKMeans. Should be called first
   * before any work is done
   *
   * @param K
   *          the number of clusters to find
   * @param designations
   *          the initial designations array to fill with values
   */
  protected void setup(final int K, final int[] designations) {
    accel = kernel.getAccelerationCache(X);

    final int N = X.size();
    selfK = new double[N];
    for (int i = 0; i < selfK.length; i++) {
      selfK[i] = kernel.eval(i, i, X, accel);
    }
    ownes = new int[K];
    meanSqrdNorms = new double[K];
    newDesignations = new int[N];

    final Random rand = new XOR96();
    for (int i = 0; i < N; i++) {
      final int to = rand.nextInt(K);
      ownes[to]++;
      newDesignations[i] = designations[i] = to;
    }

    normConsts = new double[K];
    updateNormConsts();

    for (int i = 0; i < N; i++) {
      final int i_k = designations[i];
      meanSqrdNorms[i_k] += selfK[i];
      for (int j = i + 1; j < N; j++) {
        if (i_k == designations[j]) {
          meanSqrdNorms[i_k] += 2 * kernel.eval(i, j, X, accel);
        }
      }
    }
  }

  /**
   * Updates the means based off the change of a specific data point
   *
   * @param i
   *          the index of the data point to try and update the means based on
   *          its movement
   * @param designations
   *          the old assignments for ownership of each data point to one of the
   *          means
   * @return {@code 1} if the index changed ownership, {@code 0} if the index
   *         did not change ownership
   */
  protected int updateMeansFromChange(final int i, final int[] designations) {
    return updateMeansFromChange(i, designations, meanSqrdNorms, ownes);
  }

  /**
   * Accumulates the updates to the means and ownership into the provided
   * arrays. This does not update {@link #meanSqrdNorms}, and is meant to
   * accumulate the change. To apply the changes pass the same arrays to
   * {@link #applyMeanUpdates(double[], int[]) }
   *
   * @param i
   *          the index of the data point to try and update the means based on
   *          its movement
   * @param designations
   *          the old assignments for ownership of each data point to one of the
   *          means
   * @param sqrdNorms
   *          the array to place the changes to the squared norms in
   * @param ownership
   *          the array to place the changes to the ownership counts in
   * @return {@code 1} if the index changed ownership, {@code 0} if the index
   *         did not change ownership
   */
  protected int updateMeansFromChange(final int i, final int[] designations, final double[] sqrdNorms,
      final int[] ownership) {
    final int old_d = designations[i];
    final int new_d = newDesignations[i];

    if (old_d == new_d) {// this one has not changed!
      return 0;
    }

    final int N = X.size();

    ownership[old_d]--;
    ownership[new_d]++;

    for (int j = 0; j < N; j++) {
      final int oldD_j = designations[j];
      final int newD_j = newDesignations[j];
      if (i == j) // diagonal is an easy case
      {
        sqrdNorms[old_d] -= selfK[i];
        sqrdNorms[new_d] += selfK[i];
      } else {
        // handle removing contribution from old mean
        if (old_d == oldD_j) {
          // only do this for items that were apart of the OLD center

          if (i > j && oldD_j != newD_j) {
            /*
             * j,j is also being removed from this center. To avoid removing the
             * value k_ij twice, the person with the later index gets to do the
             * update
             */
          } else {
            // safe to remove the k_ij contribution
            sqrdNorms[old_d] -= 2 * kernel.eval(i, j, X, accel);
          }
        }
        // handle adding contributiont to new mean
        if (new_d == newD_j) {
          // only do this for items that are apart of the NEW center

          if (i > j && oldD_j != newD_j) {
            /*
             * j,j is also being added to this center. To avoid adding the value
             * k_ij twice, the person with the later index gets to do the update
             */
          } else {
            sqrdNorms[new_d] += 2 * kernel.eval(i, j, X, accel);
          }
        }
      }
    }

    return 1;
  }

  /**
   * Updates the normalizing constants for each mean. Should be called after
   * every change in ownership
   */
  protected void updateNormConsts() {
    for (int i = 0; i < normConsts.length; i++) {
      normConsts[i] = 1.0 / (ownes[i] * (long) ownes[i]);
    }
  }

}
