package jsat.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Stack;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.linear.vectorcollection.DefaultVectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollectionUtils;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.FakeExecutor;
import jsat.utils.IndexTable;

/**
 * A parallel implementation of <i>Locally Scaled Density Based Clustering</i>.
 * <br>
 * <br>
 * See paper:<br>
 * <a href="http://www.springerlink.com/index/0116171485446868.pdf">Biçici, E.,
 * &amp;Yuret, D. (2007). Locally scaled density based clustering. In B.
 * Beliczynski, A. Dzielinski, M. Iwanowski,&amp;B. Ribeiro (Eds.), Adaptive and
 * Natural Computing Algorithms (pp. 739–748). Warsaw, Poland:
 * Springer-Verlag. </a>
 *
 * @author Edward Raff
 */
public class LSDBC extends ClustererBase implements Parameterized {

  private static final long serialVersionUID = 6626217924334267681L;
  /**
   * {@value #DEFAULT_NEIGHBORS} is the default number of neighbors used when
   * performing clustering
   *
   * @see #setNeighbors(int)
   */
  public static final int DEFAULT_NEIGHBORS = 15;
  /**
   * {@value #DEFAULT_ALPHA} is the default scale value used when performing
   * clustering.
   *
   * @see #setAlpha(double)
   */
  public static final double DEFAULT_ALPHA = 4;
  private static final int UNCLASSIFIED = -1;
  private DistanceMetric dm;

  private VectorCollectionFactory<VecPaired<Vec, Integer>> vectorCollectionFactory = new DefaultVectorCollectionFactory<VecPaired<Vec, Integer>>();

  /**
   * The weight parameter for forming new clusters
   */
  private double alpha;
  /**
   * The number of neighbors to use
   */
  private int k;

  /**
   * Creates a new LSDBC clustering object using the {@link EuclideanDistance}
   * and default parameter values.
   */
  public LSDBC() {
    this(new EuclideanDistance());
  }

  /**
   * Creates a new LSDBC clustering object using the given distance metric
   *
   * @param dm
   *          the distance metric to use
   */
  public LSDBC(final DistanceMetric dm) {
    this(dm, DEFAULT_ALPHA);
  }

  /**
   * Creates a new LSDBC clustering object using the given distance metric
   *
   * @param dm
   *          the distance metric to use
   * @param alpha
   *          the scale factor to use when forming clusters
   */
  public LSDBC(final DistanceMetric dm, final double alpha) {
    this(dm, alpha, DEFAULT_NEIGHBORS);
  }

  /**
   * Creates a new LSDBC clustering object using the given distance metric
   *
   * @param dm
   *          the distance metric to use
   * @param alpha
   *          the scale factor to use when forming clusters
   * @param neighbors
   *          the number of neighbors to consider when determining clusters
   */
  public LSDBC(final DistanceMetric dm, final double alpha, final int neighbors) {
    setDistanceMetric(dm);
    setAlpha(alpha);
    setNeighbors(neighbors);
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public LSDBC(final LSDBC toCopy) {
    alpha = toCopy.alpha;
    dm = toCopy.dm.clone();
    k = toCopy.k;
    vectorCollectionFactory = toCopy.vectorCollectionFactory.clone();
  }

  /**
   * Performs the main clustering loop of expandCluster.
   *
   * @param neighbors
   *          the list of neighbors
   * @param i
   *          the index of <tt>neighbors</tt> to consider
   * @param designations
   *          the array of cluster designations
   * @param clusterID
   *          the current clusterID to assign
   * @param seeds
   *          the stack to hold all seeds in
   */
  private void addSeed(final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors, final int i,
      final int[] designations, final int clusterID, final Stack<Integer> seeds) {
    final int index = neighbors.get(i).getVector().getPair();
    if (designations[index] != UNCLASSIFIED) {
      return;
    }
    designations[index] = clusterID;
    seeds.add(index);
  }

  @Override
  public LSDBC clone() {
    return new LSDBC();
  }

  @Override
  public int[] cluster(final DataSet dataSet, final ExecutorService threadpool, int[] designations) {
    if (designations == null) {
      designations = new int[dataSet.getSampleSize()];
    }

    // Compute all k-NN
    final VectorCollection<VecPaired<Vec, Integer>> vc;
    List<List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>> knnVecList = new ArrayList<List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>>(
        dataSet.getSampleSize());

    try {
      // Set up
      final List<VecPaired<Vec, Integer>> vecs = new ArrayList<VecPaired<Vec, Integer>>(dataSet.getSampleSize());

      for (int i = 0; i < dataSet.getSampleSize(); i++) {
        vecs.add(new VecPaired<Vec, Integer>(dataSet.getDataPoint(i).getNumericalValues(), i));
      }

      if (threadpool == null || threadpool instanceof FakeExecutor) {
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet);
        vc = vectorCollectionFactory.getVectorCollection(vecs, dm);
        knnVecList = VectorCollectionUtils.allNearestNeighbors(vc, vecs, k + 1);
      } else {
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadpool);
        vc = vectorCollectionFactory.getVectorCollection(vecs, dm, threadpool);
        knnVecList = VectorCollectionUtils.allNearestNeighbors(vc, vecs, k + 1, threadpool);
      }

    } catch (final InterruptedException ex) {
      Logger.getLogger(LSDBC.class.getName()).log(Level.SEVERE, null, ex);
    } catch (final ExecutionException ex) {
      Logger.getLogger(LSDBC.class.getName()).log(Level.SEVERE, null, ex);
    }

    // Sort
    final IndexTable indexTable = new IndexTable(knnVecList, new Comparator() {

      @Override
      public int compare(final Object o1, final Object o2) {
        final List<VecPaired<VecPaired<Vec, Integer>, Double>> l1 = (List<VecPaired<VecPaired<Vec, Integer>, Double>>) o1;
        final List<VecPaired<VecPaired<Vec, Integer>, Double>> l2 = (List<VecPaired<VecPaired<Vec, Integer>, Double>>) o2;

        return Double.compare(getEps(l1), getEps(l2));
      }
    });

    // Assign clusters, does very little computation. No need to parallelize
    // expandCluster
    Arrays.fill(designations, UNCLASSIFIED);
    int clusterID = 0;
    for (int i = 0; i < indexTable.length(); i++) {
      final int p = indexTable.index(i);
      if (designations[p] == UNCLASSIFIED && localMax(p, knnVecList)) {
        expandCluster(p, clusterID++, knnVecList, designations);
      }
    }

    return designations;
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int[] designations) {
    return cluster(dataSet, null, designations);
  }

  /**
   * Does the cluster assignment
   *
   * @param p
   *          the current index of a data point to assign to a cluster
   * @param clusterID
   *          the current cluster ID to assign
   * @param knnVecList
   *          the in order list of every index and its nearest neighbors
   * @param designations
   *          the array to store cluster designations in
   */
  private void expandCluster(final int p, final int clusterID,
      final List<List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>> knnVecList, final int[] designations) {
    designations[p] = clusterID;
    double pointEps;
    int n;
    final Stack<Integer> seeds = new Stack<Integer>();
    {
      final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors = knnVecList.get(p);
      for (int i = 1; i < neighbors.size(); i++) {
        addSeed(neighbors, i, designations, clusterID, seeds);
      }
      pointEps = getEps(neighbors);
      n = neighbors.get(k).length();
    }
    final double scale = Math.pow(2, alpha / n);

    while (!seeds.isEmpty()) {
      final int currentP = seeds.pop();
      final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors = knnVecList.get(currentP);
      final double currentPEps = getEps(neighbors);
      if (currentPEps <= scale * pointEps) {
        for (int i = 1; i < neighbors.size(); i++) {
          addSeed(neighbors, i, designations, clusterID, seeds);
        }
      }
    }
  }

  /**
   * Returns the scale value that will control how many points are added to a
   * cluster. Smaller values will create more, smaller clusters - and more
   * points will be labeled as noise. Larger values causes larger and fewer
   * clusters.
   *
   * @return the scale value to use
   */
  public double getAlpha() {
    return alpha;
  }

  /**
   * Returns the distance metric used when performing clustering.
   *
   * @return the distance metric used
   */
  @SuppressWarnings("unused")
  private DistanceMetric getDistanceMetric() {
    return dm;
  }

  /**
   * Convenience method. Gets the eps value for the given set of neighbors
   *
   * @param neighbors
   *          the set of neighbors, with index 0 being the point itself
   * @return the eps of the <tt>k</tt><sup>th</sup> neighbor
   */
  private double getEps(final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors) {
    return neighbors.get(k).getPair();
  }

  /**
   * Returns the number of neighbors that will be considered when clustering
   * data points
   *
   * @return the number of neighbors the algorithm will use
   */
  public int getNeighbors() {
    return k;
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
   * Returns true if the given point is a local maxima, meaning it is more dense
   * then the density of all its neighbors
   *
   * @param p
   *          the index of the data point in question
   * @param knnVecList
   *          the neighbor list
   * @return <tt>true</tt> if it is a local max, <tt> false</tt> otherwise.
   */
  private boolean localMax(final int p,
      final List<List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>> knnVecList) {
    final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors = knnVecList.get(p);
    final double myEps = getEps(neighbors);

    for (int i = 1; i < neighbors.size(); i++) {
      final int neighborP = neighbors.get(i).getVector().getPair();
      if (getEps(knnVecList.get(neighborP)) < myEps) {
        return false;
      }
    }

    return true;
  }

  /**
   * Sets the scale value that will control how many points are added to a
   * cluster. Smaller values will create more, smaller clusters - and more
   * points will be labeled as noise. Larger values causes larger and fewer
   * clusters.
   *
   * @param alpha
   *          the scale value to use
   */
  public void setAlpha(final double alpha) {
    if (alpha <= 0 || Double.isNaN(alpha) || Double.isInfinite(alpha)) {
      throw new ArithmeticException("Can not use the non positive scale value " + alpha);
    }
    this.alpha = alpha;
  }

  /**
   * Sets the distance metric used when performing clustering.
   *
   * @param dm
   *          the distance metric to use.
   */
  public void setDistanceMetric(final DistanceMetric dm) {
    if (dm != null) {
      this.dm = dm;
    }
  }

  /**
   * Sets the number of neighbors that will be considered when clustering data
   * points
   *
   * @param neighbors
   *          the number of neighbors the algorithm will use
   */
  public void setNeighbors(final int neighbors) {
    if (neighbors <= 0) {
      throw new ArithmeticException("Can not use a non positive number of neighbors");
    }
    k = neighbors;
  }

  /**
   * Sets the vector collection factory used for acceleration of neighbor
   * searches.
   *
   * @param vectorCollectionFactory
   *          the vector collection factory to use
   */
  public void setVectorCollectionFactory(
      final VectorCollectionFactory<VecPaired<Vec, Integer>> vectorCollectionFactory) {
    this.vectorCollectionFactory = vectorCollectionFactory;
  }

}
