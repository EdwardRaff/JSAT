package jsat.clustering;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;

import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.clustering.evaluation.IntraClusterSumEvaluation;
import jsat.clustering.evaluation.intra.SumOfSqrdPairwiseDistances;
import jsat.clustering.kmeans.HamerlyKMeans;
import jsat.clustering.kmeans.KMeans;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.SingularValueDecomposition;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.math.OnLineStatistics;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.random.XORWOW;

/**
 * This class implements a method for estimating the number of clusters in a
 * data set called the Gap Statistic. It works by sampling new datasets from a
 * uniform random space, and comparing the sum of squared pairwise distances
 * between the sampled data and the real data. The number of samples has a
 * significant impact on runtime, and is controlled via
 * {@link #setSamples(int) }. <br>
 * The Gap method can be applied to any distance metric and any clustering
 * algorithm. However, it is significantly faster for the
 * {@link EuclideanDistance} and was developed with the {@link KMeans}
 * algorithm. Thus that combination is the default when using the no argument
 * constructor. <br>
 * <br>
 * A slight deviation in the implementation from the original paper exists. The
 * original paper specifies that the smallest {@code K} satisfying
 * {@link #getGap() Gap}(K) &ge; Gap(K+1) - {@link #getElogWkStndDev() sd}(K+1)
 * what the value of {@code K} to use. Instead the condition used is the
 * smallest {@code K} such that Gap(K) &ge; Gap(K+1)- sd(K+1) and Gap(K) &gt; 0.
 * <br>
 * In addition, if no value of {@code K} satisfies the condition, the largest
 * value of Gap(K) will be used. <br>
 * <br>
 * Note, by default this implementation uses a heuristic for the max value of
 * {@code K} that is capped at 100 when using the
 * {@link #cluster(jsat.DataSet) } type methods.<br>
 * Note: when called with the desired number of clusters, the result of the base
 * clustering algorithm be returned directly. <br>
 * <br>
 * See: Tibshirani, R., Walther, G.,&amp;Hastie, T. (2001). <i>Estimating the
 * number of clusters in a data set via the gap statistic</i>. Journal of the
 * Royal Statistical Society: Series B (Statistical Methodology), 63(2),
 * 411–423. doi:10.1111/1467-9868.00293
 *
 * @author Edward Raff
 */
public class GapStatistic extends KClustererBase implements Parameterized {

  private static final long serialVersionUID = 8893929177942856618L;
  @ParameterHolder
  private final KClusterer base;
  private int B;
  private DistanceMetric dm;
  private boolean PCSampling;

  private double[] ElogW;
  private double[] logW;
  private double[] gap;
  private double[] s_k;

  /**
   * Creates a new Gap clusterer using k-means as the base clustering algorithm
   */
  public GapStatistic() {
    this(new HamerlyKMeans());
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public GapStatistic(final GapStatistic toCopy) {
    base = toCopy.base.clone();
    B = toCopy.B;
    dm = toCopy.dm.clone();
    PCSampling = toCopy.PCSampling;
    if (toCopy.ElogW != null) {
      ElogW = Arrays.copyOf(toCopy.ElogW, toCopy.ElogW.length);
    }
    if (toCopy.logW != null) {
      logW = Arrays.copyOf(toCopy.logW, toCopy.logW.length);
    }
    if (toCopy.gap != null) {
      gap = Arrays.copyOf(toCopy.gap, toCopy.gap.length);
    }
    if (toCopy.s_k != null) {
      s_k = Arrays.copyOf(toCopy.s_k, toCopy.s_k.length);
    }
  }

  /**
   * Creates a new Gap clusterer using the base clustering algorithm given.
   *
   * @param base
   *          the base clustering method to use for any individual number of
   *          clusters
   */
  public GapStatistic(final KClusterer base) {
    this(base, false);
  }

  /**
   * Creates a new Gap clsuterer using the base clustering algorithm given.
   *
   * @param base
   *          the base clustering method to use for any individual number of
   *          clusters
   * @param PCSampling
   *          {@code true} if the Gap statistic should be computed from a PCA
   *          transformed space, or {@code false} to go with the uniform
   *          bounding hyper cube.
   */
  public GapStatistic(final KClusterer base, final boolean PCSampling) {
    this(base, PCSampling, 10, new EuclideanDistance());
  }

  /**
   * Creates a new Gap clsuterer using the base clustering algorithm given.
   *
   * @param base
   *          the base clustering method to use for any individual number of
   *          clusters
   * @param PCSampling
   *          {@code true} if the Gap statistic should be computed from a PCA
   *          transformed space, or {@code false} to go with the uniform
   *          bounding hyper cube.
   * @param B
   *          the number of datasets to sample
   * @param dm
   *          the distance metric to evaluate with
   */
  public GapStatistic(final KClusterer base, final boolean PCSampling, final int B, final DistanceMetric dm) {
    this.base = base;
    setSamples(B);
    setDistanceMetric(dm);
    setPCSampling(PCSampling);
  }

  @Override
  public GapStatistic clone() {
    return new GapStatistic(this);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final ExecutorService threadpool, final int[] designations) {
    return cluster(dataSet, 1, (int) Math.min(Math.max(Math.sqrt(dataSet.getSampleSize()), 10), 100), threadpool,
        designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int clusters, final ExecutorService threadpool,
      final int[] designations) {
    return base.cluster(dataSet, clusters, threadpool, designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int lowK, final int highK, final ExecutorService threadpool,
      int[] designations) {
    final int D = dataSet.getNumNumericalVars();
    final int N = dataSet.getSampleSize();

    if (designations == null || designations.length < N) {
      designations = new int[N];
      // TODO we dont need all values in [1, lowK-1) in order to get the gap
      // statistic for [lowK, highK]. So lets not do that extra work.
    }

    logW = new double[highK - 1];
    ElogW = new double[highK - 1];
    gap = new double[highK - 1];
    s_k = new double[highK - 1];

    final IntraClusterSumEvaluation ssd = new IntraClusterSumEvaluation(new SumOfSqrdPairwiseDistances(dm));

    // Step 1: Cluster the observed data
    Arrays.fill(designations, 0);
    logW[0] = Math.log(ssd.evaluate(designations, dataSet));// base case
    for (int k = 2; k < highK; k++) {
      designations = base.cluster(dataSet, k, threadpool, designations);
      logW[k - 1] = Math.log(ssd.evaluate(designations, dataSet));
    }
    // Step 2:
    // use online statistics and run through all K for each B, so that we
    // minimize the memory use
    final OnLineStatistics[] expected = new OnLineStatistics[highK - 1];
    for (int i = 0; i < expected.length; i++) {
      expected[i] = new OnLineStatistics();
    }

    // dataset object we will reuse
    final SimpleDataSet Xp = new SimpleDataSet(new CategoricalData[0], D);
    for (int i = 0; i < N; i++) {
      Xp.add(new DataPoint(new DenseVector(D)));
    }

    final Random rand = new XORWOW();

    // info needed for sampling
    // min/max for each row/col to smaple uniformly from
    final double[] min = new double[D];
    final double[] max = new double[D];
    Arrays.fill(min, Double.POSITIVE_INFINITY);
    Arrays.fill(max, Double.NEGATIVE_INFINITY);
    final Matrix V_T;// the V^T from [U, D, V] of SVD decomposation
    if (PCSampling) {
      final SingularValueDecomposition svd = new SingularValueDecomposition(dataSet.getDataMatrix());
      // X' = X V , from generation strategy (b)
      final Matrix tmp = dataSet.getDataMatrixView().multiply(svd.getV());

      for (int i = 0; i < tmp.rows(); i++) {
        for (int j = 0; j < tmp.cols(); j++) {
          min[j] = Math.min(tmp.get(i, j), min[j]);
          max[j] = Math.max(tmp.get(i, j), max[j]);
        }
      }
      V_T = svd.getV().transpose();
    } else {
      V_T = null;
      final OnLineStatistics[] columnStats = dataSet.getOnlineColumnStats(false);
      for (int i = 0; i < D; i++) {
        min[i] = columnStats[i].getMin();
        max[i] = columnStats[i].getMax();
      }
    }

    // generate B reference datasets
    for (int b = 0; b < B; b++) {
      for (int i = 0; i < N; i++)// sample
      {
        final Vec xp = Xp.getDataPoint(i).getNumericalValues();
        for (int j = 0; j < D; j++) {
          xp.set(j, (max[j] - min[j]) * rand.nextDouble() + min[j]);
        }
      }

      if (PCSampling) // project if wanted
      {
        // Finally we back-transform via Z = Z' V^T to give reference data Z
        // TODO batch as a matrix matrix op would be faster, but use more memory
        final Vec tmp = new DenseVector(D);
        for (int i = 0; i < N; i++) {
          final Vec xp = Xp.getDataPoint(i).getNumericalValues();
          tmp.zeroOut();
          xp.multiply(V_T, tmp);
          tmp.copyTo(xp);
        }
      }

      // cluster each one
      Arrays.fill(designations, 0);
      expected[0].add(Math.log(ssd.evaluate(designations, Xp)));// base case
      for (int k = 2; k < highK; k++) {
        designations = base.cluster(Xp, k, threadpool, designations);
        expected[k - 1].add(Math.log(ssd.evaluate(designations, Xp)));
      }
    }

    // go through and copmute gap
    int k_first = -1;
    int biggestGap = 0;// used as a fall back incase the original condition
                       // can't be satisfied in the specified range
    for (int i = 0; i < gap.length; i++) {
      gap[i] = (ElogW[i] = expected[i].getMean()) - logW[i];
      s_k[i] = expected[i].getStandardDeviation() * Math.sqrt(1 + 1.0 / B);
      // check original condition first
      final int k = i + 1;
      if (i > 0 && lowK <= k && k <= highK) {
        if (k_first == -1 && gap[i - 1] >= gap[i] - s_k[i] && gap[i - 1] > 0) {
          k_first = k - 1;
        }
      }
      // check backup
      if (gap[i] > biggestGap && lowK <= k && k <= highK) {
        biggestGap = i;
      }
    }

    if (k_first == -1) {// never satisfied our conditions?
      k_first = biggestGap + 1;// Maybe we should go back and pick the best gap
                               // k we can find?
    }
    if (k_first == 1) // easy case
    {
      Arrays.fill(designations, 0);
      return designations;
    }

    return base.cluster(dataSet, k_first, threadpool, designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int lowK, final int highK, final int[] designations) {
    return cluster(dataSet, lowK, highK, null, designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int clusters, final int[] designations) {
    return base.cluster(dataSet, clusters, designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int[] designations) {
    return cluster(dataSet, 1, (int) Math.min(Math.max(Math.sqrt(dataSet.getSampleSize()), 10), 100), designations);
  }

  /**
   *
   * @return the distance metric used for evaluation
   */
  public DistanceMetric getDistanceMetric() {
    return dm;
  }

  /**
   * Returns the array of expected <i>E[log(W<sub>k</sub>)]</i> scores computed
   * from sampling new data sets. <br>
   * Index {@code i} of the returned array indicates the gap score for using
   * {@code i+1} clusters. A value of {@link Double#NaN} if the score was not
   * computed for that value of {@code K}
   *
   * @return the array of sampled expected scores from the last run, or
   *         {@code null} if the algorithm hasn't been run yet
   */
  public double[] getElogW() {
    return ElogW;
  }

  /**
   * Returns the array of standard deviations from the samplings used to compute
   * {@link #getElogWkStndDev() }, multiplied by <i>sqrt(1+1/B)</i>. <br>
   * Index {@code i} of the returned array indicates the gap score for using
   * {@code i+1} clusters. A value of {@link Double#NaN} if the score was not
   * computed for that value of {@code K}
   *
   * @return the array of standard deviations from the last run, or {@code null}
   *         if the algorithm hasn't been run yet
   */
  public double[] getElogWkStndDev() {
    return s_k;
  }

  /**
   * Returns the array of gap statistic values. Index {@code i} of the returned
   * array indicates the gap score for using {@code i+1} clusters. A value of
   * {@link Double#NaN} if the score was not computed for that value of
   * {@code K}
   *
   * @return the array of gap statistic values computed, or {@code null} if the
   *         algorithm hasn't been run yet.
   */
  public double[] getGap() {
    return gap;
  }

  /**
   * Returns the array of empirical <i>log(W<sub>k</sub>)</i> scores computed
   * from the data set last clustered. <br>
   * Index {@code i} of the returned array indicates the gap score for using
   * {@code i+1} clusters. A value of {@link Double#NaN} if the score was not
   * computed for that value of {@code K}
   *
   * @return the array of empirical scores from the last run, or {@code null} if
   *         the algorithm hasn't been run yet
   */
  public double[] getLogW() {
    return logW;
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
   *
   * @return the number of data sets sampled
   */
  public int getSamples() {
    return B;
  }

  /**
   *
   * @return {@code true} to sample from the projected data, {@code
   * false} to do the default and sample from the bounding hyper-cube.
   */
  public boolean isPCSampling() {
    return PCSampling;
  }

  /**
   * Sets the distance metric to use when evaluating a clustering algorithm
   *
   * @param dm
   *          the distance metric to use
   */
  public void setDistanceMetric(final DistanceMetric dm) {
    this.dm = dm;
  }

  /**
   * By default the null distribution is sampled from the bounding hyper-cube of
   * the dataset. The accuracy of the sampling can be made more accurate (and
   * invariant) by sampling the null distribution based on the principal
   * components of the dataset. This will also increase the runtime of the
   * algorithm.
   *
   * @param PCSampling
   *          {@code true} to sample from the projected data, {@code
   * false} to do the default and sample from the bounding hyper-cube.
   */
  public void setPCSampling(final boolean PCSampling) {
    this.PCSampling = PCSampling;
  }

  /**
   * The Gap statistic is measured by sampling from a reference distribution and
   * comparing with the given data set. This controls the number of sample
   * datasets to draw and evaluate.
   *
   * @param B
   *          the number of data sets to sample
   */
  public void setSamples(final int B) {
    if (B <= 0) {
      throw new IllegalArgumentException("sample size must be positive, not " + B);
    }
    this.B = B;
  }
}
