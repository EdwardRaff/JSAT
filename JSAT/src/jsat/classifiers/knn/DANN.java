package jsat.classifiers.knn;

import static java.lang.Math.pow;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;

import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.distributions.Distribution;
import jsat.distributions.discrete.UniformDiscrete;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.EigenValueDecomposition;
import jsat.linear.Matrix;
import jsat.linear.RowColumnOps;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.VecPairedComparable;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.MahalanobisDistance;
import jsat.linear.vectorcollection.DefaultVectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionFactory;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.BoundedSortedList;
import jsat.utils.FakeExecutor;

/**
 * DANN is an implementation of Discriminant Adaptive Nearest Neighbor. DANN has
 * a fixed <i>O(n)</i> classification time. At each classification, DANN uses a
 * large set of points to iteratively create and adjust a distance metic that
 * reflects the separability of classes at a localized level. This increases the
 * work considerably over a normal {@link NearestNeighbour} classifier. The
 * localized metric is similar to the {@link MahalanobisDistance} <br>
 * Because DANN builds its own metric, it is not possible to provide one. The
 * {@link VectorCollectionFactory} allowed in the constructor is to accelerate
 * the first convergence step. In homogeneous areas of the data set, queries can
 * be answered in <i>O(log n)</i> if the vector collection supports it. <br>
 * <br>
 * See: Hastie, T.,&amp;Tibshirani, R. (1996). <i>Discriminant adaptive nearest
 * neighbor classification</i>. IEEE Transactions on Pattern Analysis and
 * Machine Intelligence, 18(6), 607–616. doi:10.1109/34.506411
 *
 * @author Edward Raff
 */
public class DANN implements Classifier, Parameterized {

  private static final long serialVersionUID = -272865942127664672L;
  /**
   * The default number of neighbors to use when building a metric is
   * {@value #DEFAULT_KN}.
   */
  public static final int DEFAULT_KN = 40;
  /**
   * The default number of neighbors to use when classifying is
   * {@value #DEFAULT_K}
   */
  public static final int DEFAULT_K = 1;
  /**
   * The default regularization used when building a metric is
   * {@value #DEFAULT_EPS}
   */
  public static final double DEFAULT_EPS = 1.0;
  /**
   * The default number of iterations for creating the metric is
   * {@value #DEFAULT_ITERATIONS}
   */
  public static final int DEFAULT_ITERATIONS = 1;

  /**
   * Guesses the distribution to use for the number of neighbors to consider
   *
   * @param d
   *          the dataset to get the guess for
   * @return the guess for the K parameter
   */
  public static Distribution guessK(final DataSet d) {
    return new UniformDiscrete(1, 25);
  }

  /**
   * Guesses the distribution to use for the number of neighbors to consider
   *
   * @param d
   *          the dataset to get the guess for
   * @return the guess for the Kn parameter
   */
  public static Distribution guessKn(final DataSet d) {
    return new UniformDiscrete(40, Math.max(d.getSampleSize() / 5, 50));
  }

  private int kn;
  private int k;

  private int maxIterations;

  private double eps;

  private final VectorCollectionFactory<VecPaired<Vec, Integer>> vcf;
  private CategoricalData predicting;

  /**
   * Vectors paired with their index in the original data set
   */
  private VectorCollection<VecPaired<Vec, Integer>> vc;

  private List<VecPaired<Vec, Integer>> vecList;

  /**
   * Creates a new DANN classifier
   */
  public DANN() {
    this(DEFAULT_KN, DEFAULT_K);
  }

  /**
   * Creates a new DANN classifier
   *
   * @param kn
   *          the number of neighbors to use in casting a net to build a new
   *          metric
   * @param k
   *          the number of neighbors to use with the final metric in
   *          classification
   */
  public DANN(final int kn, final int k) {
    this(kn, k, DEFAULT_EPS);
  }

  /**
   * Creates a new DANN classifier
   *
   * @param kn
   *          the number of neighbors to use in casting a net to build a new
   *          metric
   * @param k
   *          the number of neighbors to use with the final metric in
   *          classification
   * @param eps
   *          the regularization to add to the metric creation
   */
  public DANN(final int kn, final int k, final double eps) {
    this(kn, k, eps, new DefaultVectorCollectionFactory<VecPaired<Vec, Integer>>());
  }

  /**
   * Creates a new DANN classifier
   *
   * @param kn
   *          the number of neighbors to use in casting a net to build a new
   *          metric
   * @param k
   *          the number of neighbors to use with the final metric in
   *          classification
   * @param eps
   *          the regularization to add to the metric creation
   * @param maxIterations
   *          the maximum number of times to adjust the metric for each
   *          classification
   * @param vcf
   *          the default vector collection that will be used for initial
   *          neighbor search
   */
  public DANN(final int kn, final int k, final double eps, final int maxIterations,
      final VectorCollectionFactory<VecPaired<Vec, Integer>> vcf) {
    setK(k);
    setKn(kn);
    setEpsilon(eps);
    setMaxIterations(maxIterations);
    this.vcf = vcf;
  }

  /**
   * Creates a new DANN classifier
   *
   * @param kn
   *          the number of neighbors to use in casting a net to build a new
   *          metric
   * @param k
   *          the number of neighbors to use with the final metric in
   *          classification
   * @param eps
   *          the regularization to add to the metric creation
   * @param vcf
   *          the default vector collection that will be used for initial
   *          neighbor search
   */
  public DANN(final int kn, final int k, final double eps, final VectorCollectionFactory<VecPaired<Vec, Integer>> vcf) {
    this(kn, k, eps, DEFAULT_ITERATIONS, vcf);
  }

  private List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> brute(final Vec query, final Matrix sigma,
      final int num) {
    final Vec scartch0 = new DenseVector(query.length());
    final Vec scartch1 = new DenseVector(query.length());
    final BoundedSortedList<VecPairedComparable<VecPaired<Vec, Integer>, Double>> knn = new BoundedSortedList<VecPairedComparable<VecPaired<Vec, Integer>, Double>>(
        num, num);

    for (final VecPaired<Vec, Integer> v : vecList) {
      final double d = dist(sigma, query, v, scartch0, scartch1);
      knn.add(new VecPairedComparable<VecPaired<Vec, Integer>, Double>(v, d));
    }

    return (List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>) knn;
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    final CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());

    final int n = data.numNumericalValues();
    final Matrix sigma = Matrix.eye(n);
    final Matrix B = new DenseMatrix(n, n);
    final Matrix W = new DenseMatrix(n, n);
    final Vec query = data.getNumericalValues();

    final Vec scratch0 = new DenseVector(n);

    // TODO threadlocal DoubleList / DenseVec might be better in practice for
    // memory use
    final double[] weights = new double[kn];
    final double[] priors = new double[predicting.getNumOfCategories()];
    final int[] classCount = new int[priors.length];
    double sumOfWeights;
    final Vec mean = new DenseVector(sigma.rows());
    final Vec[] classMeans = new Vec[predicting.getNumOfCategories()];
    for (int i = 0; i < classMeans.length; i++) {
      classMeans[i] = new DenseVector(mean.length());
    }

    for (int iter = 0; iter < maxIterations; iter++) {
      // Zero out prev iter
      mean.zeroOut();
      Arrays.fill(priors, 0);
      Arrays.fill(classCount, 0);
      for (final Vec classMean : classMeans) {
        classMean.zeroOut();
      }
      sumOfWeights = 0;
      B.zeroOut();
      W.zeroOut();

      final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> vecs = iter == 0 ? vc.search(query, kn)
          : brute(query, sigma, kn);
      // Compute vector mean and weight sums, class weight sums, and the class
      // means
      final double h = vecs.get(vecs.size() - 1).getPair();
      for (int i = 0; i < vecs.size(); i++) {
        final VecPaired<? extends VecPaired<Vec, Integer>, Double> vec = vecs.get(i);
        // vecs contains the distances, we need the distance squared
        weights[i] = pow(pow(1 - pow(vec.getPair(), 2) / h, 3), 3);
        sumOfWeights += weights[i];
        mean.mutableAdd(vec);

        final int j = vec.getVector().getPair();
        priors[j] += weights[i];
        classMeans[j].mutableAdd(vec);
        classCount[j]++;
      }

      // Final divide for means and priors
      mean.mutableDivide(kn);
      for (int i = 0; i < classMeans.length; i++) {
        if (classCount[i] != 0.0) {
          classMeans[i].mutableDivide(classCount[i]);
        }
        priors[i] /= sumOfWeights;
      }

      // Compute B & W
      for (int j = 0; j < classMeans.length; j++) {
        // One line for B's work
        if (priors[j] > 0) {
          classMeans[j].copyTo(scratch0);
          scratch0.mutableSubtract(mean);

          Matrix.OuterProductUpdate(B, scratch0, scratch0, priors[j]);

          // Loop for W's work
          for (int i = 0; i < vecs.size(); i++) {
            final VecPaired<? extends VecPaired<Vec, Integer>, Double> x = vecs.get(i);
            if (x.getVector().getPair() == j) {
              x.copyTo(scratch0);
              scratch0.mutableSubtract(classMeans[j]);

              Matrix.OuterProductUpdate(W, scratch0, scratch0, weights[i]);
            }
          }
        }
      }

      // Final divide for W
      W.mutableMultiply(1.0 / sumOfWeights);

      RowColumnOps.addDiag(B, 0, B.rows(), eps);

      // Check, if there is a prior of 1, nothing will ever be updated.
      // Might as well return
      for (int i = 0; i < priors.length; i++) {
        if (priors[i] == 1.0) {
          cr.setProb(i, 1.0);
          return cr;
        }
      }

      final EigenValueDecomposition evd = new EigenValueDecomposition(W);
      final Matrix D = evd.getD();
      for (int i = 0; i < D.rows(); i++) {
        D.set(i, i, pow(D.get(i, i), -0.5));
      }
      final Matrix VT = evd.getVT();
      final Matrix WW = VT.transposeMultiply(D).multiply(VT);

      sigma.zeroOut();
      WW.multiply(B).multiply(WW, sigma);
    }

    final List<? extends VecPaired<? extends VecPaired<Vec, Integer>, Double>> knn = brute(query, sigma, k);

    for (final VecPaired<? extends VecPaired<Vec, Integer>, Double> nn : knn) {
      cr.incProb(nn.getVector().getPair(), 1.0);
    }

    cr.normalize();
    return cr;
  }

  @Override
  public Classifier clone() {
    final DANN clone = new DANN(kn, k, maxIterations, vcf.clone());

    if (predicting != null) {
      clone.predicting = predicting.clone();
    }
    if (vc != null) {
      clone.vc = vc.clone();
    }
    if (vecList != null) {
      clone.vecList = new ArrayList<VecPaired<Vec, Integer>>(vecList);
    }

    return clone;
  }

  private double dist(final Matrix sigma, final Vec query, final Vec mean, final Vec scratch0, final Vec scartch1) {
    query.copyTo(scratch0);
    scratch0.mutableSubtract(mean);

    scartch1.zeroOut();
    sigma.multiply(scratch0, 1.0, scartch1);

    return scratch0.dot(scartch1);
  }

  /**
   * Returns the regularization parameter that is applied to the diagonal of the
   * matrix when creating each new metric.
   *
   * @return the regularization used.
   */
  public double getEpsilon() {
    return eps;
  }

  /**
   * Returns the number of nearest neighbors to use when predicting
   *
   * @return the number of neighbors used for classification
   */
  public int getK() {
    return k;
  }

  /**
   * Returns the number of nearest neighbors to use when adapting the distance
   * metric
   *
   * @return the number of neighbors used to adapt the metric
   */
  public int getKn() {
    return kn;
  }

  /**
   * Returns the number of times the distance metric will be updated.
   *
   * @return the number of iterations the metric will be updated
   */
  public int getMaxIterations() {
    return maxIterations;
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
   * Sets the regularization to apply the the diagonal of the scatter matrix
   * when creating each new metric.
   *
   * @param eps
   *          the regularization value
   */
  public void setEpsilon(final double eps) {
    if (eps < 0 || Double.isInfinite(eps) || Double.isNaN(eps)) {
      throw new ArithmeticException("Regularization must be a positive value");
    }
    this.eps = eps;
  }

  /**
   * Sets the number of nearest neighbors to use when predicting
   *
   * @param k
   *          the number of neighbors
   */
  public void setK(final int k) {
    if (k < 1) {
      throw new ArithmeticException("Number of neighbors must be positive");
    }
    this.k = k;
  }

  /**
   * Sets the number of nearest neighbors to use when adapting the distance
   * metric. At each iteration of the algorithm, a new distance metric will be
   * created. A larger number of neighbors is used to create a net of points,
   * around which the metric will be adapted.
   *
   * @param kn
   *          the number of neighbors to use
   */
  public void setKn(final int kn) {
    if (kn < 2) {
      throw new ArithmeticException("At least 2 neighbors are needed to adapat the metric");
    }
    this.kn = kn;
  }

  /**
   * Sets the number of times a new distance metric will be created for each
   * query. The metric should converge quickly. For this reason, and do to a
   * lack of performance difference, it is highly recommended to use the default
   * of 1 iteration.
   *
   * @param maxIterations
   *          the maximum number of times the metric will be updated
   */
  public void setMaxIterations(final int maxIterations) {
    if (maxIterations < 1) {
      throw new RuntimeException("At least one iteration must be performed");
    }
    this.maxIterations = maxIterations;
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    trainC(dataSet, null);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    predicting = dataSet.getPredicting();
    vecList = new ArrayList<VecPaired<Vec, Integer>>(dataSet.getSampleSize());
    for (int i = 0; i < dataSet.getSampleSize(); i++) {
      vecList.add(
          new VecPaired<Vec, Integer>(dataSet.getDataPoint(i).getNumericalValues(), dataSet.getDataPointCategory(i)));
    }
    if (threadPool == null || threadPool instanceof FakeExecutor) {
      vc = vcf.getVectorCollection(vecList, new EuclideanDistance());
    } else {
      vc = vcf.getVectorCollection(vecList, new EuclideanDistance(), threadPool);
    }
  }
}
