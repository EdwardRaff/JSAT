package jsat.regression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.kernels.KernelTrick;
import jsat.distributions.kernels.RBFKernel;
import jsat.linear.CholeskyDecomposition;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;

/**
 * A kernelized implementation of Ridge Regression. Ridge Regression is
 * equivalent to {@link MultipleLinearRegression} with an added L<sub>2</sub>
 * penalty for the weight vector. <br>
 * <br>
 * This algorithm is very expensive to compute O(n<sup>3</sup>), where n is the
 * number of training points.
 *
 * @author Edward Raff
 */
public class KernelRidgeRegression implements Regressor, Parameterized {

  private static final long serialVersionUID = 6275333785663250072L;

  /**
   * Guesses the distribution to use for the &lambda; parameter
   *
   * @param d
   *          the dataset to get the guess for
   * @return the guess for the &lambda; parameter
   */
  public static Distribution guessLambda(final DataSet d) {
    return new LogUniform(1e-7, 1e-2);
  }

  private double lambda;
  @ParameterHolder
  private KernelTrick k;
  private List<Vec> vecs;

  private double[] alphas;

  /**
   * Creates a new Kernel Ridge Regression learner that uses an RBF kernel
   */
  public KernelRidgeRegression() {
    this(1e-6, new RBFKernel());
  }

  /**
   * Creates a new Kernel Ridge Regression learner
   *
   * @param lambda
   *          the regularization parameter
   * @param kernel
   *          the kernel to use
   * @see #setLambda(double)
   */
  public KernelRidgeRegression(final double lambda, final KernelTrick kernel) {
    setLambda(lambda);
    setKernel(kernel);
  }

  /**
   * Copy Constructor
   *
   * @param toCopy
   *          the object to copy
   */
  protected KernelRidgeRegression(final KernelRidgeRegression toCopy) {
    this(toCopy.lambda, toCopy.getKernel().clone());
    if (toCopy.alphas != null) {
      alphas = Arrays.copyOf(toCopy.alphas, toCopy.alphas.length);
    }
    if (toCopy.vecs != null) {
      vecs = new ArrayList<Vec>(toCopy.vecs);
    }
  }

  @Override
  public KernelRidgeRegression clone() {
    return new KernelRidgeRegression(this);
  }

  /**
   * Returns the kernel in use
   *
   * @return the kernel in use
   */
  public KernelTrick getKernel() {
    return k;
  }

  /**
   * Returns the regularization constant in use
   *
   * @return the regularization constant in use
   */
  public double getLambda() {
    return lambda;
  }

  @Override
  public Parameter getParameter(final String paramName) {
    return Parameter.toParameterMap(getParameters()).get(paramName);
  }

  @Override
  public List<Parameter> getParameters() {
    return Parameter.getParamsFromMethods(this);
  }

  @Override
  public double regress(final DataPoint data) {
    final Vec x = data.getNumericalValues();
    double score = 0;
    for (int i = 0; i < alphas.length; i++) {
      score += alphas[i] * k.eval(vecs.get(i), x);
    }
    return score;
  }

  /**
   * Sets the kernel trick to use
   *
   * @param k
   *          the kernel to use
   */
  public void setKernel(final KernelTrick k) {
    this.k = k;
  }

  /**
   * Sets the regularization parameter used. The value of lambda depends on the
   * data set and kernel used, with easier problems using smaller lambdas.
   *
   * @param lambda
   *          the positive regularization constant in (0, Inf)
   */
  public void setLambda(final double lambda) {
    if (Double.isNaN(lambda) || Double.isInfinite(lambda) || lambda <= 0) {
      throw new IllegalArgumentException("lambda must be a positive constant, not " + lambda);
    }
    this.lambda = lambda;
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void train(final RegressionDataSet dataSet) {
    train(dataSet, new FakeExecutor());
  }

  @Override
  public void train(final RegressionDataSet dataSet, final ExecutorService threadPool) {
    final int N = dataSet.getSampleSize();
    vecs = new ArrayList<Vec>(N);
    // alphas initalized later
    final Vec Y = dataSet.getTargetValues();
    for (int i = 0; i < N; i++) {
      vecs.add(dataSet.getDataPoint(i).getNumericalValues());
    }

    final Matrix K = new DenseMatrix(N, N);
    final CountDownLatch cdl = new CountDownLatch(SystemInfo.LogicalCores);

    for (int id = 0; id < SystemInfo.LogicalCores; id++) {
      final int ID = id;
      threadPool.submit(new Runnable() {
        @Override
        public void run() {
          for (int i = ID; i < N; i += SystemInfo.LogicalCores) {
            K.set(i, i, k.eval(vecs.get(i), vecs.get(i)) + lambda);// diagonal
                                                                   // values
            for (int j = i + 1; j < N; j++) {
              final double K_ij = k.eval(vecs.get(i), vecs.get(j));
              K.set(i, j, K_ij);
              K.set(j, i, K_ij);
            }
          }

          cdl.countDown();
        }
      });
    }
    try {
      cdl.await();
    } catch (final InterruptedException ex) {
      Logger.getLogger(KernelRidgeRegression.class.getName()).log(Level.SEVERE, null, ex);
    }

    CholeskyDecomposition cd;
    if (threadPool instanceof FakeExecutor) {
      cd = new CholeskyDecomposition(K);
    } else {
      cd = new CholeskyDecomposition(K, threadPool);
    }
    final Vec alphaTmp = cd.solve(Y);
    alphas = alphaTmp.arrayCopy();
  }
}
