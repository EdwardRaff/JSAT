package jsat.regression;

import static java.lang.Math.pow;
import static jsat.linear.DenseVector.toDenseVec;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.LUPDecomposition;
import jsat.linear.Matrix;
import jsat.linear.SingularValueDecomposition;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.SystemInfo;

/**
 * An implementation of Ordinary Kriging with support for a uniform error
 * measurement. When an {@link #getMeasurementError() error} value is applied,
 * Kriging becomes equivalent to Gaussian Processes Regression.
 *
 * @author Edward Raff
 */
public class OrdinaryKriging implements Regressor, Parameterized {

  public static class PowVariogram implements Variogram {

    private double alpha;
    private final double beta;

    public PowVariogram() {
      this(1.5);
    }

    public PowVariogram(final double beta) {
      this.beta = beta;
    }

    @Override
    public Variogram clone() {
      final PowVariogram clone = new PowVariogram(beta);
      clone.alpha = alpha;

      return clone;
    }

    @Override
    public void train(final RegressionDataSet dataSet, final double nugget) {
      final int npt = dataSet.getSampleSize();
      double num = 0, denom = 0;
      final double nugSqrd = nugget * nugget;

      for (int i = 0; i < npt; i++) {
        final Vec xi = dataSet.getDataPoint(i).getNumericalValues();
        final double yi = dataSet.getTargetValue(i);
        for (int j = i + 1; j < npt; j++) {
          final Vec xj = dataSet.getDataPoint(j).getNumericalValues();
          final double yj = dataSet.getTargetValue(j);
          final double rb = pow(xi.pNormDist(2, xj), beta);

          num += rb * (0.5 * pow(yi - yj, 2) - nugSqrd);
          denom += rb * rb;
        }
      }
      alpha = num / denom;
    }

    @Override
    public double val(final double r) {
      return alpha * pow(r, beta);
    }
  }

  public static interface Variogram extends Cloneable {

    public Variogram clone();

    /**
     * Sets the values of the variogram
     *
     * @param dataSet
     *          the data set to learn the parameters from
     * @param nugget
     *          the nugget value to add tot he variogram, may be ignored if the
     *          variogram want to fit it automatically
     */
    public void train(RegressionDataSet dataSet, double nugget);

    /**
     * Returns the output of the variogram for the given input
     *
     * @param r
     *          the input value
     * @return the output of the variogram
     */
    public double val(double r);
  }

  private static final long serialVersionUID = -5774553215322383751L;
  /**
   * The default nugget value is {@value #DEFAULT_NUGGET}
   */
  public static final double DEFAULT_NUGGET = 0.1;
  /**
   * The default error value is {@link #DEFAULT_ERROR}
   */
  public static final double DEFAULT_ERROR = 0.1;
  private final Variogram vari;

  /**
   * The weight values for each data point
   */
  private Vec X;
  private RegressionDataSet dataSet;

  private double errorSqrd;

  private double nugget;

  List<Parameter> params = Collections.unmodifiableList(Parameter.getParamsFromMethods(this));

  private final Map<String, Parameter> paramMap = Parameter.toParameterMap(params);

  /**
   * Creates a new Ordinary Kriging with a small error value using the
   * {@link PowVariogram power} variogram.
   */
  public OrdinaryKriging() {
    this(new PowVariogram());
  }

  /**
   * Creates a new Ordinary Kriging with a small error value
   *
   * @param vari
   *          the variogram to fit to the data
   */
  public OrdinaryKriging(final Variogram vari) {
    this(vari, DEFAULT_ERROR);
  }

  /**
   * Creates a new Ordinary Kriging
   *
   * @param vari
   *          the variogram to fit to the data
   * @param error
   *          the global measurement error
   */
  public OrdinaryKriging(final Variogram vari, final double error) {
    this(vari, error, DEFAULT_NUGGET);
  }

  /**
   * Creates a new Ordinary Kriging.
   *
   * @param vari
   *          the variogram to fit to the data
   * @param error
   *          the global measurement error
   * @param nugget
   *          the nugget value to add to the variogram
   */
  public OrdinaryKriging(final Variogram vari, final double error, final double nugget) {
    this.vari = vari;
    setMeasurementError(error);
    this.nugget = nugget;
  }

  @Override
  public OrdinaryKriging clone() {
    final OrdinaryKriging clone = new OrdinaryKriging(vari.clone());

    clone.setMeasurementError(getMeasurementError());
    clone.setNugget(getNugget());
    if (X != null) {
      clone.X = X.clone();
    }
    if (dataSet != null) {
      clone.dataSet = dataSet;
    }

    return clone;
  }

  /**
   * Returns the measurement error used for Kriging, which is equivalent to
   * altering the diagonal values of the covariance. While the measurement
   * errors could be per data point, this implementation provides only a global
   * error. If the error is set to zero, it will perfectly interpolate all data
   * points.
   *
   * @return the global error used for the data
   */
  public double getMeasurementError() {
    return Math.sqrt(errorSqrd);
  }

  /**
   * Returns the nugget value passed to the variogram during training. The
   * nugget allows the variogram to start from a non-zero value, and is
   * equivalent to alerting the off diagonal values of the covariance.
   *
   * @return the nugget added to the variogram
   */
  public double getNugget() {
    return nugget;
  }

  @Override
  public Parameter getParameter(final String paramName) {
    return paramMap.get(paramName);
  }

  @Override
  public List<Parameter> getParameters() {
    return params;
  }

  @Override
  public double regress(final DataPoint data) {
    final Vec x = data.getNumericalValues();
    final int npt = X.length() - 1;
    final double[] distVals = new double[npt + 1];
    for (int i = 0; i < npt; i++) {
      distVals[i] = vari.val(x.pNormDist(2, dataSet.getDataPoint(i).getNumericalValues()));
    }
    distVals[npt] = 1.0;

    return X.dot(toDenseVec(distVals));
  }

  /**
   * Sets the measurement error used for Kriging, which is equivalent to
   * altering the diagonal values of the covariance. While the measurement
   * errors could be per data point, this implementation provides only a global
   * error. If the error is set to zero, it will perfectly interpolate all data
   * points. <br>
   * Increasing the error smooths the interpolation, and has a large impact on
   * the regression results.
   *
   * @param error
   *          the measurement error for all data points
   */
  public void setMeasurementError(final double error) {
    errorSqrd = error * error;
  }

  /**
   * Sets the nugget value passed to the variogram during training. The nugget
   * allows the variogram to start from a non-zero value, and is equivalent to
   * alerting the off diagonal values of the covariance. <br>
   * Altering the nugget value has only a minor impact on the output
   *
   * @param nugget
   *          the new nugget value
   * @throws ArithmeticException
   *           if a negative nugget value is provided
   */
  public void setNugget(final double nugget) {
    if (nugget < 0 || Double.isNaN(nugget) || Double.isInfinite(nugget)) {
      throw new ArithmeticException("Nugget must be a positive value");
    }
    this.nugget = nugget;
  }

  private void setUpVectorMatrix(final int N, final RegressionDataSet dataSet, final Matrix V, final Vec Y) {
    for (int i = 0; i < N; i++) {
      final DataPoint dpi = dataSet.getDataPoint(i);
      final Vec xi = dpi.getNumericalValues();
      for (int j = 0; j < N; j++) {
        final Vec xj = dataSet.getDataPoint(j).getNumericalValues();
        final double val = vari.val(xi.pNormDist(2, xj));
        V.set(i, j, val);
        V.set(j, i, val);
      }
      V.set(i, N, 1.0);
      V.set(N, i, 1.0);
      Y.set(i, dataSet.getTargetValue(i));
    }
    V.set(N, N, 0);
  }

  private void setUpVectorMatrix(final int N, final RegressionDataSet dataSet, final Matrix V, final Vec Y,
      final ExecutorService threadPool) {
    int pos = 0;
    final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);

    while (pos < SystemInfo.LogicalCores) {
      final int id = pos++;
      threadPool.submit(new Runnable() {

        @Override
        public void run() {
          for (int i = id; i < N; i += SystemInfo.LogicalCores) {
            final DataPoint dpi = dataSet.getDataPoint(i);
            final Vec xi = dpi.getNumericalValues();
            for (int j = 0; j < N; j++) {
              final Vec xj = dataSet.getDataPoint(j).getNumericalValues();
              final double val = vari.val(xi.pNormDist(2, xj));
              V.set(i, j, val);
              V.set(j, i, val);
            }
            V.set(i, N, 1.0);
            V.set(N, i, 1.0);
            Y.set(i, dataSet.getTargetValue(i));
          }
          latch.countDown();
        }
      });
    }

    V.set(N, N, 0);

    while (pos++ < SystemInfo.LogicalCores) {
      latch.countDown();
    }

    try {
      latch.await();
    } catch (final InterruptedException ex) {
      Logger.getLogger(OrdinaryKriging.class.getName()).log(Level.SEVERE, null, ex);
    }
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void train(final RegressionDataSet dataSet) {
    train(dataSet, null);
  }

  @Override
  public void train(final RegressionDataSet dataSet, final ExecutorService threadPool) {
    this.dataSet = dataSet;
    /**
     * Size of the data set
     */
    final int N = dataSet.getSampleSize();
    /**
     * Stores the target values
     */
    final Vec Y = new DenseVector(N + 1);

    final Matrix V = new DenseMatrix(N + 1, N + 1);

    vari.train(dataSet, nugget);

    if (threadPool == null) {
      setUpVectorMatrix(N, dataSet, V, Y);
    } else {
      setUpVectorMatrix(N, dataSet, V, Y, threadPool);
    }

    for (int i = 0; i < N; i++) {
      V.increment(i, i, -errorSqrd);
    }

    LUPDecomposition lup;
    if (threadPool == null) {
      lup = new LUPDecomposition(V);
    } else {
      lup = new LUPDecomposition(V, threadPool);
    }

    X = lup.solve(Y);
    if (Double.isNaN(lup.det()) || Math.abs(lup.det()) < 1e-5) {
      final SingularValueDecomposition svd = new SingularValueDecomposition(V);
      X = svd.solve(Y);
    }
  }

}
