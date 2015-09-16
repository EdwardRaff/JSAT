package jsat.distributions.multivariate;

import static java.lang.Math.PI;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.pow;
import static jsat.linear.MatrixStatistics.covarianceMatrix;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import jsat.classifiers.DataPoint;
import jsat.linear.CholeskyDecomposition;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.LUPDecomposition;
import jsat.linear.Matrix;
import jsat.linear.MatrixStatistics;
import jsat.linear.SingularValueDecomposition;
import jsat.linear.Vec;

/**
 * Class for the multivariate Normal distribution. It is often called the
 * Multivariate Gaussian distribution.
 *
 * @author Edward Raff
 */
public class NormalM extends MultivariateDistributionSkeleton {

  private static final long serialVersionUID = -7043369396743253382L;
  /**
   * When computing the PDF of some x, part of the equation is only dependent on
   * the covariance matrix. This part is
   * 
   * <pre>
   *       -k
   *       --          -1
   *        2          --
   * /  __\             2
   * \2 ||/   (|Sigma|)
   * </pre>
   * 
   * where k is the dimension, Sigma is the covariance matrix, and || denotes
   * the determinant. <br>
   * Taking the negative log of this gives
   * 
   * <pre>
   *         /  __\
   * (-k) log\2 ||/ - log(|Sigma|)
   * -----------------------------
   *               2
   * </pre>
   *
   * This can then be added to the log of the x dependent part, which, when
   * exponentiated, gives the correct result of dividing by this term.
   */
  private double logPDFConst;
  /**
   * When we compute the constant {@link #logPDFConst}, we only need the inverse
   * of the covariance matrix.
   */
  private Matrix invCovariance;
  private Vec mean;
  /**
   * Lower triangular cholesky decomposition used for sampling such that L * L
   * <sup>T</sup> = Covariance Matrix
   */
  private Matrix L;

  public NormalM() {
  }

  public NormalM(final Vec mean, final Matrix covariance) {
    setMeanCovariance(mean, covariance);
  }

  @Override
  public NormalM clone() {
    final NormalM clone = new NormalM();
    if (invCovariance != null) {
      clone.invCovariance = invCovariance.clone();
    }
    if (mean != null) {
      clone.mean = mean.clone();
    }
    clone.logPDFConst = logPDFConst;
    return clone;
  }

  @Override
  public double logPdf(final Vec x) {
    if (mean == null) {
      throw new ArithmeticException("No mean or variance set");
    }
    final Vec xMinusMean = x.subtract(mean);
    // Compute the part that is depdentent on x
    final double xDependent = xMinusMean.dot(invCovariance.multiply(xMinusMean)) * -0.5;
    return logPDFConst + xDependent;
  }

  @Override
  public double pdf(final Vec x) {
    final double pdf = exp(logPdf(x));
    if (Double.isInfinite(pdf) || Double.isNaN(pdf)) {// Ugly numerical error
                                                      // has occured
      return 0;
    }
    return pdf;
  }

  @Override
  public List<Vec> sample(final int count, final Random rand) {
    final List<Vec> samples = new ArrayList<Vec>(count);
    final Vec Z = new DenseVector(L.rows());

    for (int i = 0; i < count; i++) {
      for (int j = 0; j < Z.length(); j++) {
        Z.set(j, rand.nextGaussian());
      }
      final Vec sample = L.multiply(Z);
      sample.mutableAdd(mean);
      samples.add(sample);
    }

    return samples;
  }

  /**
   * Sets the covariance matrix for this matrix.
   *
   * @param covMatrix
   *          set the covariance matrix used for this distribution
   * @throws ArithmeticException
   *           if the covariance matrix is not square, does not agree with the
   *           mean, or is not positive definite. An exception may not be throw
   *           for all bad matrices.
   */
  public void setCovariance(final Matrix covMatrix) {
    if (!covMatrix.isSquare()) {
      throw new ArithmeticException("Covariance matrix must be square");
    } else if (covMatrix.rows() != mean.length()) {
      throw new ArithmeticException("Covariance matrix does not agree with the mean");
    }

    final CholeskyDecomposition cd = new CholeskyDecomposition(covMatrix.clone());
    L = cd.getLT();
    L.mutableTranspose();

    final LUPDecomposition lup = new LUPDecomposition(covMatrix.clone());
    final int k = mean.length();
    final double det = lup.det();
    if (Double.isNaN(det) || det < 1e-10) {
      // Numerical unstable or sub rank matrix. Use the SVD to work with the
      // more stable pesudo matrix
      final SingularValueDecomposition svd = new SingularValueDecomposition(covMatrix.clone());
      // We need the rank deficient PDF and pesude inverse
      logPDFConst = 0.5 * log(svd.getPseudoDet() * pow(2 * PI, svd.getRank()));
      invCovariance = svd.getPseudoInverse();
    } else {
      logPDFConst = (-k * log(2 * PI) - log(det)) * 0.5;
      invCovariance = lup.solve(Matrix.eye(k));
    }
  }

  /**
   * Sets the mean and covariance for this distribution. For an <i>n</i>
   * dimensional distribution, <tt>mean</tt> should be of length <i>n</i> and
   * <tt>covariance</tt> should be an <i>n</i> by <i>n</i> matrix. It is also a
   * requirement that the matrix be symmetric positive definite.
   *
   * @param mean
   *          the mean for the distribution. A copy will be used.
   * @param covariance
   *          the covariance for this distribution. A copy will be used.
   * @throws ArithmeticException
   *           if the <tt>mean</tt> and <tt>covariance</tt> do not agree, or the
   *           covariance is not positive definite. An exception may not be
   *           throw for all bad matrices.
   */
  public void setMeanCovariance(final Vec mean, final Matrix covariance) {
    if (!covariance.isSquare()) {
      throw new ArithmeticException("Covariance matrix must be square");
    } else if (mean.length() != covariance.rows()) {
      throw new ArithmeticException("The mean vector and matrix must have the same dimension," + mean.length()
          + " does not match [" + covariance.rows() + ", " + covariance.rows() + "]");
    }
    // Else, we are good!
    this.mean = mean.clone();
    setCovariance(covariance);
  }

  @Override
  public <V extends Vec> boolean setUsingData(final List<V> dataSet) {
    final Vec origMean = mean;
    try {
      final Vec newMean = MatrixStatistics.meanVector(dataSet);
      final Matrix covariance = MatrixStatistics.covarianceMatrix(mean, dataSet);

      mean = newMean;
      setCovariance(covariance);
      return true;
    } catch (final ArithmeticException ex) {
      mean = origMean;
      return false;
    }
  }

  @Override
  public boolean setUsingDataList(final List<DataPoint> dataSet) {
    final Vec origMean = mean;
    try {
      final Vec newMean = new DenseVector(dataSet.get(0).getNumericalValues().length());
      double sumOfWeights = 0.0, sumOfSquaredWeights = 0.0;
      for (int i = 0; i < dataSet.size(); i++) {
        final DataPoint dp = dataSet.get(i);
        newMean.mutableAdd(dp.getWeight(), dp.getNumericalValues());
        sumOfWeights += dp.getWeight();
        sumOfSquaredWeights += Math.pow(dp.getWeight(), 2);
      }
      newMean.mutableDivide(sumOfWeights);

      // Now compute the covariance matrix
      final Matrix covariance = new DenseMatrix(newMean.length(), newMean.length());
      covarianceMatrix(newMean, dataSet, covariance, sumOfWeights, sumOfSquaredWeights);

      mean = newMean;
      setCovariance(covariance);
      return true;
    } catch (final ArithmeticException ex) {
      mean = origMean;
      return false;
    }
  }
}
