package jsat.distributions.kernels;

import java.util.List;

import jsat.DataSet;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.linear.Vec;
import jsat.parameters.Parameterized;

/**
 * The PUK kernel is an alternative to the RBF Kernel. By altering the
 * {@link #setOmega(double) omega} parameter the behavior of the PUK kernel can
 * be controlled. The {@link #setSigma(double) sigma} parameter works in the
 * same way as the RBF Kernel.<br>
 * <br>
 * See: Üstün, B., Melssen, W. J.,&amp;Buydens, L. M. C. (2006). <i>Facilitating
 * the application of Support Vector Regression by using a universal Pearson VII
 * function based kernel</i>. Chemometrics and Intelligent Laboratory Systems,
 * 81(1), 29–40. doi:10.1016/j.chemolab.2005.09.003
 *
 * @author Edward Raff
 */
public class PukKernel extends BaseL2Kernel implements Parameterized {

  private static final long serialVersionUID = 8727097671803148320L;

  /**
   * Guesses the distribution to use for the &omega; parameter
   *
   * @param d
   *          the dataset to get the guess for
   * @return the guess for the &omega; parameter
   * @see #setOmega(double)
   */
  public static Distribution guessOmega(final DataSet d) {
    return new LogUniform(0.25, 50);
  }

  /**
   * Guesses the distribution to use for the &lambda; parameter
   *
   * @param d
   *          the dataset to get the guess for
   * @return the guess for the &lambda; parameter
   * @see #setSigma(double)
   */
  public static Distribution guessSigma(final DataSet d) {
    return RBFKernel.guessSigma(d);
  }

  private double sigma;

  private double omega;

  private double cnst;

  /**
   * Creates a new PUK Kernel
   *
   * @param sigma
   *          the width parameter of the kernel
   * @param omega
   *          the shape parameter of the kernel
   */
  public PukKernel(final double sigma, final double omega) {
    setSigma(sigma);
    setOmega(omega);
  }

  @Override
  public PukKernel clone() {
    return new PukKernel(sigma, omega);
  }

  @Override
  public double eval(final int a, final int b, final List<? extends Vec> trainingSet, final List<Double> cache) {
    return getVal(Math.sqrt(getSqrdNorm(b, b, trainingSet, cache)));
  }

  @Override
  public double eval(final int a, final Vec b, final List<Double> qi, final List<? extends Vec> vecs,
      final List<Double> cache) {
    return getVal(Math.sqrt(getSqrdNorm(a, b, qi, vecs, cache)));
  }

  @Override
  public double eval(final Vec a, final Vec b) {
    return getVal(a.pNormDist(2.0, b));
  }

  public double getOmega() {
    return omega;
  }

  public double getSigma() {
    return sigma;
  }

  private double getVal(final double pNormDist) {
    final double tmp = 2 * pNormDist * cnst / sigma;
    return 1 / Math.pow(1 + tmp * tmp, omega);
  }

  /**
   * Sets the omega parameter value, which controls the shape of the kernel
   *
   * @param omega
   *          the positive parameter value
   */
  public void setOmega(final double omega) {
    if (omega <= 0 || Double.isNaN(omega) || Double.isInfinite(omega)) {
      throw new ArithmeticException("omega must be positive, not " + omega);
    }
    this.omega = omega;
    cnst = Math.sqrt(Math.pow(2, 1 / omega) - 1);
  }

  /**
   * Sets the sigma parameter value, which controls the width of the kernel
   *
   * @param sigma
   *          the positive parameter value
   */
  public void setSigma(final double sigma) {
    if (sigma <= 0 || Double.isNaN(sigma) || Double.isInfinite(sigma)) {
      throw new ArithmeticException("sigma must be positive, not " + sigma);
    }
    this.sigma = sigma;
  }

}
