package jsat.distributions.kernels;

import java.util.List;

import jsat.DataSet;
import jsat.distributions.Distribution;
import jsat.linear.Vec;

/**
 * Provides an implementation of the Rational Quadratic Kernel, which is of the
 * form: <br>
 * k(x, y) = 1 - ||x-y||<sup>2</sup> / (||x-y||<sup>2</sup> + c)
 *
 * @author Edward Raff
 */
public class RationalQuadraticKernel extends BaseL2Kernel {

  private static final long serialVersionUID = 6773399185851115840L;

  /**
   * Guess the distribution to use for the C parameter.
   *
   * @param d
   *          the data set to get the guess for
   * @return the guess for the C parameter
   * @see #setC(double)
   */
  public static Distribution guessC(final DataSet d) {
    // TODO come up with a better estiamte
    return RBFKernel.guessSigma(d);// suprisingly this seens to work well
  }

  private double c;

  /**
   * Creates a new RQ Kernel
   *
   * @param c
   *          the positive additive coefficient
   */
  public RationalQuadraticKernel(final double c) {
    this.c = c;
  }

  @Override
  public RationalQuadraticKernel clone() {
    return new RationalQuadraticKernel(c);
  }

  @Override
  public double eval(final int a, final int b, final List<? extends Vec> trainingSet, final List<Double> cache) {
    final double dist = getSqrdNorm(a, b, trainingSet, cache);
    return 1 - dist / (dist + c);
  }

  @Override
  public double eval(final int a, final Vec b, final List<Double> qi, final List<? extends Vec> vecs,
      final List<Double> cache) {
    final double dist = getSqrdNorm(a, b, qi, vecs, cache);
    return 1 - dist / (dist + c);
  }

  @Override
  public double eval(final Vec a, final Vec b) {
    final double dist = Math.pow(a.pNormDist(2, b), 2);
    return 1 - dist / (dist + c);
  }

  /**
   * Returns the positive additive coefficient
   *
   * @return the positive additive coefficient
   */
  public double getC() {
    return c;
  }

  /**
   * Sets the positive additive coefficient
   *
   * @param c
   *          the positive additive coefficient
   */
  public void setC(final double c) {
    if (c <= 0 || Double.isNaN(c) || Double.isInfinite(c)) {
      throw new IllegalArgumentException("coefficient must be in (0, Inf), not " + c);
    }
    this.c = c;
  }
}
