package jsat.distributions;

import static java.lang.Math.PI;
import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.rootfinding.RiddersMethod;

/**
 *
 * @author Edward Raff
 */
public class Kolmogorov extends ContinuousDistribution {

  private static final long serialVersionUID = 7319511918364286930L;

  private final Function fCDF = new Function() {

    private static final long serialVersionUID = -716553015542184045L;

    @Override
    public double f(final double... x) {
      return cdf(x[0]) - x[1];
    }

    @Override
    public double f(final Vec x) {
      return f(x.get(0), x.get(1));
    }
  };

  public Kolmogorov() {
  }

  @Override
  public double cdf(final double x) {
    if (x < 0) {
      throw new ArithmeticException("Invalid value of x, x must be > 0, not " + x);
    } else if (x == 0) {
      return 0;
    } else if (x >= 5) {// By this point, floating point isnt accurate enough to
                        // distinguish between 1.0 and the true value.
      return 1;
    }

    /*
     * Uses 2 formulas, see
     * http://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#
     * Kolmogorov_distribution
     * 
     * Each formula converges very rapidly, interface 3 terms to full IEEE
     * precision for one or the other - crossover point is 1.18 according to
     * Numerical Recipies, 3rd Edition p(334-335)
     */
    double tmp = 0;
    final double x2 = x * x;
    if (x < 1.18) {

      for (int j = 1; j <= 3; j++) {
        tmp += exp(-pow(2 * j - 1, 2) * PI * PI / (8 * x2));
      }

      return sqrt(2 * PI) / x * tmp;
    } else {
      // for(int j = 1; j <= 3; j++ )
      // tmp += exp(-2*j*j*x*x)*pow(-1,j-1);

      tmp = exp(-2 * x2) + exp(-18 * x2) - exp(-8 * x2);// In order of 1st, 3rd,
                                                        // and 2nd to reduce
                                                        // chances of
                                                        // cancelation

      return 1 - 2 * tmp;
    }

  }

  @Override
  public ContinuousDistribution clone() {
    return new Kolmogorov();
  }

  @Override
  public boolean equals(final Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    return getClass() == obj.getClass();
  }

  @Override
  public double[] getCurrentVariableValues() {
    return new double[] {};
  }

  @Override
  public String getDistributionName() {
    return "Kolmogorov";
  }

  @Override
  public String[] getVariables() {
    return new String[] {};
  }

  @Override
  public int hashCode() {
    return 31;
  }

  @Override
  public double invCdf(final double p) {
    return RiddersMethod.root(0, 5, fCDF, p, p);
  }

  @Override
  public double max() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public double mean() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public double median() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public double min() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public double mode() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public double pdf(final double x) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public void setUsingData(final Vec data) {

  }

  @Override
  public void setVariable(final String var, final double value) {

  }

  @Override
  public double skewness() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public double variance() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

}
