package jsat.distributions;

import static java.lang.Math.PI;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import jsat.linear.Vec;
import jsat.math.SpecialMath;

/**
 * Implementation of the
 * <a href="http://en.wikipedia.org/wiki/L%C3%A9vy_distribution">Levy
 * ContinuousDistribution </a>
 *
 * @author Edward Raff
 */
public class Levy extends ContinuousDistribution {

  private static final long serialVersionUID = 3132169946527422816L;
  private double location;
  private double scale;
  private double logScale;

  public Levy(final double scale, final double location) {
    setScale(scale);
    setLocation(location);
  }

  @Override
  public double cdf(final double x) {
    if (x < location) {
      return 0;
    }
    return SpecialMath.erfc(sqrt(scale / (2 * (x - location))));
  }

  @Override
  public Levy clone() {
    return new Levy(scale, location);
  }

  @Override
  public boolean equals(final Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    final Levy other = (Levy) obj;
    if (Double.doubleToLongBits(location) != Double.doubleToLongBits(other.location)) {
      return false;
    }
    return Double.doubleToLongBits(scale) == Double.doubleToLongBits(other.scale);
  }

  @Override
  public double[] getCurrentVariableValues() {
    return new double[] { scale, location };
  }

  @Override
  public String getDistributionName() {
    return "Levy";
  }

  /**
   * Returns the location parameter used by this distribution.
   *
   * @return distribution
   */
  public double getLocation() {
    return location;
  }

  /**
   * Returns the scale parameter used by this distribution
   *
   * @return the scale parameter
   */
  public double getScale() {
    return scale;
  }

  @Override
  public String[] getVariables() {
    return new String[] { "Scale", "Location" };
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    long temp;
    temp = Double.doubleToLongBits(location);
    result = prime * result + (int) (temp ^ temp >>> 32);
    temp = Double.doubleToLongBits(scale);
    result = prime * result + (int) (temp ^ temp >>> 32);
    return result;
  }

  @Override
  public double invCdf(final double p) {
    if (p < 0 || p > 1) {
      throw new ArithmeticException("Invalid probability " + p);
    }
    return scale / (2 * pow(SpecialMath.invErfc(p), 2)) + location;
  }

  @Override
  public double logPdf(final double x) {
    if (x < location) {
      return Double.NEGATIVE_INFINITY;
    }
    final double mu = x - location;
    return -(-mu * logScale + scale + 3 * mu * log(mu) + mu * log(PI) + mu * log(2)) / (2 * mu);
  }

  @Override
  public double max() {
    return Double.POSITIVE_INFINITY;
  }

  @Override
  public double mean() {
    return Double.POSITIVE_INFINITY;
  }

  @Override
  public double min() {
    return location;
  }

  @Override
  public double mode() {
    return scale / 3 + location;
  }

  @Override
  public double pdf(final double x) {
    if (x < location) {
      return 0;
    }
    return exp(logPdf(x));
  }

  /**
   * Sets location of the Levy distribution.
   *
   * @param location
   *          the new location
   */
  public void setLocation(final double location) {
    if (Double.isNaN(location) || Double.isInfinite(location)) {
      throw new ArithmeticException("location must be a real number");
    }
    this.location = location;
  }

  /**
   * Sets the scale of the Levy distribution
   *
   * @param scale
   *          the new scale value, must be positive
   */
  public void setScale(final double scale) {
    if (scale <= 0 || Double.isNaN(scale) || Double.isInfinite(scale)) {
      throw new ArithmeticException("Scale must be a positive value, not " + scale);
    }
    this.scale = scale;
    logScale = log(scale);
  }

  @Override
  public void setUsingData(final Vec data) {
    setLocation(data.min());

    setScale(2 * pow(SpecialMath.invErfc(0.5), 2) * (data.median() - location));

  }

  @Override
  public void setVariable(final String var, final double value) {
    if (var.equals(getVariables()[0])) {
      setScale(value);
    } else if (var.equals(getVariables()[1])) {
      setLocation(value);
    }
  }

  @Override
  public double skewness() {
    return Double.NaN;
  }

  @Override
  public double standardDeviation() {
    return Double.POSITIVE_INFINITY;
  }

  @Override
  public double variance() {
    return Double.POSITIVE_INFINITY;
  }

}
