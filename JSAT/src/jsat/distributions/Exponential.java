package jsat.distributions;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.pow;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class Exponential extends ContinuousDistribution {

  private static final long serialVersionUID = 1675502925386052588L;
  private double lambda;

  public Exponential() {
    this(1);
  }

  public Exponential(final double lambda) {
    if (lambda <= 0) {
      throw new RuntimeException("The rate parameter must be greater than zero, not " + lambda);
    }
    this.lambda = lambda;
  }

  @Override
  public double cdf(final double d) {
    if (d < 0) {
      return 0;
    }
    return 1 - exp(-lambda * d);
  }

  @Override
  public ContinuousDistribution clone() {
    return new Exponential(lambda);
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
    final Exponential other = (Exponential) obj;
    return Double.doubleToLongBits(lambda) == Double.doubleToLongBits(other.lambda);
  }

  @Override
  public double[] getCurrentVariableValues() {
    return new double[] { lambda };
  }

  @Override
  public String getDescriptiveName() {
    return "Exponential(\u03BB=" + lambda + ")";
  }

  @Override
  public String getDistributionName() {
    return "Exponential";
  }

  @Override
  public String[] getVariables() {
    return new String[] { "\u03BB" };
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    long temp;
    temp = Double.doubleToLongBits(lambda);
    result = prime * result + (int) (temp ^ temp >>> 32);
    return result;
  }

  @Override
  public double invCdf(final double d) {
    if (d < 0 || d > 1) {
      throw new ArithmeticException("Inverse CDF only exists on the range [0,1]");
    }
    return -log(1 - d) / lambda;
  }

  @Override
  public double logPdf(final double x) {
    if (x < 0) {
      return 0;
    }
    return log(lambda) + -lambda * x;
  }

  @Override
  public double max() {
    return Double.POSITIVE_INFINITY;
  }

  @Override
  public double mean() {
    return 1 / lambda;
  }

  @Override
  public double median() {
    return 1 / lambda * log(2);
  }

  @Override
  public double min() {
    return 0;
  }

  @Override
  public double mode() {
    return 0;
  }

  @Override
  public double pdf(final double d) {
    if (d < 0) {
      return 0;
    }
    return lambda * exp(-lambda * d);
  }

  @Override
  public void setUsingData(final Vec data) {
    /**
     * mean of an exponential distribution is lambda^-1
     */
    lambda = 1 / data.mean();
    if (lambda <= 0) {
      lambda = 1;
    }
  }

  @Override
  public void setVariable(final String var, final double value) {
    if (var.equals("\u03BB")) {
      if (value <= 0) {
        throw new RuntimeException("The rate parameter must be greater than zero");
      }
      lambda = value;
    }
  }

  @Override
  public double skewness() {
    return 2;
  }

  @Override
  public double variance() {
    return pow(lambda, -2);
  }

}
