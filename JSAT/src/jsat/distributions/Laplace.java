package jsat.distributions;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.signum;

import jsat.linear.Vec;
import jsat.text.GreekLetters;

/**
 *
 * @author Edward Raff
 */
public final class Laplace extends ContinuousDistribution {

  private static final long serialVersionUID = -4799360517803678236L;
  /**
   * location
   */
  private double mu;
  /*
   * Scale
   */
  private double b;

  public Laplace(final double mu, final double b) {
    setB(b);
    setMu(mu);
  }

  @Override
  public double cdf(final double x) {
    final double xMu = x - mu;
    return 0.5 * (1 + signum(x) * (1 - exp(-abs(xMu) / b)));
  }

  @Override
  public ContinuousDistribution clone() {
    return new Laplace(mu, b);
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
    final Laplace other = (Laplace) obj;
    if (Double.doubleToLongBits(b) != Double.doubleToLongBits(other.b)) {
      return false;
    }
    return Double.doubleToLongBits(mu) == Double.doubleToLongBits(other.mu);
  }

  public double getB() {
    return b;
  }

  @Override
  public double[] getCurrentVariableValues() {
    return new double[] { mu, b };
  }

  @Override
  public String getDistributionName() {
    return "Laplace";
  }

  public double getMu() {
    return mu;
  }

  @Override
  public String[] getVariables() {
    return new String[] { GreekLetters.mu, "b" };
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    long temp;
    temp = Double.doubleToLongBits(b);
    result = prime * result + (int) (temp ^ temp >>> 32);
    temp = Double.doubleToLongBits(mu);
    result = prime * result + (int) (temp ^ temp >>> 32);
    return result;
  }

  @Override
  public double invCdf(final double p) {
    return mu - b * signum(p - 0.5) * log(1 - 2 * abs(p - 0.5));
  }

  @Override
  public double max() {
    return Double.POSITIVE_INFINITY;
  }

  @Override
  public double mean() {
    return mu;
  }

  @Override
  public double median() {
    return mu;
  }

  @Override
  public double min() {
    return Double.NEGATIVE_INFINITY;
  }

  @Override
  public double mode() {
    return mu;
  }

  @Override
  public double pdf(final double x) {
    return 1 / (2 * b) * exp(-abs(x - mu) / b);
  }

  public void setB(final double b) {
    if (b <= 0) {
      throw new ArithmeticException("The scale parameter must be > 0");
    }
    this.b = b;
  }

  public void setMu(final double mu) {
    this.mu = mu;
  }

  @Override
  public void setUsingData(final Vec data) {
    // Donst sent mu yet incase b turns out to be a bad value
    final double tmpMu = data.mean();

    double newB = 0;
    // TODO add APIs so that sparce vector can do this more efficiently
    for (int i = 0; i < data.length(); i++) {
      newB += abs(data.get(i) - tmpMu);
    }
    newB /= data.length();

    setB(newB);
    setMu(tmpMu);

  }

  @Override
  public void setVariable(final String var, final double value) {
    if (var.equals(GreekLetters.mu)) {
      setMu(value);
    } else if (var.equals("b")) {
      setB(value);
    }
  }

  @Override
  public double skewness() {
    return 0;
  }

  @Override
  public double variance() {
    return 2 * b * b;
  }

}
