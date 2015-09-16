package jsat.distributions;

import static java.lang.Math.PI;
import static java.lang.Math.exp;
import static java.lang.Math.expm1;
import static java.lang.Math.log;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static jsat.math.SpecialMath.erf;
import static jsat.math.SpecialMath.invErf;
import jsat.linear.Vec;
import jsat.text.GreekLetters;

/**
 *
 * @author Edward Raff
 */
public class LogNormal extends ContinuousDistribution {

  private static final long serialVersionUID = -6938582328705527274L;
  double mu;
  double sig;

  public LogNormal() {
    this(0, 1);
  }

  public LogNormal(final double mu, final double sig) {
    this.mu = mu;
    this.sig = sig;
  }

  @Override
  public double cdf(final double x) {
    if (x <= 0) {
      return 0;
    }
    return 0.5 + 0.5 * erf((log(x) - mu) / sqrt(2 * sig * sig));
  }

  @Override
  public ContinuousDistribution clone() {
    return new LogNormal(mu, sig);
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
    final LogNormal other = (LogNormal) obj;
    if (Double.doubleToLongBits(mu) != Double.doubleToLongBits(other.mu)) {
      return false;
    }
    return Double.doubleToLongBits(sig) == Double.doubleToLongBits(other.sig);
  }

  @Override
  public double[] getCurrentVariableValues() {
    return new double[] { mu, sig };
  }

  @Override
  public String getDistributionName() {
    return "LogNormal";
  }

  @Override
  public String[] getVariables() {
    return new String[] { GreekLetters.mu, GreekLetters.sigma };
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    long temp;
    temp = Double.doubleToLongBits(mu);
    result = prime * result + (int) (temp ^ temp >>> 32);
    temp = Double.doubleToLongBits(sig);
    result = prime * result + (int) (temp ^ temp >>> 32);
    return result;
  }

  @Override
  public double invCdf(final double p) {
    final double expo = mu + sqrt(2) * sqrt(sig * sig) * invErf(2 * p - 1.0);
    return exp(expo);
  }

  @Override
  public double max() {
    return Double.POSITIVE_INFINITY;
  }

  @Override
  public double mean() {
    return exp(mu + sig * sig * 0.5);
  }

  @Override
  public double median() {
    return exp(mu);
  }

  @Override
  public double min() {
    return 0;
  }

  @Override
  public double mode() {
    return exp(mu - sig * sig);
  }

  @Override
  public double pdf(final double x) {
    if (x <= 0) {
      return 0;
    }
    final double num = exp(-pow(log(x) - mu, 2) / (2 * sig * sig));
    final double denom = x * sqrt(2 * PI * sig * sig);
    return num / denom;
  }

  @Override
  public void setUsingData(final Vec data) {
    final double mean = data.mean();
    final double var = data.variance();

    mu = log(mean) - 0.5 * log(1 + var / (mean * mean));
    sig = sqrt(1 + var / (mean * mean));
  }

  @Override
  public void setVariable(final String var, final double value) {
    if (var.equals(GreekLetters.mu)) {
      mu = value;
    } else if (var.equals(GreekLetters.sigma)) {
      if (value <= 0) {
        throw new ArithmeticException("Standard deviation must be > 0, not " + value);
      } else {
        sig = value;
      }
    }
  }

  @Override
  public double skewness() {
    return (exp(sig * sig) + 2) * sqrt(expm1(sig * sig));
  }

  @Override
  public double variance() {
    return expm1(sig * sig) * exp(2 * mu + sig * sig);
  }

}
