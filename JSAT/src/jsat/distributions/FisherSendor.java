package jsat.distributions;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static jsat.math.SpecialMath.betaIncReg;
import static jsat.math.SpecialMath.invBetaIncReg;
import static jsat.math.SpecialMath.lnBeta;
import jsat.linear.Vec;

/**
 *
 * Also known as the F distribution.
 *
 * @author Edward Raff
 */
public class FisherSendor extends ContinuousDistribution {

  private static final long serialVersionUID = 7628304882101574242L;
  double v1;
  double v2;

  public FisherSendor(final double v1, final double v2) {
    if (v1 <= 0) {
      throw new ArithmeticException("v1 must be > 0 not " + v1);
    }
    if (v2 <= 0) {
      throw new ArithmeticException("v2 must be > 0 not " + v2);
    }
    this.v1 = v1;
    this.v2 = v2;
  }

  @Override
  public double cdf(final double x) {
    if (x <= 0) {
      return 0;
    }
    return betaIncReg(v1 * x / (v1 * x + v2), v1 / 2, v2 / 2);
  }

  @Override
  public ContinuousDistribution clone() {
    return new FisherSendor(v1, v2);
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
    final FisherSendor other = (FisherSendor) obj;
    if (Double.doubleToLongBits(v1) != Double.doubleToLongBits(other.v1)) {
      return false;
    }
    return Double.doubleToLongBits(v2) == Double.doubleToLongBits(other.v2);
  }

  @Override
  public double[] getCurrentVariableValues() {
    return new double[] { v1, v2 };
  }

  @Override
  public String getDistributionName() {
    return "F";
  }

  @Override
  public String[] getVariables() {
    return new String[] { "v1", "v2" };
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    long temp;
    temp = Double.doubleToLongBits(v1);
    result = prime * result + (int) (temp ^ temp >>> 32);
    temp = Double.doubleToLongBits(v2);
    result = prime * result + (int) (temp ^ temp >>> 32);
    return result;
  }

  @Override
  public double invCdf(final double p) {
    if (p < 0 || p > 1) {
      throw new ArithmeticException("Probability must be in the range [0,1], not" + p);
    }
    final double u = invBetaIncReg(p, v1 / 2, v2 / 2);
    return v2 * u / (v1 * (1 - u));
  }

  @Override
  public double logPdf(final double x) {
    if (x <= 0) {
      return 0;
    }
    final double leftSide = v1 / 2 * log(v1) + v2 / 2 * log(v2) - lnBeta(v1 / 2, v2 / 2);
    final double rightSide = (v1 / 2 - 1) * log(x) - (v1 + v2) / 2 * log(v2 + v1 * x);
    return leftSide + rightSide;
  }

  @Override
  public double max() {
    return Double.POSITIVE_INFINITY;
  }

  @Override
  public double mean() {
    if (v2 <= 2) {
      return Double.NaN;
    }

    return v2 / (v2 - 2);
  }

  @Override
  public double median() {
    return v2 / v1 * (1.0 / invBetaIncReg(0.5, v2 / 2, v1 / 2) - 1);
  }

  @Override
  public double min() {
    return 0;
  }

  @Override
  public double mode() {
    if (v1 <= 2) {
      return Double.NaN;
    }

    return (v1 - 2) / v1 * v2 / (v2 + 2);
  }

  @Override
  public double pdf(final double x) {
    if (x <= 0) {
      return 0;
    }
    return exp(logPdf(x));
  }

  @Override
  public void setUsingData(final Vec data) {
    final double mu = data.mean();

    // Only true if v2 > 2
    final double tmp = 2 * mu / (-1 + mu);

    if (tmp < 2) {
      return;// We couldnt approximate anything

    } else {
      v2 = tmp;
      if (v2 < 4) {
        return;// We cant approximate v1
      }
    }

    // only true if v2 > 4
    final double v2sqr = v2 * v2;
    final double var = data.variance();
    final double denom = -2 * v2sqr - 16 * var + 20 * v2 * var - 8 * v2sqr * var + v2sqr * v2 * var;

    v1 = 2 * (-2 * v2sqr + v2sqr * v2) / denom;
  }

  @Override
  public void setVariable(final String var, final double value) {
    if (var.equals("v1")) {
      if (value > 0) {
        v1 = value;
      } else {
        throw new ArithmeticException("v1 must be > 0 not " + value);
      }
    } else if (var.equals("v2")) {
      if (value > 0) {
        v2 = value;
      } else {
        throw new ArithmeticException("v2 must be > 0 not " + value);
      }
    }
  }

  @Override
  public double skewness() {

    if (v2 <= 6) {// Does not have a skewness for d2 <= 6
      return Double.NaN;
    }
    final double num = (2 * v1 + v2 - 2) * sqrt(8 * (v2 - 4));
    final double denom = (v2 - 6) * sqrt(v1 * (v1 + v2 - 2));

    return num / denom;
  }

  @Override
  public double variance() {
    if (v2 <= 4) {
      return Double.NaN;
    }

    return 2 * v2 * v2 * (v1 + v2 - 2) / (v1 * pow(v2 - 2, 2) * (v2 - 4));
  }

}
