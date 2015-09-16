package jsat.distributions;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;

/**
 * The ContinuousDistribution represents the contract for a continuous in one dimension.
 *
 * @author Edward Raff
 */
public abstract class ContinuousDistribution extends Distribution {

  private static final long serialVersionUID = -5079392926462355615L;

  /**
   * Computes the log of the Probability Density Function. Note, that then the probability is zero,
   * {@link Double#NEGATIVE_INFINITY} would be the true value. Instead, this method will always return the negative of
   * {@link Double#MAX_VALUE}. This is to avoid propagating bad values through computation.
   *
   * @param x the value to get the log(PDF) of
   * @return the value of log(PDF(x))
   */
  public double logPdf(double x) {
    double pdf = pdf(x);
    if (pdf <= 0) {
      return -Double.MAX_VALUE;
    }
    return Math.log(pdf);
  }

  /**
   * Computes the value of the Probability Density Function (PDF) at the given point
   *
   * @param x the value to get the PDF
   * @return the PDF(x)
   */
  abstract public double pdf(double x);

  /**
   * The descriptive name of a distribution returns the name of the distribution, followed by the parameters of the
   * distribution and their values.
   *
   * @return the name of the distribution that includes parameter values
   */
  public String getDescriptiveName() {
    StringBuilder sb = new StringBuilder(getDistributionName());
    sb.append("(");
    String[] vars = getVariables();
    double[] vals = getCurrentVariableValues();

    sb.append(vars[0]).append(" = ").append(vals[0]);

    for (int i = 1; i < vars.length; i++) {
      sb.append(", ").append(vars[i]).append(" = ").append(vals[i]);
    }

    sb.append(")");

    return sb.toString();
  }

  /**
   * Return the name of the distribution.
   *
   * @return the name of the distribution.
   */
  abstract public String getDistributionName();

  /**
   * Returns an array, where each value contains the name of a parameter in the distribution. The order must always be
   * the same, and match up with the values returned by {@link #getCurrentVariableValues() }
   *
   * @return a string of the variable names this distribution uses
   */
  abstract public String[] getVariables();

  /**
   * Returns an array, where each value contains the value of a parameter in the distribution. The order must always be
   * the same, and match up with the values returned by {@link #getVariables() }
   *
   * @return the current values of the parameters used by this distribution, in the same order as their names are
   * returned by {@link #getVariables() }
   */
  abstract public double[] getCurrentVariableValues();

  /**
   * Sets one of the variables of this distribution by the name.
   *
   * @param var the variable to set
   * @param value the value to set
   */
  abstract public void setVariable(String var, double value);

  @Override
  abstract public ContinuousDistribution clone();

  /**
   * Attempts to set the variables used by this distribution based on population sample data, assuming the sample data
   * is from this type of distribution.
   *
   * @param data the data to use to attempt to fit against
   */
  abstract public void setUsingData(Vec data);

  @Override
  public String toString() {
    return getDistributionName();
  }

  /**
   * Wraps the {@link #pdf(double) } function of the given distribution in a function object for use.
   *
   * @param dist the distribution to wrap the pdf of
   * @return a function for evaluating the pdf of the given distribution
   */
  public static Function getFunctionPDF(final ContinuousDistribution dist) {
    return new Function() {

      /**
       *
       */
      private static final long serialVersionUID = -897452735980141746L;

      @Override
      public double f(double... x) {
        return f(DenseVector.toDenseVec(x));
      }

      @Override
      public double f(Vec x) {
        return dist.pdf(x.get(0));
      }
    };
  }

}
