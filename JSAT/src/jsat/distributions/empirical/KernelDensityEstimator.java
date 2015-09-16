package jsat.distributions.empirical;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import jsat.distributions.ContinuousDistribution;
import jsat.distributions.empirical.kernelfunc.EpanechnikovKF;
import jsat.distributions.empirical.kernelfunc.GaussKF;
import jsat.distributions.empirical.kernelfunc.KernelFunction;
import jsat.distributions.empirical.kernelfunc.UniformKF;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.OnLineStatistics;
import jsat.utils.ProbailityMatch;

/**
 * Kernel Density Estimator, KDE, uses the data set itself to approximate the
 * underlying probability distribution using {@link KernelFunction Kernel
 * Functions}.
 *
 * @author Edward Raff
 */
public class KernelDensityEstimator extends ContinuousDistribution {
  /*
   * README Implementation note: The values are stored in sorted order, which
   * allows for fast evaluations. Instead of doing the full loop on each
   * function call, O(n) time, we know the bounds on the values that will effect
   * results, so we can do 2 binary searches and then a loop. Though this is
   * still technically, O(n), its more accurately described as O(n * epsilon *
   * log(n)) , where n * epsilon << n
   */

  private static final long serialVersionUID = 7708020456632603947L;

  /**
   * Automatically selects a good Kernel function for the data set that balances
   * Execution time and accuracy
   *
   * @param dataPoints
   * @return a kernel that will work well for the given distribution
   */
  public static KernelFunction autoKernel(final Vec dataPoints) {
    if (dataPoints.length() < 30) {
      return GaussKF.getInstance();
    } else if (dataPoints.length() < 1000) {
      return EpanechnikovKF.getInstance();
    } else {
      // For very large data sets, Uniform is FAST and just as accurate
      return UniformKF.getInstance();
    }
  }

  public static double BandwithGuassEstimate(final Vec X) {
    if (X.length() == 1) {
      return 1;
    } else if (X.standardDeviation() == 0) {
      return 1.06 * Math.pow(X.length(), -1.0 / 5.0);
    }
    return 1.06 * X.standardDeviation() * Math.pow(X.length(), -1.0 / 5.0);
  }

  /**
   * The various values
   */
  private double[] X;
  /**
   * Weights corresponding to each value. If all the same, weights should have a
   * length of 0
   */
  private double[] weights;
  /**
   * For unweighted data, this is equal to X.length
   */
  private double sumOFWeights;

  /**
   * The bandwidth
   */
  private double h;

  private double Xmean, Xvar, Xskew;

  private final KernelFunction k;

  @SuppressWarnings("unused")
  private final Function cdfFunc = new Function() {

    /**
     *
     */
    private static final long serialVersionUID = -4100975560125048798L;

    @Override
    public double f(final double... x) {
      return cdf(x[0]);
    }

    @Override
    public double f(final Vec x) {
      return f(x.get(0));
    }
  };

  /**
   * Copy constructor
   */
  private KernelDensityEstimator(final double[] X, final double h, final double Xmean, final double Xvar,
      final double Xskew, final KernelFunction k, final double sumOfWeights, final double[] weights) {
    this.X = Arrays.copyOf(X, X.length);
    this.h = h;
    this.Xmean = Xmean;
    this.Xvar = Xvar;
    this.Xskew = Xskew;
    this.k = k;
    sumOFWeights = sumOfWeights;
    this.weights = Arrays.copyOf(weights, weights.length);
  }

  public KernelDensityEstimator(final Vec dataPoints) {
    this(dataPoints, autoKernel(dataPoints));
  }

  public KernelDensityEstimator(final Vec dataPoints, final KernelFunction k) {
    this(dataPoints, k, BandwithGuassEstimate(dataPoints));
  }

  public KernelDensityEstimator(final Vec dataPoints, final KernelFunction k, final double h) {
    setUpX(dataPoints);
    this.k = k;
    this.h = h;
  }

  public KernelDensityEstimator(final Vec dataPoints, final KernelFunction k, final double h, final double[] weights) {
    setUpX(dataPoints, weights);
    this.k = k;
    this.h = h;
  }

  public KernelDensityEstimator(final Vec dataPoints, final KernelFunction k, final double[] weights) {
    this(dataPoints, k, BandwithGuassEstimate(dataPoints), weights);
  }

  @Override
  public double cdf(final double x) {
    // Only values within a certain range will have an effect on the result, so
    // we will skip to that range!
    int from = Arrays.binarySearch(X, x - h * k.cutOff());
    int to = Arrays.binarySearch(X, x + h * k.cutOff());
    // Mostly likely the exact value of x is not in the list, so it returns the
    // inseration points
    from = from < 0 ? -from - 1 : from;
    to = to < 0 ? -to - 1 : to;

    double sum = 0;

    for (int i = Math.max(0, from); i < Math.min(X.length, to + 1); i++) {
      sum += k.intK((x - X[i]) / h) * getWeight(i);
    }

    /*
     * Slightly different, all things below the from value for the cdf would be
     * adding 1 to the value, as the value of x would be the integration over
     * the entire range, which by definition, is equal to 1.
     */
    // We perform the addition after the summation to reduce the difference size
    if (weights.length == 0) {// No weights
      sum += Math.max(0, from);
    } else {
      sum += weights[from];
    }

    return sum / X.length;
  }

  @Override
  public KernelDensityEstimator clone() {
    return new KernelDensityEstimator(X, h, Xmean, Xvar, Xskew, k, sumOFWeights, weights);
  }

  @Override
  public boolean equals(final Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof KernelDensityEstimator)) {
      return false;
    }
    final KernelDensityEstimator other = (KernelDensityEstimator) obj;
    if (Double.doubleToLongBits(Xmean) != Double.doubleToLongBits(other.Xmean)) {
      return false;
    }
    if (Double.doubleToLongBits(Xskew) != Double.doubleToLongBits(other.Xskew)) {
      return false;
    }
    if (Double.doubleToLongBits(Xvar) != Double.doubleToLongBits(other.Xvar)) {
      return false;
    }

    if (Double.doubleToLongBits(h) != Double.doubleToLongBits(other.h)) {
      return false;
    }
    if (Double.doubleToLongBits(sumOFWeights) != Double.doubleToLongBits(other.sumOFWeights)) {
      return false;
    }
    if (k == null) {
      if (other.k != null) {
        return false;
      }
    } else if (k.getClass() != other.k.getClass()) {
      return false;
    }
    if (!Arrays.equals(X, other.X)) {
      return false;
    }
    return Arrays.equals(weights, other.weights);
  }

  /**
   *
   * @return the bandwidth parameter
   */
  public double getBandwith() {
    return h;
  }

  @Override
  public double[] getCurrentVariableValues() {
    return new double[] { h };
  }

  @Override
  public String getDistributionName() {
    return "Kernel Density Estimate";
  }

  @Override
  public String[] getVariables() {
    return new String[] { "h" };
  }

  private double getWeight(final int i) {
    if (weights.length == 0) {
      return 1.0;
    } else if (i == 0) {
      return weights[i];
    } else {
      return weights[i] - weights[i - 1];
    }
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + Arrays.hashCode(X);
    long temp;
    temp = Double.doubleToLongBits(Xmean);
    result = prime * result + (int) (temp ^ temp >>> 32);
    temp = Double.doubleToLongBits(Xskew);
    result = prime * result + (int) (temp ^ temp >>> 32);
    temp = Double.doubleToLongBits(Xvar);
    result = prime * result + (int) (temp ^ temp >>> 32);
    temp = Double.doubleToLongBits(h);
    result = prime * result + (int) (temp ^ temp >>> 32);
    result = prime * result + (k == null ? 0 : k.hashCode());
    temp = Double.doubleToLongBits(sumOFWeights);
    result = prime * result + (int) (temp ^ temp >>> 32);
    result = prime * result + Arrays.hashCode(weights);
    return result;
  }

  @Override
  public double invCdf(final double p) {
    int index;
    double kd0;

    if (weights.length == 0) {
      final double r = p * X.length;
      index = (int) r;
      final double pd0 = r - index, pd1 = 1 - pd0;
      kd0 = k.intK(pd1);
    } else// CDF can be found from the weights summings
    {
      final double XEstimate = p * sumOFWeights;
      index = Arrays.binarySearch(weights, XEstimate);
      index = index < 0 ? -index - 1 : index;
      if (X[index] != 0) {// TODO fix this bit
        kd0 = 1.0;// -Math.abs((XEstimate-X[index])/X[index]);
      } else {
        kd0 = 1.0;
      }
    }

    if (index == X.length - 1) {// at the tail end
      return X[index] * kd0;
    }
    final double x = X[index] * kd0 + X[index + 1] * (1 - kd0);

    return x;
  }

  @Override
  public double max() {
    return X[X.length - 1] + h;
  }

  @Override
  public double mean() {
    return Xmean;
  }

  @Override
  public double min() {
    return X[0] - h;
  }

  @Override
  public double mode() {
    double maxP = 0, pTmp;
    double maxV = Double.NaN;
    for (final double element : X) {
      if ((pTmp = pdf(element)) > maxP) {
        maxP = pTmp;
        maxV = element;
      }
    }

    return maxV;
  }

  @Override
  public double pdf(final double x) {
    return pdf(x, -1);
  }

  /**
   * Computes the Leave One Out PDF of the estimator
   *
   * @param x
   *          the value to get the pdf of
   * @param j
   *          the sorted index of the value to leave. If a negative value is
   *          given, the PDF with all values is returned
   * @return the pdf with the given index left out
   */
  private double pdf(final double x, final int j) {
    /*
     * n ===== /x - x \ 1 \ | i| f(x) = --- > K|------| n h / \ h / ===== i = 1
     *
     */

    // Only values within a certain range will have an effect on the result, so
    // we will skip to that range!
    int from = Arrays.binarySearch(X, x - h * k.cutOff());
    int to = Arrays.binarySearch(X, x + h * k.cutOff());
    // Mostly likely the exact value of x is not in the list, so it returns the
    // inseration points
    from = from < 0 ? -from - 1 : from;
    to = to < 0 ? -to - 1 : to;

    // Univariate opt, if uniform weights, the sum is just the number of
    // elements divide by half
    if (weights.length == 0 && k instanceof UniformKF) {
      return (to - from) * 0.5 / (sumOFWeights * h);
    }

    double sum = 0;
    for (int i = Math.max(0, from); i < Math.min(X.length, to + 1); i++) {
      if (i != j) {
        sum += k.k((x - X[i]) / h) * getWeight(i);
      }
    }

    return sum / (sumOFWeights * h);
  }

  /**
   * Sets the bandwidth used for smoothing. Higher values make the pdf smoother,
   * but can obscure features. Too small a bandwidth will causes spikes at only
   * the data points.
   *
   * @param val
   *          new bandwidth
   */
  public void setBandwith(final double val) {
    if (val <= 0 || Double.isInfinite(val)) {
      throw new ArithmeticException("Bandwith parameter h must be greater than zero, not " + 0);
    }
    h = val;
  }

  private void setUpX(final Vec S) {
    Xmean = S.mean();
    Xvar = S.variance();
    Xskew = S.skewness();
    X = S.arrayCopy();
    Arrays.sort(X);
    sumOFWeights = X.length;
    weights = new double[0];
  }

  private void setUpX(final Vec S, final double[] weights) {
    if (S.length() != weights.length) {
      throw new RuntimeException("Weights and variables do not have the same length");
    }

    final OnLineStatistics stats = new OnLineStatistics();

    X = new double[S.length()];
    this.weights = Arrays.copyOf(weights, S.length());

    // Probability is the X value, match is the weights - so that they can be
    // sorted together.
    final List<ProbailityMatch<Double>> sorter = new ArrayList<ProbailityMatch<Double>>(S.length());
    for (int i = 0; i < S.length(); i++) {
      sorter.add(new ProbailityMatch<Double>(S.get(i), weights[i]));
    }
    Collections.sort(sorter);
    for (int i = 0; i < sorter.size(); i++) {
      X[i] = sorter.get(i).getProbability();
      this.weights[i] = sorter.get(i).getMatch();
      stats.add(X[i], this.weights[i]);
    }
    // Now do some helpful preprocessing on weights. We will make index i store
    // the sum for [0, i].
    // Each individual weight can still be retrieved in O(1) by accessing a 2nd
    // index and a subtraction
    // Methods that need the sum can now access it in O(1) time from the weights
    // array instead of doing an O(n) summations
    for (int i = 1; i < this.weights.length; i++) {
      this.weights[i] += this.weights[i - 1];
    }
    sumOFWeights = this.weights[this.weights.length - 1];
    Xmean = stats.getMean();
    Xvar = stats.getVarance();
    Xskew = stats.getSkewness();
  }

  @Override
  public void setUsingData(final Vec data) {
    setUpX(data);
    h = BandwithGuassEstimate(data);
  }

  @Override
  public void setVariable(final String var, final double value) {
    if (var.equals("h")) {
      setBandwith(value);
    }
  }

  @Override
  public double skewness() {
    // TODO cant find anything about what this should really be...
    return Xskew;
  }

  @Override
  public double variance() {
    return Xvar + h * h * k.k2();
  }

}
