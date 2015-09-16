package jsat.math;

import java.io.Serializable;

/**
 *
 * This class provides a means of updating summary statistics as each new data
 * point is added. The data points are not stored, and values are updated with
 * an online algorithm. <br>
 * As such, this class has constant memory usage, regardless of how many values
 * are added. But the results may not be as numerically accurate, and can
 * degrade badly given specific data sequences.
 *
 * @author Edward Raff
 */
public class OnLineStatistics implements Serializable, Cloneable {

  private static final long serialVersionUID = -4286295481362462983L;

  /**
   * Computes a new set of counts that is the sum of the counts from the given
   * distributions. <br>
   * <br>
   * NOTE: Adding two statistics is not as numerically stable. If A and B have
   * values of similar size and scale, the values of the 3rd and 4th moments
   * {@link #getSkewness() } and {@link #getKurtosis() } will suffer from
   * catastrophic cancellations, and may not be as accurate.
   *
   * @param A
   *          the first set of statistics
   * @param B
   *          the second set of statistics
   * @return a new set of statistics that is the addition of the two.
   */
  public static OnLineStatistics add(final OnLineStatistics A, final OnLineStatistics B) {
    final OnLineStatistics toRet = A.clone();
    toRet.add(B);
    return toRet;
  }

  /**
   * Computes a new set of statistics that is the equivalent of having removed
   * all observations in {@code B} from {@code A}. <br>
   * NOTE: removing statistics is not as numerically stable. The values of the
   * 3rd and 4th moments {@link #getSkewness() } and {@link #getKurtosis() }
   * will be inaccurate for many inputs. The {@link #getMin() min} and
   * {@link #getMax() max} can not be determined in this setting, and will not
   * be altered.
   *
   * @param A
   *          the first set of statistics, which must have a larger value for
   *          {@link #getSumOfWeights() } than {@code B}
   * @param B
   *          the set of statistics to remove from {@code A}.
   * @return a new set of statistics that is the removal of {@code B} from
   *         {@code A}
   */
  public static OnLineStatistics remove(final OnLineStatistics A, final OnLineStatistics B) {
    final OnLineStatistics toRet = A.clone();
    toRet.remove(B);
    return toRet;
  }

  /**
   * The current mean
   */
  private double mean;

  /**
   * The current number of samples seen
   */
  private double n;

  // Intermediat value updated at each step, variance computed from it
  private double m2, m3, m4;

  private Double min, max;

  /**
   * Creates a new set of statistical counts with no information
   */
  public OnLineStatistics() {
    this(0, 0, 0, 0, 0);
  }

  /**
   * Creates a new set of statistical counts with these initial values, and can
   * then be updated in an online fashion
   *
   * @param n
   *          the total weight of all data points added. This value must be non
   *          negative
   * @param mean
   *          the starting mean. If <tt>n</tt> is zero, this value will be
   *          ignored.
   * @param variance
   *          the starting variance. If <tt>n</tt> is zero, this value will be
   *          ignored.
   * @param skew
   *          the starting skewness. If <tt>n</tt> is zero, this value will be
   *          ignored.
   * @param kurt
   *          the starting kurtosis. If <tt>n</tt> is zero, this value will be
   *          ignored.
   * @throws ArithmeticException
   *           if <tt>n</tt> is a negative number
   */
  public OnLineStatistics(final double n, final double mean, final double variance, final double skew,
      final double kurt) {
    if (n < 0) {
      throw new ArithmeticException("Can not have a negative set of weights");
    }
    this.n = n;
    if (n != 0) {
      this.mean = mean;
      m2 = variance * (n - 1);
      m3 = Math.pow(m2, 3.0 / 2.0) * skew / Math.sqrt(n);
      m4 = (3 + kurt) * m2 * m2 / n;
    } else {
      this.mean = m2 = m3 = m4 = 0;
    }
    min = max = null;
  }

  private OnLineStatistics(final double n, final double mean, final double m2, final double m3, final double m4,
      final Double min, final Double max) {
    this.n = n;
    this.mean = mean;
    this.m2 = m2;
    this.m3 = m3;
    this.m4 = m4;
    this.min = min;
    this.max = max;
  }

  /**
   * Copy Constructor
   *
   * @param other
   *          the version to make a copy of
   */
  public OnLineStatistics(final OnLineStatistics other) {
    this(other.n, other.mean, other.m2, other.m3, other.m4, other.min, other.max);
  }

  /**
   * Adds a data sample with unit weight to the counts.
   *
   * @param x
   *          the data value to add
   */
  public void add(final double x) {
    add(x, 1.0);
  }

  /**
   * Adds a data sample the the counts with the provided weight of influence.
   *
   * @param x
   *          the data value to add
   * @param weight
   *          the weight to give the value
   * @throws ArithmeticException
   *           if a negative weight is given
   */
  public void add(final double x, final double weight) {
    // See http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    if (weight < 0) {
      throw new ArithmeticException("Can not add a negative weight");
    } else if (weight == 0) {
      return;
    }

    final double n1 = n;
    n += weight;
    final double delta = x - mean;
    final double delta_n = delta * weight / n;
    final double delta_n2 = delta_n * delta_n;
    final double term1 = delta * delta_n * n1;

    mean += delta_n;
    m4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3;
    m3 += term1 * delta_n * (n - 2) - 3 * delta_n * m2;
    m2 += weight * delta * (x - mean);

    if (min == null) {
      min = max = x;
    } else {
      min = Math.min(min, x);
      max = Math.max(max, x);
    }
  }

  /**
   * Adds to the current statistics all the samples that were collected in
   * {@code B}. <br>
   * NOTE: Adding two statistics is not as numerically stable. If A and B have
   * values of similar size and scale, the values of the 3rd and 4th moments
   * {@link #getSkewness() } and {@link #getKurtosis() } will suffer from
   * catastrophic cancellations, and may not be as accurate.
   *
   * @param B
   *          the set of statistics to add to this set
   */
  public void add(final OnLineStatistics B) {
    final OnLineStatistics A = this;
    // XXX double compare.
    if (A.n == B.n && B.n == 0) {
      return;// nothing to do!
    } else if (B.n == 0) {
      return;// still nothing!
    } else if (A.n == 0) {
      n = B.n;
      mean = B.mean;
      m2 = B.m2;
      m3 = B.m3;
      m4 = B.m4;
      min = B.min;
      max = B.max;
      return;
    }

    final double nX = B.n + A.n;
    final double nXsqrd = nX * nX;
    final double nAnB = B.n * A.n;
    final double AnSqrd = A.n * A.n;
    final double BnSqrd = B.n * B.n;

    final double delta = B.mean - A.mean;
    final double deltaSqrd = delta * delta;
    final double deltaCbd = deltaSqrd * delta;
    final double deltaQad = deltaSqrd * deltaSqrd;
    final double newMean = (A.n * A.mean + B.n * B.mean) / (A.n + B.n);
    final double newM2 = A.m2 + B.m2 + deltaSqrd / nX * nAnB;
    final double newM3 = A.m3 + B.m3 + deltaCbd * nAnB * (A.n - B.n) / nXsqrd
        + 3 * delta * (A.n * B.m2 - B.n * A.m2) / nX;
    final double newM4 = A.m4 + B.m4 + deltaQad * (nAnB * (AnSqrd - nAnB + BnSqrd) / (nXsqrd * nX))
        + 6 * deltaSqrd * (AnSqrd * B.m2 + BnSqrd * A.m2) / nXsqrd + 4 * delta * (A.n * B.m3 - B.n * A.m3) / nX;

    n = nX;
    mean = newMean;
    m2 = newM2;
    m3 = newM3;
    m4 = newM4;
    min = Math.min(A.min, B.min);
    max = Math.max(A.max, B.max);
  }

  @Override
  public OnLineStatistics clone() {
    return new OnLineStatistics(n, mean, m2, m3, m4, min, max);
  }

  public double getKurtosis() {
    return n * m4 / (m2 * m2) - 3;
  }

  public double getMax() {
    return max;
  }

  public double getMean() {
    return mean;
  }

  public double getMin() {
    return min;
  }

  public double getSkewness() {
    return Math.sqrt(n) * m3 / Math.pow(m2, 3.0 / 2.0);
  }

  public double getStandardDeviation() {
    return Math.sqrt(getVarance());
  }

  /**
   * Returns the sum of the weights for all data points added to the statistics.
   * If all weights were 1, then this value is the number of data points added.
   *
   * @return the sum of weights for every point currently contained in the
   *         statistics.
   */
  public double getSumOfWeights() {
    return n;
  }

  public double getVarance() {
    return m2 / (n - 1);
  }

  /**
   * Effectively removes a sample with the given value and weight from the
   * total. Removing values that have not been added may yield results that have
   * no meaning <br>
   * <br>
   * NOTE: {@link #getSkewness() } and {@link #getKurtosis() } are not currently
   * updated correctly
   *
   * @param x
   *          the value of the sample
   * @param weight
   *          the weight of the sample
   * @throws ArithmeticException
   *           if a negative weight is given
   */
  public void remove(final double x, final double weight) {
    if (weight < 0) {
      throw new ArithmeticException("Can not remove a negative weight");
    } else if (weight == 0) {
      return;
    }

    final double n1 = n;
    n -= weight;
    final double delta = x - mean;
    final double delta_n = delta * weight / n;
    final double delta_n2 = delta_n * delta_n;
    final double term1 = delta * delta_n * n1;

    mean -= delta_n;

    m2 -= weight * delta * (x - mean);
    // TODO m3 and m4 arent getting updated correctly
    m3 -= term1 * delta_n * (n - 2 + weight) - 3 * delta_n * m2;
    m4 -= term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3;
  }

  /**
   * Removes from this set of statistics the observations that where collected
   * in {@code B}.<br>
   * NOTE: removing statistics is not as numerically stable. The values of the
   * 3rd and 4th moments {@link #getSkewness() } and {@link #getKurtosis() }
   * will be inaccurate for many inputs. The {@link #getMin() min} and
   * {@link #getMax() max} can not be determined in this setting, and will not
   * be altered.
   *
   * @param B
   *          the set of statistics to remove
   */
  public void remove(final OnLineStatistics B) {
    final OnLineStatistics A = this;
    // XXX double compare.
    if (A.n == B.n) {
      n = mean = m2 = m3 = m4 = 0;
      min = max = null;
      return;
    } else if (B.n == 0) {
      return;// removed nothing!
    } else if (A.n < B.n) {
      throw new ArithmeticException("Can not have negative samples");
    }

    final double nX = A.n - B.n;
    final double nXsqrd = nX * nX;
    final double nAnB = B.n * A.n;
    final double AnSqrd = A.n * A.n;
    final double BnSqrd = B.n * B.n;

    final double delta = B.mean - A.mean;
    final double deltaSqrd = delta * delta;
    final double deltaCbd = deltaSqrd * delta;
    final double deltaQad = deltaSqrd * deltaSqrd;
    final double newMean = (A.n * A.mean - B.n * B.mean) / (A.n - B.n);
    final double newM2 = A.m2 - B.m2 - deltaSqrd / nX * nAnB;
    final double newM3 = A.m3 - B.m3 - deltaCbd * nAnB * (A.n - B.n) / nXsqrd
        - 3 * delta * (A.n * B.m2 - B.n * A.m2) / nX;
    final double newM4 = A.m4 - B.m4 - deltaQad * (nAnB * (AnSqrd - nAnB + BnSqrd) / (nXsqrd * nX))
        - 6 * deltaSqrd * (AnSqrd * B.m2 - BnSqrd * A.m2) / nXsqrd - 4 * delta * (A.n * B.m3 - B.n * A.m3) / nX;

    n = nX;
    mean = newMean;
    m2 = newM2;
    m3 = newM3;
    m4 = newM4;
  }

}
