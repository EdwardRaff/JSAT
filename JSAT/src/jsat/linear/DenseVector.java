package jsat.linear;

import static java.lang.Math.abs;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

import java.util.Arrays;
import java.util.List;

/**
 * A vector implementation that is dense, meaning all values are allocated -
 * even if their values will be implicitly zero.
 *
 * @author Edward Raff
 */
public class DenseVector extends Vec {

  private static final long serialVersionUID = -889493251793828934L;

  /**
   * Returns a new dense vector backed by the given array. This is a weak
   * reference, the given array should no longer be altered - as it will effect
   * the values of the dense vector.
   *
   * @param array
   *          the array to use as the backing of a dense vector
   * @return a Dense Vector that is backed using the given array
   */
  public static DenseVector toDenseVec(final double... array) {
    return new DenseVector(array);
  }

  protected double[] array;
  private Double sumCache = null;
  private Double varianceCache = null;
  private Double minCache = null;
  private Double maxCache = null;
  private final int startIndex;

  private final int endIndex;

  /**
   * Creates a new Dense Vector that uses the given array as its values. Its
   * values will not be copied, and raw access and mutations tot he given array
   * may occur.
   *
   * @param array
   *          the backing array to use for a new vector of the same length
   */
  public DenseVector(final double[] array) {
    this(array, 0, array.length);
  }

  /**
   * Creates a new Dense Vector that uses the given array as its values. Its
   * values will not be copied, and raw access and mutations tot he given array
   * may occur.
   *
   * @param array
   *          the backing array to use for a new vector
   * @param start
   *          the first index in the array, inclusive, to mark the start of the
   *          vector.
   * @param end
   *          the last index in the array, exclusive, to mark the end of the
   *          vector.
   */
  public DenseVector(final double[] array, final int start, final int end) {
    this.array = array;
    startIndex = start;
    endIndex = end;
  }

  /**
   * Creates a new Dense Vector of zeros
   *
   * @param length
   *          the length of the vector
   */
  public DenseVector(final int length) {
    if (length < 0) {
      throw new ArithmeticException("You can not have a negative dimension vector");
    }
    array = new double[length];
    startIndex = 0;
    endIndex = array.length;
  }

  /**
   * Creates a new vector of the length of the given list, and values copied
   * over in order.
   *
   * @param list
   *          the list of values to copy into a new vector
   */
  public DenseVector(final List<Double> list) {
    array = new double[list.size()];
    for (int i = 0; i < list.size(); i++) {
      array[i] = list.get(i);
    }
    startIndex = 0;
    endIndex = array.length;
  }

  /**
   * Creates a new Dense Vector that contains a copy of the values in the given
   * vector
   *
   * @param toCopy
   *          the vector to copy
   */
  public DenseVector(final Vec toCopy) {
    this(toCopy.length());
    for (final IndexValue iv : toCopy) {
      set(iv.getIndex(), iv.getValue());
    }
  }

  @Override
  public double[] arrayCopy() {
    return Arrays.copyOfRange(array, startIndex, endIndex);
  }

  /**
   * nulls out the cached summary statistics, should be called every time the
   * data set changes
   */
  private void clearCaches() {
    sumCache = null;
    varianceCache = null;
    minCache = null;
    maxCache = null;
  }

  @Override
  public Vec clone() {
    final DenseVector copy = new DenseVector(length());

    System.arraycopy(array, startIndex, copy.array, 0, length());

    return copy;
  }

  public DenseVector deepCopy() {
    return new DenseVector(Arrays.copyOf(array, array.length));
  }

  @Override
  public double dot(final Vec v) {
    if (length() != v.length()) {
      throw new ArithmeticException("Vectors must have the same length");
    }

    if (v.isSparse()) {
      return v.dot(this);
    }

    double dot = 0;
    for (int i = startIndex; i < endIndex; i++) {
      dot += array[i] * v.get(i - startIndex);
    }

    return dot;
  }

  @Override
  public boolean equals(final Object obj) {
    if (!(obj instanceof Vec)) {
      return false;
    }
    final Vec otherVec = (Vec) obj;

    if (length() != otherVec.length()) {
      return false;
    }
    for (int i = startIndex; i < endIndex; i++) {
      if (get(i) != otherVec.get(i)) {
        return false;
      }
    }

    return true;
  }

  @Override
  public boolean equals(final Object obj, double range) {
    if (!(obj instanceof Vec)) {
      return false;
    }
    final Vec otherVec = (Vec) obj;
    range = Math.abs(range);

    if (length() != otherVec.length()) {
      return false;
    }
    for (int i = startIndex; i < endIndex; i++) {
      if (Math.abs(get(i) - otherVec.get(i)) > range) {
        return false;
      }
    }

    return true;
  }

  @Override
  public double get(final int index) {
    return array[index + startIndex];
  }

  @Override
  public boolean isSparse() {
    return false;
  }

  @Override
  public double kurtosis() {
    final double mean = mean();

    double tmp = 0;

    for (int i = startIndex; i < endIndex; i++) {
      tmp += pow(array[i] - mean, 4);
    }
    tmp /= length();

    return tmp / pow(variance(), 2) - 3;
  }

  @Override
  public int length() {
    return endIndex - startIndex;
  }

  @Override
  public double max() {
    if (maxCache != null) {
      return maxCache;
    }
    double result = array[startIndex];
    for (int i = startIndex + 1; i < endIndex; i++) {
      result = Math.max(result, array[i]);
    }

    return maxCache = result;
  }

  @Override
  public double median() {
    final double[] copy = Arrays.copyOfRange(array, startIndex, endIndex);

    Arrays.sort(copy);

    if (copy.length % 2 == 1) {
      return copy[copy.length / 2];
    } else {
      return copy[copy.length / 2] / 2 + copy[copy.length / 2 + 1] / 2;// Divisions
                                                                       // by 2
                                                                       // then
                                                                       // add is
                                                                       // more
                                                                       // numericaly
                                                                       // stable
    }
  }

  @Override
  public double min() {
    if (minCache != null) {
      return minCache;
    }

    double result = array[startIndex];
    for (int i = startIndex + 1; i < endIndex; i++) {
      result = Math.min(result, array[i]);
    }

    return minCache = result;
  }

  @Override
  public void multiply(final double c, final Matrix A, final Vec b) {
    if (length() != A.rows()) {
      throw new ArithmeticException(
          "Vector x Matrix dimensions do not agree [1," + length() + "] x [" + A.rows() + ", " + A.cols() + "]");
    }
    if (b.length() != A.cols()) {
      throw new ArithmeticException("Destination vector is not the right size");
    }

    for (int i = 0; i < length(); i++) {
      final double this_i = c * array[i + startIndex];
      for (int j = 0; j < A.cols(); j++) {
        b.increment(j, this_i * A.get(i, j));
      }
    }
  }

  @Override
  public void mutableAdd(final double c) {
    clearCaches();
    for (int i = startIndex; i < endIndex; i++) {
      array[i] += c;
    }
  }

  @Override
  public void mutableAdd(final double c, final Vec b) {
    if (length() != b.length()) {
      throw new ArithmeticException("Can not add vectors of unequal length");
    }

    clearCaches();
    if (b.isSparse()) {
      for (final IndexValue iv : b) {
        array[iv.getIndex()] += c * iv.getValue();
      }
    } else {
      for (int i = startIndex; i < endIndex; i++) {
        array[i] += c * b.get(i);
      }
    }
  }

  @Override
  public void mutableDivide(final double c) {
    clearCaches();
    for (int i = startIndex; i < endIndex; i++) {
      array[i] /= c;
    }
  }

  @Override
  public void mutableMultiply(final double c) {
    clearCaches();
    for (int i = startIndex; i < endIndex; i++) {
      array[i] *= c;
    }
  }

  @Override
  public void mutablePairwiseDivide(final Vec b) {
    if (length() != b.length()) {
      throw new ArithmeticException("Vectors must have the same length");
    }
    for (int i = startIndex; i < endIndex; i++) {
      array[i] /= b.get(i);
    }
  }

  @Override
  public void mutablePairwiseMultiply(final Vec b) {
    if (length() != b.length()) {
      throw new ArithmeticException("Vectors must have the same length");
    }
    for (int i = startIndex; i < endIndex; i++) {
      array[i] *= b.get(i);
    }
  }

  @Override
  public void mutableSubtract(final double c) {
    clearCaches();
    for (int i = startIndex; i < endIndex; i++) {
      array[i] -= c;
    }
  }

  @Override
  public void normalize() {
    double sum = 0;

    for (int i = startIndex; i < endIndex; i++) {
      sum += array[i] * array[i];
    }

    sum = Math.sqrt(sum);

    mutableDivide(Math.max(sum, 1e-10));
  }

  @Override
  public double pNorm(final double p) {
    if (p <= 0) {
      throw new IllegalArgumentException("norm must be a positive value, not " + p);
    }
    double result = 0;
    if (p == 1) {
      for (int i = startIndex; i < endIndex; i++) {
        result += abs(array[i]);
      }
    } else if (p == 2) {
      for (int i = startIndex; i < endIndex; i++) {
        result += array[i] * array[i];
      }
      result = Math.sqrt(result);
    } else if (Double.isInfinite(p)) {
      for (int i = startIndex; i < endIndex; i++) {
        result = Math.max(result, abs(array[i]));
      }
    } else {
      for (int i = startIndex; i < endIndex; i++) {
        result += Math.pow(Math.abs(array[i]), p);
      }
      result = pow(result, 1 / p);
    }
    return result;
  }

  @Override
  public double pNormDist(final double p, final Vec y) {
    if (length() != y.length()) {
      throw new ArithmeticException("Vectors must be of the same length");
    }

    double norm = 0;
    if (y.isSparse()) {
      int lastIndx = -1;
      for (final IndexValue iv : y) {
        for (int i = lastIndx + 1; i < iv.getIndex(); i++) {
          // add all the indecies we skipped
          norm += Math.pow(Math.abs(array[i]), p);
        }
        lastIndx = iv.getIndex();
        // add current
        norm += Math.pow(Math.abs(array[iv.getIndex()] - iv.getValue()), p);
      }

      // Tailing zeros
      for (int i = lastIndx + 1; i < y.length(); i++) {
        norm += Math.pow(Math.abs(array[i]), p);
      }
    } else {
      for (int i = startIndex; i < endIndex; i++) {
        norm += Math.pow(Math.abs(array[i] - y.get(i)), p);
      }
    }
    return Math.pow(norm, 1.0 / p);
  }

  @Override
  public void set(final int index, final double val) {
    clearCaches();
    array[index + startIndex] = val;
  }

  @Override
  public double skewness() {
    final double mean = mean();

    double tmp = 0;

    for (int i = startIndex; i < endIndex; i++) {
      tmp += pow(array[i] - mean, 3);
    }

    final double s1 = tmp / (pow(standardDeviation(), 3) * (array.length - 1));

    if (array.length >= 3) {// We can use the bias corrected formula
      return sqrt(array.length * (array.length - 1)) / (array.length - 2) * s1;
    }

    return s1;
  }

  @Override
  public DenseVector sortedCopy() {
    final double[] copy = Arrays.copyOfRange(array, startIndex, endIndex);

    Arrays.sort(copy);

    return new DenseVector(copy);
  }

  @Override
  public double sum() {
    if (sumCache != null) {
      return sumCache;
      /*
       * Uses Kahan summation algorithm, which is more accurate then naively
       * summing the values in floating point. Though it does not guarenty the
       * best possible accuracy
       *
       * See: http://en.wikipedia.org/wiki/Kahan_summation_algorithm
       */
    }

    double sum = 0;
    double c = 0;
    for (int i = startIndex; i < endIndex; i++) {
      final double y = array[i] - c;
      final double t = sum + y;
      c = t - sum - y;
      sum = t;
    }

    return sumCache = sum;
  }

  @Override
  public double variance() {
    if (varianceCache != null) {
      return varianceCache;
    }
    final double mu = mean();
    double tmp = 0;

    final double N = length();

    for (int i = startIndex; i < endIndex; i++) {
      tmp += pow(array[i] - mu, 2) / N;
    }

    return varianceCache = tmp;
  }
}
