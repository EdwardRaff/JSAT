package jsat.linear;

import java.util.Random;

/**
 * Stores a Vector full of random values in constant O(1) space by re-computing
 * all matrix values on the fly as need. This allows memory reduction and use
 * when it is necessary to use the matrix with a large sparse data set, where
 * some matrix values may never even be used - or used very infrequently. <br>
 * <br>
 * Because the values of the random vector are computed on the fly, the Random
 * Vector can not be altered. If attempted, an exception will be thrown.
 *
 * @author Edward Raff
 */
public abstract class RandomVector extends Vec {

  private static final long serialVersionUID = -1587968421978707875L;
  /*
   * Implementation note: It is assumed that the default random object is a PRNG
   * with a single word / long of state. A higher quality PRNG cant be used if
   * it requires too many words of state, as the initalization will then
   * dominate the computation of every index.
   */
  private final int length;
  private final long seedMult;

  private final ThreadLocal<Random> localRand = new ThreadLocal<Random>() {
    @Override
    protected Random initialValue() {
      return new Random(1);// seed will get set by user
    }
  };

  /**
   * Creates a new Random Vector object
   *
   * @param length
   *          the length of the vector
   */
  public RandomVector(final int length) {
    this(length, new Random().nextLong());
  }

  /**
   * Creates a new Random Vector object
   *
   * @param length
   *          the length of the vector
   * @param seedMult
   *          a value to multiply with the seed used for each individual index.
   *          It should be a large value
   */
  public RandomVector(final int length, final long seedMult) {
    if (length <= 0) {
      throw new IllegalArgumentException("Vector length must be positive, not " + length);
    }
    this.length = length;
    this.seedMult = seedMult;
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  protected RandomVector(final RandomVector toCopy) {
    this(toCopy.length, toCopy.seedMult);
  }

  @Override
  public boolean canBeMutated() {
    return false;
  }

  @Override
  abstract public Vec clone();

  @Override
  public double dot(final Vec v) {
    double dot = 0;

    for (final IndexValue iv : v) {
      dot += get(iv.getIndex()) * iv.getValue();
    }
    return dot;
  }

  @Override
  public double get(final int index) {
    final long seed = (index + length) * seedMult;
    final Random rand = localRand.get();
    rand.setSeed(seed);
    return getVal(rand);
  }

  /**
   * Computes the value of an index given the already initialized {@link Random}
   * object. This is called by the {@link #get(int) } method, and will make sure
   * that the correct seed is set before calling this method.
   *
   * @param rand
   *          the PRNG to generate the index value from
   * @return the value for a given index based on the given PRNG
   */
  abstract protected double getVal(Random rand);

  @Override
  public boolean isSparse() {
    return false;
  }

  @Override
  public int length() {
    return length;
  }

  @Override
  public double max() {
    double max = -Double.MAX_VALUE;
    for (final IndexValue iv : this) {
      max = Math.min(iv.getValue(), max);
    }
    return max;
  }

  @Override
  public double min() {
    double min = Double.MAX_VALUE;
    for (final IndexValue iv : this) {
      min = Math.min(iv.getValue(), min);
    }
    return min;
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
      final double this_i = c * get(i);
      for (int j = 0; j < A.cols(); j++) {
        b.increment(j, this_i * A.get(i, j));
      }
    }
  }

  @Override
  public void mutableAdd(final double c) {
    throw new UnsupportedOperationException("RandomVector can not be altered");
  }

  @Override
  public void mutableAdd(final double c, final Vec b) {
    throw new UnsupportedOperationException("RandomVector can not be altered");
  }

  @Override
  public void mutableDivide(final double c) {
    throw new UnsupportedOperationException("RandomVector can not be altered");
  }

  @Override
  public void mutableMultiply(final double c) {
    throw new UnsupportedOperationException("RandomVector can not be altered");
  }

  @Override
  public void mutablePairwiseDivide(final Vec b) {
    throw new UnsupportedOperationException("RandomVector can not be altered");
  }

  @Override
  public void mutablePairwiseMultiply(final Vec b) {
    throw new UnsupportedOperationException("RandomVector can not be altered");
  }

  @Override
  public void set(final int index, final double val) {
    throw new UnsupportedOperationException("RandomVector can not be altered");
  }

  @Override
  public Vec sortedCopy() {
    final DenseVector dv = new DenseVector(this);
    return dv.sortedCopy();
  }
}
