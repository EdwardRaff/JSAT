package jsat.linear;

import static java.lang.Math.abs;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

import jsat.math.Function;
import jsat.math.IndexFunction;

/**
 * Vec is a object representing the math concept of a vector. A vector could be
 * either sparse or dense, where sparse vectors have a high number of zero
 * values that are not explicitly stored. <br>
 * <br>
 * This abstract class provides a large number of pre-implemented methods. Some
 * of which are implemented only for a dense vector, or may not be completely
 * efficient for the underlying implementation. Methods that should be
 * considered for overloading by an implementation will be indicated in the
 * documentation.
 *
 * @author Edward Raff
 */
public abstract class Vec implements Cloneable, Iterable<IndexValue>, Serializable {

  private static final long serialVersionUID = 9035784536820782955L;

  /**
   * Creates a dense vector full of random values in the range [0, 1]
   *
   * @param length
   *          the length of the random vector to create
   * @return a random vector of the specified length
   */
  public static Vec random(final int length) {
    return random(length, new Random());
  }

  /**
   * Creates a dense vector full of random values in the range [0, 1]
   *
   * @param length
   *          the length of the random vector to create
   * @param rand
   *          the source of randomness
   * @return a random vector of the specified length
   */
  public static Vec random(final int length, final Random rand) {
    final Vec v = new DenseVector(length);
    for (int i = 0; i < length; i++) {
      v.set(i, rand.nextDouble());
    }
    return v;
  }

  /**
   * Creates a dense vector full of zeros.
   *
   * @param length
   *          the length of the vector to create
   * @return a vector of zeros
   */
  public static Vec zeros(final int length) {
    return new DenseVector(length);
  }

  /**
   * Returns a new vector that is the result of {@code this + c}
   *
   * @param c
   *          the constant to add
   * @return the result of adding {@code c} to {@code this}
   */
  public Vec add(final double c) {
    final Vec toRet = getThisSide(null);
    toRet.mutableAdd(c);
    return toRet;
  }

  /**
   * Returns a new vector that is the result of {@code this + b}
   *
   * @param b
   *          the vector to add
   * @return the result of {@code b + this}
   */
  public Vec add(final Vec b) {
    final Vec toRet = getThisSide(b);
    toRet.mutableAdd(b);
    return toRet;
  }

  /**
   * Applies the given function to each and every value in the vector. <br>
   * <br>
   * This method should be overloaded for a serious implementation.
   *
   * @param f
   *          the single variable function to apply
   */
  public void applyFunction(final Function f) {
    for (int i = 0; i < length(); i++) {
      set(i, f.f(get(i)));
    }
  }

  /**
   * Applies the given function to each and every value in the vector. The
   * function takes 2 arguments, an arbitrary value, and then an index. The
   * index passed to the function is the index in the array that the value came
   * from. <br>
   * <br>
   * <b><i>NOTE:</i></b> Because negative values are invalid indexes. The given
   * function should return 0.0 when given a negative index, if and only if,
   * f(0,index) = 0 for any valid index. If f(0, index) != 0 for even one value
   * of index, it should return any non zero value when given a negative index.
   * <br>
   * <br>
   * IE: f(value_i, i) = x <br>
   * <br>
   * This method should be overloaded for a serious implementation.
   *
   * @param f
   *          the 2 dimensional index function to apply
   */
  public void applyIndexFunction(final IndexFunction f) {
    for (int i = 0; i < length(); i++) {
      set(i, f.indexFunc(get(i), i));
    }
  }

  /**
   * Creates a new array that contains all the values of this vector in the
   * appropriate indices
   *
   * @return a new array that is a copy of this vector
   */
  public double[] arrayCopy() {
    final double[] array = new double[length()];
    for (final IndexValue iv : this) {
      array[iv.getIndex()] = iv.getValue();
    }
    return array;
  }

  /**
   * Indicates whether or not this vector can be mutated. If {@code false}, any
   * method that contains "mutate" will not work. <br>
   * <br>
   * By default, this returns {@code true}
   *
   * @return {@code true} if the vector supports being altered, {@code false}
   *         other wise.
   */
  public boolean canBeMutated() {
    return true;
  }

  @Override
  abstract public Vec clone();

  /**
   * Copies the values of this Vector into another vector
   *
   * @param destination
   *          the vector to store the values in.
   * @throws ArithmeticException
   *           if the vectors are not of the same length
   */
  public void copyTo(final Vec destination) {
    if (length() != destination.length()) {
      throw new ArithmeticException("Source and destination must be the same size");
    }
    if (isSparse()) {
      destination.zeroOut();
      for (final IndexValue iv : this) {
        destination.set(iv.getIndex(), iv.getValue());
      }
    } else {
      for (int i = 0; i < length(); i++) {
        destination.set(i, get(i));
      }
    }
  }

  /**
   * Copies the values of this vector into a column of another Matrix.
   *
   * @param A
   *          the matrix to store the contents of this vector in
   * @param col
   *          the column of the matrix to store the values to
   */
  public void copyToCol(final Matrix A, final int col) {
    if (length() != A.rows()) {
      throw new ArithmeticException("Destination matrix does not have the same number of rows as this has rows");
    }
    for (int i = 0; i < length(); i++) {
      A.set(i, col, get(i));
    }
  }

  /**
   * Copies the values of this vector into a row of another Matrix
   *
   * @param A
   *          the matrix to store the contents of this vector in
   * @param row
   *          the row of the matrix to store the values to
   * @throws ArithmeticException
   *           if the columns of the matrix is not the same as the length of
   *           this vector.
   */
  public void copyToRow(final Matrix A, final int row) {
    if (length() != A.cols()) {
      throw new ArithmeticException("Destination matrix does not have the same number of columns as this has rows");
    }
    for (int i = 0; i < length(); i++) {
      A.set(row, i, get(i));
    }
  }

  /**
   * Returns a new vector that is the result of {@code this / c}
   *
   * @param c
   *          the constant to divide by
   * @return the result of {@code this / c}
   */
  public Vec divide(final double c) {
    final Vec toRet = getThisSide(null);
    toRet.mutableDivide(c);
    return toRet;
  }

  /**
   * Computes the dot product between two vectors, which is equivalent to<br>
   * <big>&Sigma;</big> this<sub>i</sub>*v<sub>i</sub> <br>
   * <br>
   * This method should be overloaded for a serious implementation.
   *
   * @param v
   *          the other vector
   * @return the dot product of this vector and another
   */
  public double dot(final Vec v) {
    double dot = 0;
    if (!isSparse() && v.isSparse()) {
      for (final IndexValue iv : v) {
        dot += get(iv.getIndex()) * iv.getValue();
      }
    } else if (isSparse() && !v.isSparse()) {
      for (final IndexValue iv : this) {
        dot += iv.getValue() * v.get(iv.getIndex());
      }
    } else if (isSparse() && v.isSparse()) {
      final Iterator<IndexValue> aIter = this.getNonZeroIterator();
      final Iterator<IndexValue> bIter = v.getNonZeroIterator();

      if (nnz() == 0 || v.nnz() == 0) {
        return 0;// All zeros? dot is zer
      }

      // each must have at least one
      IndexValue aCur = aIter.next();
      IndexValue bCur = bIter.next();

      while (aCur != null && bCur != null)// set to null when have none left
      {
        if (aCur.getIndex() == bCur.getIndex()) {
          dot += aCur.getValue() * bCur.getValue();
          if (aIter.hasNext()) {
            aCur = aIter.next();
          } else {
            aCur = null;
          }

          if (bIter.hasNext()) {
            bCur = bIter.next();
          } else {
            bCur = null;
          }
        } else if (aCur.getIndex() < bCur.getIndex()) {
          // Move a over to try and get the indecies equal
          if (aIter.hasNext()) {
            aCur = aIter.next();
          } else {
            aCur = null;
          }
        } else// b is too small, move it over and try to get them lined up
          if (bIter.hasNext()) {
          bCur = bIter.next();
        } else {
          bCur = null;
        }
      }

    } else {
      for (int i = 0; i < length(); i++) {
        dot += get(i) * v.get(i);
      }
    }

    return dot;
  }

  @Override
  public boolean equals(final Object obj) {
    return equals(obj, 0.0);
  }

  public boolean equals(final Object obj, double range) {
    if (!(obj instanceof Vec)) {
      return false;
    }
    final Vec other = (Vec) obj;
    range = abs(range);

    final Iterator<IndexValue> thisIter = iterator();
    final Iterator<IndexValue> otherIter = other.iterator();
    if (!thisIter.hasNext()) {
      return !otherIter.hasNext();
    } else if (!otherIter.hasNext()) {
      return false;
    }

    IndexValue av = thisIter.next();
    IndexValue bv = otherIter.next();

    do {
      boolean nextA = false, nextB = false;
      if (av.getIndex() == bv.getIndex()) {
        if (abs(av.getValue() - bv.getValue()) > range) {
          return false;
        }
        nextA = nextB = true;
      } else if (av.getIndex() < bv.getIndex()) {
        if (abs(av.getValue()) > range) {
          return false;
        }
        nextA = true;
      } else if (av.getIndex() > bv.getIndex()) {
        if (abs(bv.getValue()) > range) {
          return false;
        }
        nextB = true;
      }

      if (nextA) {
        av = thisIter.hasNext() ? thisIter.next() : null;
      }
      if (nextB) {
        bv = otherIter.hasNext() ? otherIter.next() : null;
      }
    } while (av != null && bv != null);

    while (av != null) {
      if (abs(av.getValue()) > range) {
        return false;
      }
      av = thisIter.hasNext() ? thisIter.next() : null;
    }

    while (bv != null) {
      if (abs(bv.getValue()) > range) {
        return false;
      }
      bv = otherIter.hasNext() ? otherIter.next() : null;
    }

    return true;
  }

  /**
   * Gets the value stored at a specific index in the vector
   *
   * @param index
   *          the index to access
   * @return the double value in the vector
   * @throws IndexOutOfBoundsException
   *           if the index given is greater than or equal to its
   *           {@link #length() }
   */
  abstract public double get(int index);

  /**
   * Returns an iterator that will go over the non zero values in the given
   * vector. The iterator does not support the {@link Iterator#remove() }
   * method.
   *
   * @return an iterator for the non zero index value pairs.
   */
  public Iterator<IndexValue> getNonZeroIterator() {
    return getNonZeroIterator(0);
  }

  /**
   * Returns an iterator that will go over the non zero values starting from the
   * specified index in the given vector. The iterator does not support the
   * {@link Iterator#remove() } method. <br>
   * <br>
   * This method should be overloaded for a serious implementation.
   *
   * @param start
   *          the first index (inclusive) to start returning non-zero values
   *          from
   * @return an iterator for the non zero index value pairs
   */
  public Iterator<IndexValue> getNonZeroIterator(final int start) {
    // Need a little class magic
    final Vec magic = this;
    int i;
    for (i = start; i < magic.length(); i++) {
      if (magic.get(i) != 0.0) {
        break;
      }
    }
    final int fnz = magic.length() == 0 || magic.length() <= i || magic.get(i) == 0.0 ? -1 : i;
    final Iterator<IndexValue> itor = new Iterator<IndexValue>() {
      int curIndex = 0;
      int nextNonZero = fnz;

      IndexValue indexValue = new IndexValue(-1, Double.NaN);

      @Override
      public boolean hasNext() {
        return nextNonZero >= 0;
      }

      @Override
      public IndexValue next() {
        if (nextNonZero == -1) {
          return null;
        }
        indexValue.setIndex(nextNonZero);
        indexValue.setValue(get(nextNonZero));

        int i = nextNonZero + 1;
        nextNonZero = -1;
        for (; i < magic.length(); i++) {
          if (get(i) != 0.0) {
            nextNonZero = i;
            break;
          }
        }

        return indexValue;
      }

      @Override
      public void remove() {
        throw new UnsupportedOperationException("Not supported yet.");
      }
    };

    return itor;
  }

  /**
   * Returns a suitable vector that can be altered for some function of the form
   * a <i>op</i> b, where {@code a = this}
   *
   * @param other
   *          the other vector. May be {@code null}
   * @return the mutable vector
   */
  private Vec getThisSide(final Vec other) {
    if (canBeMutated()) {
      return clone();
    }
    if (other == null) {
      if (isSparse()) {
        return new SparseVector(this);
      } else {
        return new DenseVector(this);
      }
    }
    if (isSparse() && other.isSparse()) {
      return new SparseVector(this);
    } else {
      return new DenseVector(this);
    }
  }

  /**
   * Provides a hashcode for Vectors. All vector implementations should return
   * the same result for cases when {@link #equals(java.lang.Object) } returns
   * true. Below is the code used for this class<br>
   * <p>
   * <code>
   * int result = 1;<br>
   * <br>
   * for (int i = 0; i &lt; length(); i++) <br>
   * {<br>
   * double val = get(i);<br>
   * if(val != 0)<br>
   * {<br>
   * long bits = Double.doubleToLongBits(val);<br>
   * result = 31 * result + (int)(bits ^ (bits &gt;&gt;&gt; 32));<br>
   * result = 31 * result + i;<br>
   * }<br>
   * }<br>
   * <br>
   * return 31* result + length();<br>
   * </code>
   * </p>
   *
   * @return the hash code for a vector
   */
  @Override
  public int hashCode() {
    int result = 1;

    for (int i = 0; i < length(); i++) {
      final double val = get(i);
      if (val != 0) {
        final long bits = Double.doubleToLongBits(val);
        result = 31 * result + (int) (bits ^ bits >>> 32);
        result = 31 * result + i;
      }
    }

    return 31 * result + length();
  }

  /**
   * Increments the value stored at a specified index in the vector
   *
   * @param index
   *          the index to access
   * @param val
   *          the value to store in the index
   * @throws IndexOutOfBoundsException
   *           if the index given is greater than or equal to its
   *           {@link #length() }
   */
  public void increment(final int index, final double val) {
    set(index, val + get(index));
  }

  /**
   * Indicates whether or not this vector is optimized for sparce computation,
   * meaning that most values in the vector are zero - and considered implicit.
   * Only non-zero values are stored.
   *
   * @return <tt>true</tt> if the vector is sparce, <tt>false</tt> otherwise.
   */
  abstract public boolean isSparse();

  @Override
  public Iterator<IndexValue> iterator() {
    return getNonZeroIterator(0);
  }

  /**
   * Computes the kurtosis of this vector, which is the 4th moment.
   *
   * @return the kurtosis
   */
  public double kurtosis() {
    final double mean = mean();

    double tmp = 0;
    final int length = length();
    int used = 0;

    for (final IndexValue iv : this) {
      tmp += pow(iv.getValue() - mean, 4);
      used++;
    }

    // All the zero's we skipped
    tmp += pow(-mean, 4) * (length - used);

    return tmp / (pow(standardDeviation(), 4) * (length - 1)) - 3;
  }

  /**
   * Returns the length of this vector
   *
   * @return the length of this vector
   */
  abstract public int length();

  /**
   * Returns the maximum value stored in this vector
   *
   * @return the maximum value in this vector
   */
  public double max() {
    if (isSparse() && nnz() < length()) {
      double max = 0.0;
      for (final IndexValue iv : this) {
        max = Math.max(max, iv.getValue());
      }
      return max;
    } else {
      double max = get(0);
      for (int i = 1; i < length(); i++) {
        max = Math.max(max, get(i));
      }
      return max;
    }
  }

  /**
   * Computes the mean value of all values stored in this vector
   *
   * @return the mean value
   */
  public double mean() {
    return sum() / length();
  }

  /**
   * Returns the median value in this vector
   *
   * @return the median
   */
  public double median() {
    final Vec copy = sortedCopy();
    if (copy.length() % 2 != 0) {
      return copy.get(copy.length() / 2);
    } else {
      return copy.get(copy.length() / 2) / 2 + copy.get(copy.length() / 2 + 1) / 2;
    }
  }

  /**
   * Returns the minimum value stored in this vector
   *
   * @return the minimum value in this vector
   */
  public double min() {
    if (isSparse() && nnz() < length()) {
      double min = 0.0;
      for (final IndexValue iv : this) {
        min = Math.min(min, iv.getValue());
      }
      return min;
    } else {
      double min = get(0);
      for (int i = 1; i < length(); i++) {
        min = Math.min(min, get(i));
      }
      return min;
    }
  }

  /**
   * Returns a new vector that is the result of {@code this * c}
   *
   * @param c
   *          the constant to multiply by
   * @return the result of {@code this * c}
   */
  public Vec multiply(final double c) {
    final Vec toRet = getThisSide(null);
    toRet.mutableMultiply(c);
    return toRet;
  }

  /**
   * If this is vector <tt>a</tt>, this this computes b = b + c <tt>a</tt>
   * <sup>T</sup>*<tt>A</tt>
   *
   * @param c
   *          the constant factor to multiply by
   * @param A
   *          the matrix to multiple by
   * @param b
   *          the vector to mutate by adding the result to
   */
  public void multiply(final double c, final Matrix A, final Vec b) {
    if (length() != A.rows()) {
      throw new ArithmeticException(
          "Vector x Matrix dimensions do not agree [1," + length() + "] x [" + A.rows() + ", " + A.cols() + "]");
    }
    if (b.length() != A.cols()) {
      throw new ArithmeticException("Destination vector is not the right size");
    }

    if (!isSparse()) {
      for (int i = 0; i < length(); i++) {
        final double this_i = c * get(i);
        for (int j = 0; j < A.cols(); j++) {
          b.increment(j, this_i * A.get(i, j));
        }
      }
    } else {
      for (final IndexValue iv : this) {
        final int i = iv.getIndex();
        final double this_i = c * iv.getValue();
        for (int j = 0; j < A.cols(); j++) {
          b.increment(j, this_i * A.get(i, j));
        }
      }
    }
  }

  /**
   * Returns a new vector that is the result of the vector matrix product
   * <tt>this<sup>T</sup>A</tt>
   *
   * @param A
   *          the matrix to multiply with
   * @return the vector matrix product
   */
  public Vec multiply(final Matrix A) {
    final DenseVector b = new DenseVector(A.cols());
    this.multiply(A, b);
    return b;
  }

  /**
   * If this is vector <tt>a</tt>, this this computes b = b + <tt>a</tt>
   * <sup>T</sup>*<tt>A</tt>
   *
   * @param A
   *          the matrix to multiple by
   * @param b
   *          the vector to mutate by adding the result to
   */
  public void multiply(final Matrix A, final Vec b) {
    multiply(1, A, b);
  }

  /**
   * Alters this vector such that <tt>this</tt> = <tt>this</tt> + <tt>c</tt>
   * <br>
   * <br>
   * This method should be overloaded for a serious implementation.
   *
   * @param c
   *          a scalar constant to add to each value in this vector
   */
  public void mutableAdd(final double c) {
    for (int i = 0; i < length(); i++) {
      increment(i, c);
    }
  }

  /**
   * Alters this vector such that <tt>this</tt> = <tt>this</tt> + <tt>c</tt> *
   * <tt>b</tt> <br>
   * <br>
   * This method should be overloaded for a serious implementation.
   *
   * @param c
   *          a scalar constant
   * @param b
   *          the vector to add to this
   */
  public void mutableAdd(final double c, final Vec b) {
    if (length() != b.length()) {
      throw new ArithmeticException("Vectors must have the same length, not " + length() + " and " + b.length());
    }
    if (b.isSparse()) {
      for (final IndexValue iv : b) {
        increment(iv.getIndex(), c * iv.getValue());
      }
    } else {
      for (int i = 0; i < length(); i++) {
        increment(i, c * b.get(i));
      }
    }
  }

  /**
   * Alters this vector such that <tt>this</tt> = <tt>this</tt> + <tt>b</tt>
   *
   * @param b
   *          the vector to add to this
   * @throws ArithmeticException
   *           if the vectors do not have the same length
   */
  public void mutableAdd(final Vec b) {
    this.mutableAdd(1, b);
  }

  /**
   * Mutates {@code this /= c} <br>
   * <br>
   * This method should be overloaded for a serious implementation.
   *
   * @param c
   *          the constant to divide by
   */
  public void mutableDivide(final double c) {
    for (int i = 0; i < length(); i++) {
      set(i, get(i) / c);
    }
  }

  /**
   * Mutates {@code this *= c} <br>
   * <br>
   * This method should be overloaded for a serious implementation.
   *
   * @param c
   *          the constant to multiply by
   */
  public void mutableMultiply(final double c) {
    for (int i = 0; i < length(); i++) {
      set(i, get(i) * c);
    }
  }

  /**
   * Mutates {@code this} by dividing each value by the value in {@code b} that
   * has the same index <br>
   * <br>
   * This method should be overloaded for a serious implementation.
   *
   * @param b
   *          the vector to pairwise divide by
   */
  public void mutablePairwiseDivide(final Vec b) {
    if (length() != b.length()) {
      throw new ArithmeticException("Vector lengths do not agree " + length() + " vs " + b.length());
    }
    for (int i = 0; i < length(); i++) {
      set(i, get(i) / b.get(i));
    }
  }

  /**
   * Mutates {@code this} by multiplying each value by the value in {@code b}
   * that has the same index. <br>
   * <br>
   * This method should be overloaded for a serious implementation.
   *
   * @param b
   *          the vector to pairwise multiply by
   */
  public void mutablePairwiseMultiply(final Vec b) {
    if (length() != b.length()) {
      throw new ArithmeticException("Vector lengths do not agree " + length() + " vs " + b.length());
    }
    for (int i = 0; i < length(); i++) {
      set(i, get(i) * b.get(i));
    }
  }

  /**
   * Alters this vector such that <tt>this</tt> = <tt>this</tt> - <tt>c</tt>
   *
   * @param c
   *          the scalar constant to subtract from all values in this vector
   */
  public void mutableSubtract(final double c) {
    mutableAdd(-c);
  }

  /**
   * Alters this vector such that <tt>this</tt> = <tt>this</tt> - <tt>c</tt> *
   * <tt>b</tt>
   *
   * @param c
   *          a scalar constant
   * @param b
   *          the vector to subtract from this
   * @throws ArithmeticException
   *           if the vectors do not have the same length
   */
  public void mutableSubtract(final double c, final Vec b) {
    this.mutableAdd(-c, b);
  }

  /**
   * Alters this vector such that <tt>this</tt> = <tt>this</tt> - <tt>b</tt>
   *
   * @param b
   *          the vector to subtract from this
   * @throws ArithmeticException
   *           if the vectors are not the same length
   */
  public void mutableSubtract(final Vec b) {
    this.mutableAdd(-1, b);
  }

  /**
   * Computes the number of non zero values in this vector
   *
   * @return the number of non zero values stored
   */
  public int nnz() {
    int nnz = 0;
    for (final IndexValue i : this) {
      nnz++;
    }
    return nnz;
  }

  /**
   * Mutates this vector to be normalized by the L<sub>2</sub> norm
   */
  public void normalize() {
    mutableDivide(Math.max(pNorm(2.0), 1e-10));
  }

  /**
   * Returns a new vector that is the result of normalizing this vector by the L
   * <sub>2</sub> norm
   *
   * @return a normalized version of this vector
   */
  public Vec normalized() {
    final Vec toRet = getThisSide(null);
    toRet.normalize();
    return toRet;
  }

  /**
   * Returns a new vector that is the result of dividing each value in
   * {@code this} by the value in the same index in {@code b}
   *
   * @param b
   *          the vector to pairwise divide by
   * @return the result of pairwise division of {@code this} by {@code b}
   */
  public Vec pairwiseDivide(final Vec b) {
    final Vec toRet = getThisSide(b);
    toRet.mutablePairwiseDivide(b);
    return toRet;
  }

  /**
   * Returns a new vector that is the result of multiplying each value in
   * {@code this} by its corresponding value in {@code b}
   *
   * @param b
   *          the vector to pairwise multiply by
   * @return the result of the pairwise multiplication of {@code b} onto the
   *         values of {@code this}
   */
  public Vec pairwiseMultiply(final Vec b) {
    final Vec toRet = getThisSide(b);
    toRet.mutablePairwiseMultiply(b);
    return toRet;
  }

  /**
   * Returns the p-norm of this vector.
   *
   * @param p
   *          the norm type. 2 is a common value
   * @return the p-norm of this vector
   */
  public double pNorm(final double p) {
    if (p <= 0) {
      throw new IllegalArgumentException("norm must be a positive value, not " + p);
    }
    double result = 0;
    if (p == 1) {
      for (final IndexValue iv : this) {
        result += abs(iv.getValue());
      }
    } else if (p == 2) {
      for (final IndexValue iv : this) {
        result += iv.getValue() * iv.getValue();
      }
      result = Math.sqrt(result);
    } else if (Double.isInfinite(p)) {
      for (final IndexValue iv : this) {
        result = Math.max(result, abs(iv.getValue()));
      }
    } else {
      for (final IndexValue iv : this) {
        result += pow(abs(iv.getValue()), p);
      }
      result = pow(result, 1 / p);
    }
    return result;
  }

  /**
   * Returns the p-norm distance between this and another vector y.
   *
   * @param p
   *          the distance type. 2 is the common value
   * @param y
   *          the other vector to compare against
   * @return the p-norm distance
   */
  public double pNormDist(final double p, final Vec y) {
    final Iterator<IndexValue> thisIter = iterator();
    final Iterator<IndexValue> otherIter = y.iterator();
    if (!thisIter.hasNext()) {
      if (!otherIter.hasNext()) {
        return 0;
      } else {
        return y.pNorm(p);
      }
    } else if (!otherIter.hasNext()) {
      return pNorm(p);
    }

    double result = 0;

    IndexValue av = thisIter.next();
    IndexValue bv = otherIter.next();

    do {
      boolean nextA = false, nextB = false;
      if (av.getIndex() == bv.getIndex()) {
        result += pow(abs(av.getValue() - bv.getValue()), p);
        nextA = nextB = true;
      } else if (av.getIndex() < bv.getIndex()) {
        result += pow(abs(av.getValue()), p);
        nextA = true;
      } else if (av.getIndex() > bv.getIndex()) {
        result += pow(abs(bv.getValue()), p);
        nextB = true;
      }

      if (nextA) {
        av = thisIter.hasNext() ? thisIter.next() : null;
      }
      if (nextB) {
        bv = otherIter.hasNext() ? otherIter.next() : null;
      }
    } while (av != null && bv != null);

    // accumulate left overs
    while (av != null) {
      result += pow(abs(av.getValue()), p);
      av = thisIter.hasNext() ? thisIter.next() : null;
    }

    while (bv != null) {
      result += pow(abs(bv.getValue()), p);
      bv = otherIter.hasNext() ? otherIter.next() : null;
    }

    return pow(result, 1 / p);
  }

  /**
   * Sets the value stored at a specified index in the vector
   *
   * @param index
   *          the index to access
   * @param val
   *          the value to store in the index
   * @throws IndexOutOfBoundsException
   *           if the index given is greater than or equal to its
   *           {@link #length() }
   */
  abstract public void set(int index, double val);

  /**
   * Computes the skewness of this vector, which is the 3rd moment.
   *
   * @return the skewness
   */
  public double skewness() {
    final double mean = mean();

    double tmp = 0;
    final int length = length();
    int used = 0;

    for (final IndexValue iv : this) {
      tmp += pow(iv.getValue() - mean, 3);
      used++;
    }

    // All the zero's we skiped
    tmp += pow(-mean, 3) * (length - used);

    final double s1 = tmp / (pow(standardDeviation(), 3) * (length - 1));

    if (length >= 3) {// We can use the bias corrected formula
      return sqrt(length * (length - 1)) / (length - 2) * s1;
    }

    return s1;
  }

  /**
   * Returns a copy of this array with the values moved around so that they are
   * in sorted order
   *
   * @return a new array in sorted order
   */
  public Vec sortedCopy() {
    final double[] arrayCopy = arrayCopy();
    Arrays.sort(arrayCopy);
    return new DenseVector(arrayCopy);
  }

  /**
   * Computes the standard deviation of the values in this vector
   *
   * @return the standard deviation
   */
  public double standardDeviation() {
    return Math.sqrt(variance());
  }

  /**
   * Returns a new vector that is the result of {@code this - c}
   *
   * @param c
   *          the constant to subtract
   * @return the result of {@code this - c}
   */
  public Vec subtract(final double c) {
    return add(-c);
  }

  /**
   * Returns a new vector that is the result of {@code this - b}
   *
   * @param b
   *          the vector to subtract from {@code this}
   * @return the result of {@code this - b}
   */
  public Vec subtract(final Vec b) {
    final Vec toRet = getThisSide(b);
    toRet.mutableSubtract(b);
    return toRet;
  }

  /**
   * Computes the sum of the values in this vector
   *
   * @return the sum of this vector's values
   */
  public double sum() {
    /*
     * Uses Kahan summation algorithm, which is more accurate then naively
     * summing the values in floating point. Though it does not guarenty the
     * best possible accuracy
     *
     * See: http://en.wikipedia.org/wiki/Kahan_summation_algorithm
     */

    double sum = 0;
    double c = 0;
    for (final IndexValue iv : this) {
      final double d = iv.getValue();
      final double y = d - c;
      final double t = sum + y;
      c = t - sum - y;
      sum = t;
    }

    return sum;
  }

  @Override
  public String toString() {
    final StringBuilder sb = new StringBuilder("[");
    sb.append(get(0));
    for (int i = 1; i < length(); i++) {
      sb.append(",").append(get(i));
    }
    sb.append("]");
    return sb.toString();
  }

  /**
   * Computes the variance of the values in this vector, which is
   * {@link #standardDeviation() }<sup>2</sup>
   *
   * @return the variance
   */
  public double variance() {
    final double mu = mean();
    double variance = 0;

    final double N = length();

    int used = 0;
    for (final IndexValue x : this) {
      used++;
      variance += Math.pow(x.getValue() - mu, 2) / N;
    }
    // Now add all the zeros we skipped into it
    variance += (length() - used) * Math.pow(0 - mu, 2) / N;

    return variance;
  }

  /**
   * Zeroes out all values in this vector <br>
   * <br>
   * This method should be overloaded for a serious implementation.
   */
  public void zeroOut() {
    for (int i = 0; i < length(); i++) {
      set(i, 0.0);
    }
  }

}
