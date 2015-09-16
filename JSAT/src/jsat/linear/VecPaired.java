package jsat.linear;

import java.util.Comparator;
import java.util.Iterator;

/**
 * This data structure allows to wrap a Vector so that it is associated with
 * some object time. Note, that operations that return a vector will not be a
 * Paired Vector, as there is no reason to associate a different vector with
 * this vector's pair.
 *
 * @author Edward Raff
 */
public class VecPaired<V extends Vec, P> extends Vec {

  private static final long serialVersionUID = 8039272826439917423L;

  /**
   * This method is used assuming multiple VecPaired are used together. The
   * implementation of the vector may have logic to handle the case that the
   * other vector is of the same type. This will go through every layer of
   * VecPaired to return the final base vector.
   *
   * @param b
   *          a Vec, that may or may not be an instance of {@link VecPaired}
   * @return the final Vec backing b, which may be b itself.
   */
  public static Vec extractTrueVec(Vec b) {
    while (b instanceof VecPaired) {
      b = ((VecPaired) b).getVector();
    }
    return b;
  }

  public static <V extends Vec, P extends Comparable<P>> Comparator<VecPaired<V, P>> vecPairedComparator() {
    final Comparator<VecPaired<V, P>> comp = new Comparator<VecPaired<V, P>>() {

      @Override
      public int compare(final VecPaired<V, P> o1, final VecPaired<V, P> o2) {
        return o1.getPair().compareTo(o2.getPair());
      }
    };
    return comp;
  }

  private V vector;

  private P pair;

  public VecPaired(final V v, final P p) {
    this.vector = v;
    this.pair = p;
  }

  @Override
  public Vec add(final double c) {
    return vector.add(c);
  }

  @Override
  public Vec add(Vec b) {
    b = extractTrueVec(b);
    return vector.add(b);
  }

  @Override
  public double[] arrayCopy() {
    return vector.arrayCopy();
  }

  @Override
  public Vec clone() {
    return new VecPaired(vector.clone(), pair);
  }

  @Override
  public Vec divide(final double c) {
    return vector.divide(c);
  }

  @Override
  public double dot(Vec v) {
    v = extractTrueVec(v);
    return this.vector.dot(v);
  }

  @Override
  public boolean equals(final Object obj) {
    return vector.equals(obj);
  }

  @Override
  public boolean equals(final Object obj, final double range) {
    return vector.equals(obj, range);
  }

  @Override
  public double get(final int index) {
    return vector.get(index);
  }

  @Override
  public Iterator<IndexValue> getNonZeroIterator() {
    if (extractTrueVec(vector) instanceof SparseVector) {
      return extractTrueVec(vector).getNonZeroIterator();
    }
    return super.getNonZeroIterator();
  }

  public P getPair() {
    return pair;
  }

  public V getVector() {
    return vector;
  }

  @Override
  public int hashCode() {
    return vector.hashCode();
  }

  @Override
  public boolean isSparse() {
    return vector.isSparse();
  }

  @Override
  public double kurtosis() {
    return vector.kurtosis();
  }

  @Override
  public int length() {
    return vector.length();
  }

  @Override
  public double max() {
    return vector.max();
  }

  @Override
  public double mean() {
    return vector.mean();
  }

  @Override
  public double median() {
    return vector.median();
  }

  @Override
  public double min() {
    return vector.min();
  }

  @Override
  public Vec multiply(final double c) {
    return vector.multiply(c);
  }

  @Override
  public void multiply(final double c, final Matrix A, final Vec b) {
    vector.multiply(c, A, b);
  }

  @Override
  public void mutableAdd(final double c) {
    vector.mutableAdd(c);
  }

  @Override
  public void mutableAdd(final double c, Vec b) {
    b = extractTrueVec(b);

    this.vector.mutableAdd(c, b);
  }

  @Override
  public void mutableAdd(Vec b) {
    b = extractTrueVec(b);
    vector.mutableAdd(b);
  }

  @Override
  public void mutableDivide(final double c) {
    vector.mutableDivide(c);
  }

  @Override
  public void mutableMultiply(final double c) {
    vector.mutableMultiply(c);
  }

  @Override
  public void mutablePairwiseDivide(Vec b) {
    b = extractTrueVec(b);
    vector.mutablePairwiseDivide(b);
  }

  @Override
  public void mutablePairwiseMultiply(Vec b) {
    b = extractTrueVec(b);
    vector.mutablePairwiseDivide(b);
  }

  @Override
  public void mutableSubtract(Vec b) {
    b = extractTrueVec(b);
    vector.mutableSubtract(b);
  }

  @Override
  public int nnz() {
    return vector.nnz();
  }

  @Override
  public void normalize() {
    vector.normalize();
  }

  @Override
  public Vec normalized() {
    return vector.normalized();
  }

  @Override
  public Vec pairwiseDivide(Vec b) {
    b = extractTrueVec(b);
    return vector.pairwiseDivide(b);
  }

  @Override
  public Vec pairwiseMultiply(Vec b) {
    b = extractTrueVec(b);
    return vector.pairwiseMultiply(b);
  }

  @Override
  public double pNorm(final double p) {
    return vector.pNorm(p);
  }

  @Override
  public double pNormDist(final double p, Vec y) {
    y = extractTrueVec(y);
    return vector.pNormDist(p, y);
  }

  @Override
  public void set(final int index, final double val) {
    vector.set(index, val);
  }

  public void setPair(final P pair) {
    this.pair = pair;
  }

  public void setVector(final V vector) {
    this.vector = vector;
  }

  @Override
  public double skewness() {
    return vector.skewness();
  }

  @Override
  public Vec sortedCopy() {
    return vector.sortedCopy();
  }

  @Override
  public double standardDeviation() {
    return vector.standardDeviation();
  }

  @Override
  public Vec subtract(Vec b) {
    b = extractTrueVec(b);
    return vector.subtract(b);
  }

  @Override
  public double sum() {
    return vector.sum();
  }

  ;

  @Override
  public String toString() {
    return vector.toString();
  }

  @Override
  public double variance() {
    return vector.variance();
  }

}
