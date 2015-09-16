package jsat.linear;

import java.util.Iterator;

/**
 * A wrapper for a vector that represents the vector multiplied by a scalar
 * constant. This allows for using and altering the value multiplied by a
 * constant factor quickly, especially when the multiplicative factor must be
 * changed often. Mutable operations will alter the underling vector, and all
 * operations will automatically be scaled on a per element basis as needed.
 * <br>
 * <br>
 * If a point is reached where the multiplicative constant will be infrequently
 * modified relative to the use of the vector, it may be more efficient to use
 * the original vector scaled appropriately. This can be done by first calling
 * {@link #embedScale() } and then calling {@link #getBase() } . <br>
 * <br>
 * When the multiplicative constant is set to zero, the underlying base vector
 * is {@link Vec#zeroOut() zeroed out} and the constant reset to 1.
 *
 * @author Edward Raff
 */
public class ScaledVector extends Vec {

  private static final long serialVersionUID = 7357893957632067299L;
  private double scale;
  private final Vec base;

  /**
   * Creates a new scaled vector
   *
   * @param scale
   *          the initial scaling constant
   * @param base
   *          the vector to implicitly scale
   */
  public ScaledVector(final double scale, final Vec base) {
    this.scale = scale;
    this.base = base;
  }

  /**
   * Creates a new scaled vector with a default scale of 1.
   *
   * @param vec
   *          the vector to implicitly scale
   */
  public ScaledVector(final Vec vec) {
    this(1.0, vec);
  }

  @Override
  public double[] arrayCopy() {
    final double[] copy = base.arrayCopy();
    for (int i = 0; i < copy.length; i++) {
      copy[i] *= scale;
    }
    return copy;
  }

  @Override
  public Vec clone() {
    return new ScaledVector(scale, base.clone());
  }

  @Override
  public double dot(final Vec v) {
    return scale * base.dot(v);
  }

  /**
   * Embeds the current scale factor into the base vector, so that the current
   * scale factor can be set to 1.
   */
  public void embedScale() {
    base.mutableMultiply(scale);
    scale = 1;
  }

  @Override
  public double get(final int index) {
    return base.get(index) * scale;
  }

  /**
   * Returns the base vector that is being scaled
   *
   * @return the base vector that is being scaled
   */
  public Vec getBase() {
    return base;
  }

  @Override
  public Iterator<IndexValue> getNonZeroIterator(final int start) {
    final Iterator<IndexValue> origIter = base.getNonZeroIterator(start);

    final Iterator<IndexValue> wrapedIter = new Iterator<IndexValue>() {

      @Override
      public boolean hasNext() {
        return origIter.hasNext();
      }

      @Override
      public IndexValue next() {
        final IndexValue iv = origIter.next();
        if (iv != null) {
          iv.setValue(scale * iv.getValue());
        }
        return iv;
      }

      @Override
      public void remove() {
        origIter.remove();
      }
    };

    return wrapedIter;
  }

  /**
   * Returns the current scale in use
   *
   * @return the current scale in use
   */
  public double getScale() {
    return scale;
  }

  @Override
  public boolean isSparse() {
    return base.isSparse();
  }

  @Override
  public double kurtosis() {
    return base.kurtosis(); // kurtosis is scale invariant
  }

  @Override
  public int length() {
    return base.length();
  }

  @Override
  public double max() {
    if (scale >= 0) {
      return base.max() * scale;
    } else {
      return base.min() * scale;
    }
  }

  @Override
  public double mean() {
    return scale * base.mean();
  }

  @Override
  public double median() {
    return scale * base.median();
  }

  @Override
  public double min() {
    if (scale >= 0) {
      return base.min() * scale;
    } else {
      return base.max() * scale;
    }
  }

  @Override
  public void multiply(final double c, final Matrix A, final Vec b) {
    base.multiply(c / scale, A, b);
  }

  @Override
  public void mutableAdd(final double c) {
    base.mutableAdd(c / scale);
  }

  @Override
  public void mutableAdd(final double c, final Vec b) {
    base.mutableAdd(c / scale, b);
  }

  @Override
  public void mutableDivide(final double c) {
    scale /= c;
    if (scale == 0.0) {
      zeroOut();
    }
  }

  @Override
  public void mutableMultiply(final double c) {
    scale *= c;
    if (scale == 0.0) {
      zeroOut();
    } else if (Math.abs(scale) < 1e-10 || Math.abs(scale) > 1e10) {
      embedScale();
    }
  }

  @Override
  public void mutablePairwiseDivide(final Vec b) {
    base.mutablePairwiseDivide(b);
  }

  @Override
  public void mutablePairwiseMultiply(final Vec b) {
    base.mutablePairwiseMultiply(b);
  }

  @Override
  public int nnz() {
    return base.nnz();
  }

  @Override
  public double pNorm(final double p) {
    return scale * base.pNorm(p);
  }

  @Override
  public void set(final int index, final double val) {
    base.set(index, val / scale);
  }

  /**
   * Explicitly sets the current scale to the given value<br>
   * <br>
   * NOTE: If the scale is set to zero, the underlying base vector will be
   * zeroed out, and the scale set to 1.
   *
   * @param scale
   *          the new multiplicative constant to set for the scale
   */
  public void setScale(final double scale) {
    if (scale == 0.0) {
      zeroOut();
    } else {
      this.scale = scale;
    }
  }

  @Override
  public double skewness() {
    return base.skewness();// skew is scale invariant
  }

  @Override
  public Vec sortedCopy() {
    return new ScaledVector(scale, base.sortedCopy());
  }

  @Override
  public double standardDeviation() {
    return scale * base.standardDeviation();
  }

  @Override
  public double sum() {
    return scale * base.sum();
  }

  @Override
  public void zeroOut() {
    scale = 1.0;
    base.zeroOut();
  }

}
