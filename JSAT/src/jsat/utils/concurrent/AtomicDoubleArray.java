package jsat.utils.concurrent;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicLongArray;

/**
 * Provides a double array that can have individual values updated atomically.
 * It is backed by and mimics the implementation of {@link AtomicLongArray}. As
 * such, the methods have the same documentation
 *
 * @author Edward Raff
 */
public class AtomicDoubleArray implements Serializable {

  private static final long serialVersionUID = -8799170460903375842L;
  private final AtomicLongArray larray;

  /**
   * Creates a new AtomixDouble Array that is of the same length as the input
   * array. The values will be copied into the new object
   *
   * @param array
   *          the array of values to copy
   */
  public AtomicDoubleArray(final double[] array) {
    this(array.length);
    for (int i = 0; i < array.length; i++) {
      set(i, array[i]);
    }
  }

  /**
   * Creates a new AtomicDoubleArray of the given length, with all values
   * initialized to zero
   *
   * @param length
   *          the length of the array
   */
  public AtomicDoubleArray(final int length) {
    larray = new AtomicLongArray(length);
    final long ZERO = Double.doubleToRawLongBits(0.0);
    for (int i = 0; i < length; i++) {
      larray.set(i, ZERO);
    }
  }

  /**
   * Atomically adds the given value to the element at index {@code i}.
   *
   * @param i
   *          the index
   * @param delta
   *          the value to add
   * @return the updated value
   */
  public double addAndGet(final int i, final double delta) {
    while (true) {
      final double orig = get(i);
      final double newVal = orig + delta;
      if (compareAndSet(i, orig, newVal)) {
        return newVal;
      }
    }
  }

  /**
   * Atomically sets the element at position {@code i} to the given updated
   * value if the current value {@code ==} the expected value.
   *
   * @param i
   *          the index
   * @param expected
   *          the expected value
   * @param update
   *          the new value
   * @return true if successful. False return indicates that the actual value
   *         was not equal to the expected value.
   */
  public boolean compareAndSet(final int i, final double expected, final double update) {
    final long expectedL = Double.doubleToRawLongBits(expected);
    final long updateL = Double.doubleToRawLongBits(update);
    return larray.compareAndSet(i, expectedL, updateL);
  }

  /**
   * Gets the current value at position {@code i}.
   *
   * @param i
   *          the index
   * @return the current value
   */
  public double get(final int i) {
    return Double.longBitsToDouble(larray.get(i));
  }

  /**
   * Atomically adds the given value to the element at index {@code i}.
   *
   * @param i
   *          the index
   * @param delta
   *          the value to add
   * @return the previous value
   */
  public double getAndAdd(final int i, final double delta) {
    while (true) {
      final double orig = get(i);
      final double newVal = orig + delta;
      if (compareAndSet(i, orig, newVal)) {
        return orig;
      }
    }
  }

  /**
   * Atomically decrements by one the element at index {@code i}.
   *
   * @param i
   *          the index
   * @return the previous value
   */
  public double getAndDecrement(final int i) {
    return getAndAdd(i, -1.0);
  }

  /**
   * Atomically increments by one the element at index {@code i}.
   *
   * @param i
   *          the index
   * @return the previous value
   */
  public double getAndIncrement(final int i) {
    return getAndAdd(i, 1.0);
  }

  /**
   * Atomically sets the element at position {@code i} to the given value and
   * returns the old value.
   *
   * @param i
   *          the index
   * @param newValue
   *          the new value
   * @return the previous value
   */
  public double getAndSet(final int i, final double newValue) {
    final long oldL = larray.getAndSet(i, Double.doubleToRawLongBits(newValue));
    return Double.longBitsToDouble(oldL);
  }

  /**
   * Eventually sets the element at position {@code i} to the given value.
   *
   * @param i
   *          the index
   * @param newValue
   *          the new value
   */
  public void lazySet(final int i, final double newValue) {
    larray.lazySet(i, Double.doubleToRawLongBits(newValue));
  }

  /**
   * Returns the length of the array.
   *
   * @return the length of the array
   */
  public int length() {
    return larray.length();
  }

  /**
   * Sets the element at position {@code i} to the given value.
   *
   * @param i
   *          the index
   * @param newValue
   *          the new value
   */
  public void set(final int i, final double newValue) {
    larray.set(i, Double.doubleToRawLongBits(newValue));
  }

  /**
   * Atomically sets the element at position {@code i} to the given updated
   * value if the current value {@code ==} the expected value.
   *
   * <p>
   * May <a href="package-summary.html#Spurious">fail spuriously</a> and does
   * not provide ordering guarantees, so is only rarely an appropriate
   * alternative to {@code compareAndSet}.
   *
   * @param i
   *          the index
   * @param expected
   *          the expected value
   * @param update
   *          the new value
   * @return true if successful.
   */
  public boolean weakCompareAndSet(final int i, final double expected, final double update) {
    final long expectedL = Double.doubleToRawLongBits(expected);
    final long updateL = Double.doubleToRawLongBits(update);
    return larray.weakCompareAndSet(i, expectedL, updateL);
  }

}
