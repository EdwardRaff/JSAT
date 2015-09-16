package jsat.utils.concurrent;

import static java.lang.Double.doubleToRawLongBits;
import static java.lang.Double.longBitsToDouble;

import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * @author Edward Raff
 */
final public class AtomicDouble {

  private final AtomicLong base;

  public AtomicDouble(final double value) {
    base = new AtomicLong();
    set(value);
  }

  /**
   * Atomically adds the given value to the current value.
   *
   * @param delta
   *          the value to add
   * @return the updated value
   */
  public final double addAndGet(final double delta) {
    while (true) {
      final double orig = get();
      final double newVal = orig + delta;
      if (compareAndSet(orig, newVal)) {
        return newVal;
      }
    }
  }

  public boolean compareAndSet(final double expect, final double update) {
    return base.compareAndSet(doubleToRawLongBits(expect), doubleToRawLongBits(update));
  }

  public double get() {
    return longBitsToDouble(base.get());
  }

  public double getAndAdd(final double delta) {
    while (true) {
      final double orig = get();
      final double newVal = orig + delta;
      if (compareAndSet(orig, newVal)) {
        return orig;
      }
    }
  }

  public void set(final double val) {
    base.set(Double.doubleToRawLongBits(val));
  }

  @Override
  public String toString() {
    return "" + get();
  }

  public boolean weakCompareAndSet(final double expect, final double update) {
    return base.weakCompareAndSet(doubleToRawLongBits(expect), doubleToRawLongBits(update));
  }

}
