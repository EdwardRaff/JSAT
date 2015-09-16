package jsat.utils;

import java.io.Serializable;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * Provides a modifiable implementation of a List using an array of longs. This
 * provides considerable memory efficency improvements over using an
 * {@link ArrayList} to store longs. Null is not allowed into the list.
 *
 * @author Edward Raff
 */
public class LongList extends AbstractList<Long>implements Serializable {

  private static final long serialVersionUID = 3060216677615816178L;

  /**
   * Creates and returns a view of the given long array that requires only a
   * small object allocation.
   *
   * @param array
   *          the array to wrap into a list
   * @param length
   *          the number of values of the array to use, starting from zero
   * @return a list view of the array
   */
  public static List<Long> view(final long[] array, final int length) {
    return new LongList(array, length);
  }

  private long[] array;

  private int end;

  /**
   * Creates a new LongList
   */
  public LongList() {
    this(10);
  }

  public LongList(final Collection<Long> c) {
    this(c.size());
    addAll(c);
  }

  /**
   * Creates a new LongList with the given initial capacity
   *
   * @param capacity
   *          the starting internal storage space size
   */
  public LongList(final int capacity) {
    array = new long[capacity];
    end = 0;
  }

  private LongList(final long[] array, final int end) {
    this.array = array;
    this.end = end;
  }

  /**
   * Operates exactly as {@link #add(int, java.lang.Long) }
   *
   * @param index
   *          the index to add the value into
   * @param element
   *          the value to add
   */
  public void add(final int index, final long element) {
    boundsCheck(index);
    enlargeIfNeeded(1);
    System.arraycopy(array, index, array, index + 1, end - index);
    array[index] = element;
  }

  @Override
  public void add(final int index, final Long element) {
    add(index, element.longValue());
  }

  /**
   * Operates exactly as {@link #add(java.lang.Long) }
   *
   * @param e
   *          the value to add
   * @return true if it was added, false otherwise
   */
  public boolean add(final long e) {
    enlargeIfNeeded(1);
    array[end++] = e;
    return true;
  }

  @Override
  public boolean add(final Long e) {
    if (e == null) {
      return false;
    }
    return add(e.longValue());
  }

  private void boundsCheck(final int index) throws IndexOutOfBoundsException {
    if (index >= end) {
      throw new IndexOutOfBoundsException("List of of size " + size() + ", index requested was " + index);
    }
  }

  @Override
  public void clear() {
    end = 0;
  }

  private void enlargeIfNeeded(final int i) {
    while (end + i > array.length) {
      array = Arrays.copyOf(array, Math.max(array.length * 2, 8));
    }
  }

  @Override
  public Long get(final int index) {
    return getL(index);
  }

  /**
   * Operates exactly as {@link #get(int) }
   *
   * @param index
   *          the index of the value to get
   * @return the value at the index
   */
  public long getL(final int index) {
    boundsCheck(index);
    return array[index];
  }

  @Override
  public Long remove(final int index) {
    if (index < 0 || index > size()) {
      throw new IndexOutOfBoundsException("Can not remove invalid index " + index);
    }
    final long removed = array[index];

    for (int i = index; i < end - 1; i++) {
      array[i] = array[i + 1];
    }
    end--;
    return removed;
  }

  /**
   * Operates exactly as {@link #set(int, java.lang.Long) }
   *
   * @param index
   *          the index to set
   * @param element
   *          the value to set
   * @return the previous value at the index
   */
  public long set(final int index, final long element) {
    boundsCheck(index);
    final long prev = array[index];
    array[index] = element;
    return prev;
  }

  @Override
  public Long set(final int index, final Long element) {
    return set(index, element.longValue());
  }

  @Override
  public int size() {
    return end;
  }
}
