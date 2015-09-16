package jsat.utils;

import static jsat.utils.ClosedHashingUtil.DELETED;
import static jsat.utils.ClosedHashingUtil.EMPTY;
import static jsat.utils.ClosedHashingUtil.INT_MASK;
import static jsat.utils.ClosedHashingUtil.OCCUPIED;
import static jsat.utils.ClosedHashingUtil.getNextPow2TwinPrime;
import java.util.AbstractMap;
import java.util.AbstractSet;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.Set;

/**
 * A hash map for storing the primitive types of integers (as keys) to doubles
 * (as table). The implementation is based on Algorithm D (Open addressing with
 * double hashing) from Knuth's TAOCP page 528.
 *
 * @author Edward Raff
 */
public final class IntDoubleMap extends AbstractMap<Integer, Double> {

  /**
   * EntrySet class supports remove operations
   */
  private final class EntrySet extends AbstractSet<Entry<Integer, Double>> {

    final IntDoubleMap parentRef;

    public EntrySet(final IntDoubleMap parent) {
      parentRef = parent;
    }

    @Override
    public Iterator<Entry<Integer, Double>> iterator() {
      // find the first starting inded
      int START = 0;
      while (START < status.length && status[START] != OCCUPIED) {
        START++;
      }
      if (START == status.length) {
        return Collections.emptyIterator();
      }
      final int startPos = START;

      return new Iterator<Entry<Integer, Double>>() {
        int pos = startPos;
        int prevPos = -1;

        @Override
        public boolean hasNext() {
          return pos < status.length;
        }

        @Override
        public Entry<Integer, Double> next() {
          // final int make so that object remains good after we call next again
          final int oldPos = prevPos = pos++;
          // find next
          while (pos < status.length && status[pos] != OCCUPIED) {
            pos++;
          }
          // and return new object
          return new Entry<Integer, Double>() {

            @Override
            public Integer getKey() {
              return keys[oldPos];
            }

            @Override
            public Double getValue() {
              return table[oldPos];
            }

            @Override
            public Double setValue(final Double value) {
              final double old = table[oldPos];
              table[oldPos] = value;
              return old;
            }
          };
        }

        @Override
        public void remove() {
          // its ok to just call remove b/c nothing is re-ordered when we remove
          // an element, we just set the status to DELETED
          parentRef.remove(keys[prevPos]);
        }
      };
    }

    @Override
    public int size() {
      return used;
    }

  }

  private final static long EXTRA_INDEX_INFO = (long) Integer.MIN_VALUE << 32;

  /**
   * Returns a non-negative hash value
   *
   * @param key
   * @return
   */
  public static int h(int key) {
    // http://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    key = (key >>> 16 ^ key) & 0x45f9f3b;
    key = (key >>> 16 ^ key) & 0x45f9f3b;
    key = key >>> 16 ^ key;
    return key;
  }

  private final float loadFactor;
  private int used = 0;

  private byte[] status;

  private int[] keys;

  private double[] table;

  public IntDoubleMap() {
    this(32);
  }

  public IntDoubleMap(final int capacity) {
    this(capacity, 0.75f);
  }

  public IntDoubleMap(final int capacity, final float loadFactor) {
    if (capacity < 1) {
      throw new IllegalArgumentException("Capacity must be a positive value, not " + capacity);
    }
    if (loadFactor <= 0 || loadFactor >= 1 || Float.isNaN(loadFactor)) {
      throw new IllegalArgumentException("loadFactor must be in (0, 1), not " + loadFactor);
    }
    this.loadFactor = loadFactor;

    final int size = getNextPow2TwinPrime(Math.max(capacity, 4));
    status = new byte[size];
    keys = new int[size];
    table = new double[size];
  }

  @Override
  public void clear() {
    used = 0;
    Arrays.fill(status, EMPTY);
  }

  public boolean containsKey(final int key) {
    final int index = (int) (getIndex(key) & INT_MASK);
    return status[index] == OCCUPIED;// would be FREE if we didn't have the key
  }

  @Override
  public boolean containsKey(final Object key) {
    if (key instanceof Integer) {
      return containsKey(((Number) key).intValue());
    } else {
      return false;
    }
  }

  private void enlargeIfNeeded() {
    if (used < keys.length * loadFactor) {
      return;
    }
    // enlarge
    final byte[] oldSatus = status;
    final int[] oldKeys = keys;
    final double[] oldTable = table;

    final int newSize = getNextPow2TwinPrime(status.length * 3 / 2);// it will
                                                                    // actually
                                                                    // end up
                                                                    // doubling
                                                                    // in size
                                                                    // since we
                                                                    // have twin
                                                                    // primes
                                                                    // spaced
                                                                    // that was
    status = new byte[newSize];
    keys = new int[newSize];
    table = new double[newSize];

    used = 0;
    for (int oldIndex = 0; oldIndex < oldSatus.length; oldIndex++) {
      if (oldSatus[oldIndex] == OCCUPIED) {
        put(oldKeys[oldIndex], oldTable[oldIndex]);
      }
    }
  }

  @Override
  public Set<Entry<Integer, Double>> entrySet() {
    return new EntrySet(this);
  }

  /**
   * Gets the index of the given key. Based on that {@link #status} variable,
   * the index is either the location to insert OR the location of the key.
   *
   * This method returns 2 integer table in the long. The lower 32 bits are the
   * index that either contains the key, or is the first empty index.
   *
   * The upper 32 bits is the index of the first position marked as
   * {@link #DELETED} either {@link Integer#MIN_VALUE} if no position was marked
   * as DELETED while searching.
   *
   * @param key
   *          they key to search for
   * @return the mixed long containing the index of the first DELETED position
   *         and the position that the key is in or the first EMPTY position
   *         found
   */
  private long getIndex(final int key) {
    long extraInfo = EXTRA_INDEX_INFO;
    // D1
    final int hash = h(key) & INT_MASK;
    int i = hash % keys.length;

    // D2
    int satus_i = status[i];
    if (keys[i] == key && satus_i != DELETED || satus_i == EMPTY) {
      return extraInfo | i;
    }
    if (extraInfo == EXTRA_INDEX_INFO && satus_i == DELETED) {
      extraInfo = (long) i << 32;
    }

    // D3
    final int c = 1 + hash % (keys.length - 2);

    while (true)// this loop will terminate
    {
      // D4
      i -= c;
      if (i < 0) {
        i += keys.length;
      }
      // D5
      satus_i = status[i];
      if (keys[i] == key && satus_i != DELETED || satus_i == EMPTY) {
        return extraInfo | i;
      }
      if (extraInfo == EXTRA_INDEX_INFO && satus_i == DELETED) {
        extraInfo = (long) i << 32;
      }
    }
  }

  public int[] getRawKeyTable() {
    return keys;
  }

  public byte[] getRawStatusTable() {
    return status;
  }

  public double[] getRawValueTable() {
    return table;
  }

  /**
   *
   * @param key
   * @param delta
   * @return the new value stored for the given key
   */
  public double increment(final int key, final double delta) {
    if (Double.isNaN(delta)) {
      throw new IllegalArgumentException("NaN is not an allowable value");
    }

    final long pair_index = getIndex(key);
    final int deletedIndex = (int) (pair_index >>> 32);
    final int valOrFreeIndex = (int) (pair_index & INT_MASK);

    if (status[valOrFreeIndex] == OCCUPIED) {// easy case
      return table[valOrFreeIndex] += delta;
    }
    // else, not present
    double toReturn;
    int i = valOrFreeIndex;
    if (deletedIndex >= 0) {// use occupied spot instead
      i = deletedIndex;
    }

    status[i] = OCCUPIED;
    keys[i] = key;
    toReturn = table[i] = delta;
    used++;

    enlargeIfNeeded();

    return toReturn;
  }

  public double put(final int key, final double value) {
    if (Double.isNaN(value)) {
      throw new IllegalArgumentException("NaN is not an allowable value");
    }
    final long pair_index = getIndex(key);
    final int deletedIndex = (int) (pair_index >>> 32);
    final int valOrFreeIndex = (int) (pair_index & INT_MASK);

    double prev;
    if (status[valOrFreeIndex] == OCCUPIED) // easy case
    {
      prev = table[valOrFreeIndex];
      table[valOrFreeIndex] = value;
      return prev;
    }
    // else, not present
    prev = Double.NaN;
    int i = valOrFreeIndex;
    if (deletedIndex >= 0) {// use occupied spot instead
      i = deletedIndex;
    }

    status[i] = OCCUPIED;
    keys[i] = key;
    table[i] = value;
    used++;

    enlargeIfNeeded();

    return prev;
  }

  @Override
  public Double put(final Integer key, final Double value) {
    final double prev = put(key.intValue(), value.doubleValue());
    if (Double.isNaN(prev)) {
      return null;
    } else {
      return prev;
    }
  }

  /**
   *
   * @param key
   * @return the old value stored for this key, or {@link Double#NaN} if the key
   *         was not present in the map
   */
  public double remove(final int key) {
    final long pair_index = getIndex(key);
    final int valOrFreeIndex = (int) (pair_index & INT_MASK);
    if (status[valOrFreeIndex] == EMPTY) {// ret index is always EMPTY or
                                          // OCCUPIED
      return Double.NaN;
    }
    // else
    final double toRet = table[valOrFreeIndex];
    status[valOrFreeIndex] = DELETED;
    used--;
    return toRet;
  }

  @Override
  public Double remove(final Object key) {
    if (key instanceof Integer) {
      final double oldValue = remove(((Number) key).intValue());
      if (Double.isNaN(oldValue)) {
        return null;
      } else {
        return oldValue;
      }
    }

    return null;
  }

  @Override
  public int size() {
    return used;
  }
}
