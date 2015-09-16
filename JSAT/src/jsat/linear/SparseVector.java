package jsat.linear;

import static java.lang.Math.abs;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import jsat.math.Function;
import jsat.math.IndexFunction;
import jsat.utils.DoubleList;
import jsat.utils.IndexTable;

/**
 * Provides a vector implementation that is sparse. It does not allocate space
 * for a vector of the specified size, and only stores non zero values. All
 * values not stored are implicitly zero. <br>
 * Operations that change several zero values in a sparse vector to non-zero
 * values may have degraded performance. <br>
 * Sparce vector should never be used unless at least half the values are zero.
 * If more then half the values are non-zero, it will use more memory then an
 * equivalent {@link DenseVector}. The more values that are zero in the vector,
 * the better its performance will be.
 *
 * @author Edward Raff
 */
public class SparseVector extends Vec {

  private static final long serialVersionUID = 8591745505666264662L;
  /**
   * Length of the vector
   */
  private int length;
  /**
   * number of indices used in this vector
   */
  protected int used;
  /**
   * The mapping to true index values
   */
  protected int[] indexes;
  /**
   * The Corresponding values for each index
   */
  protected double[] values;

  private Double sumCache = null;
  private Double varianceCache = null;
  private Double minCache = null;
  private Double maxCache = null;

  /**
   * Creates a new sparse vector of the given length that is all zero values.
   *
   * @param length
   *          the length of the sparse vector
   */
  public SparseVector(final int length) {
    this(length, 10);
  }

  /**
   * Creates a new sparse vector of the specified length, and pre-allocates
   * enough internal state to hold {@code capacity} non zero values. The vector
   * itself will start out with all zero values.
   *
   * @param length
   *          the length of the sparse vector
   * @param capacity
   *          the number of non zero values to allocate space for
   */
  public SparseVector(final int length, final int capacity) {
    this(new int[capacity], new double[capacity], length, 0);
  }

  /**
   * Creates a new sparse vector backed by the given arrays. Modifying the
   * arrays will modify the vector, and no validation will be done. This
   * constructor should only be used in performance necessary scenarios<br>
   * To make sure the input values are valid, the {@code indexes } values must
   * be increasing and all values less than {@code length} and greater than
   * {@code -1} up to the first {@code used} indices.<br>
   * All the values stored in {@code values} must be non zero and can not be a
   * special value. <br>
   * {@code used} must be greater than -1 and less than the length of the
   * {@code indexes} and {@code values} arrays. <br>
   * The {@code indexes} and {@code values} arrays must be the exact same length
   *
   * @param indexes
   *          the array to store the index locations in
   * @param values
   *          the array to store the index values in
   * @param length
   *          the length of the sparse vector
   * @param used
   *          the number of non zero values in the vector taken from the given
   *          input arrays.
   */
  public SparseVector(final int[] indexes, final double[] values, final int length, final int used) {
    if (values.length != indexes.length) {
      throw new IllegalArgumentException();
    }
    if (used < 0 || used > length || used > values.length) {
      throw new IllegalArgumentException();
    }
    if (length <= 0) {
      throw new IllegalArgumentException();
    }
    this.used = used;
    this.length = length;
    this.indexes = indexes;
    this.values = values;
  }

  /**
   * Creates a new sparse vector of the same length as {@code vals} and sets
   * each value to the values in the list.
   *
   * @param vals
   *          the list of values to create a vector from
   */
  public SparseVector(final List<Double> vals) {
    this(vals.size());
    int z = 0;
    for (int i = 0; i < vals.size(); i++) {
      if (vals.get(i) != 0) {
        if (z >= indexes.length) {
          indexes = Arrays.copyOf(indexes, indexes.length * 3 / 2);
          values = Arrays.copyOf(values, values.length * 3 / 2);
        }
        indexes[z] = i;
        values[z++] = vals.get(i);
      }
    }
  }

  /**
   * Creates a new sparse vector by copying the values from another
   *
   * @param toCopy
   *          the vector to copy the values of
   */
  public SparseVector(final Vec toCopy) {
    this(toCopy.length(), toCopy.nnz());
    for (final IndexValue iv : toCopy) {
      indexes[used] = iv.getIndex();
      values[used++] = iv.getValue();
    }
  }

  @Override
  public void applyFunction(final Function f) {
    if (f.f(0.0) != 0.0) {
      super.applyFunction(f);
    } else// Then we only need to apply it to the non zero values!
    {
      for (int i = 0; i < used; i++) {
        values[i] = f.f(values[i]);
      }
    }
  }

  @Override
  public void applyIndexFunction(final IndexFunction f) {
    if (f.f(0.0, -1) != 0.0) {
      super.applyIndexFunction(f);
    } else// Then we only need to apply it to the non zero values!
    {
      /*
       * The indexFunction may turn a value to zero, if so, we need to shift
       * everything over and skip based on how many zeros have been created
       */
      int skip = 0;
      for (int i = 0; i < used; i++) {
        indexes[i - skip] = indexes[i];
        values[i - skip] = f.indexFunc(values[i], i);
        if (values[i - skip] == 0.0) {
          skip++;
        }
      }

      used -= skip;
    }
  }

  @Override
  public double[] arrayCopy() {
    final double[] array = new double[length()];

    for (int i = 0; i < used; i++) {
      array[indexes[i]] = values[i];
    }

    return array;
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
  public SparseVector clone() {
    final SparseVector copy = new SparseVector(length, Math.max(used, 10));

    System.arraycopy(values, 0, copy.values, 0, used);
    System.arraycopy(indexes, 0, copy.indexes, 0, used);
    copy.used = used;

    return copy;
  }

  @Override
  public void copyTo(final Vec destination) {
    if (destination instanceof SparseVector) {
      final SparseVector other = (SparseVector) destination;
      if (other.indexes.length < used) {
        other.indexes = Arrays.copyOf(indexes, used);
        other.values = Arrays.copyOf(values, used);
        other.used = used;
        other.clearCaches();
      } else {
        other.used = used;
        other.clearCaches();
        System.arraycopy(indexes, 0, other.indexes, 0, used);
        System.arraycopy(values, 0, other.values, 0, used);
      }
    } else {
      super.copyTo(destination);
    }
  }

  @Override
  public double dot(final Vec v) {
    double dot = 0;

    if (v instanceof SparseVector) {
      final SparseVector b = (SparseVector) v;
      int p1 = 0, p2 = 0;
      while (p1 < used && p2 < b.used) {
        final int a1 = indexes[p1], a2 = b.indexes[p2];
        if (a1 == a2) {
          dot += values[p1++] * b.values[p2++];
        } else if (a1 > a2) {
          p2++;
        } else {
          p1++;
        }
      }
    } else if (v.isSparse()) {
      return super.dot(v);
    } else {
      for (int i = 0; i < used; i++) {
        dot += values[i] * v.get(indexes[i]);
      }
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

    int z = 0;
    for (int i = 0; i < length(); i++) {
      // Move through until we hit the next null element, comparing the other
      // vec to zero
      while (z < used && indexes[z] > i) {
        if (otherVec.get(i++) != 0) {
          return false;
        }
      }

      // We made it! (or are at the end). Is our non zero value the same?
      if (z < used && indexes[z] == i) {
        if (values[z++] != otherVec.get(i)) {
          return false;
        }
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

    int z = 0;
    for (int i = 0; i < length(); i++) {
      // Move through until we hit the next null element, comparing the other
      // vec to zero
      while (z < used && indexes[z] > i) {
        if (Math.abs(otherVec.get(i++)) > range) {
          return false;
        }
      }

      // We made it! (or are at the end). Is our non zero value the same?
      if (z < used && indexes[z] == i) {
        if (Math.abs(values[z++] - otherVec.get(i)) > range) {
          return false;
        }
      }
    }

    return true;
  }

  @Override
  public double get(final int index) {
    if (index > length - 1 || index < 0) {
      throw new ArithmeticException("Can not access an index larger then the vector or a negative index");
    }

    final int location = Arrays.binarySearch(indexes, 0, used, index);

    if (location < 0) {
      return 0.0;
    } else {
      return values[location];
    }
  }

  /**
   * Returns the index of the last non-zero value, or -1 if all values are zero.
   *
   * @return the index of the last non-zero value, or -1 if all values are zero.
   */
  public int getLastNonZeroIndex() {
    if (used == 0) {
      return -1;
    }
    return indexes[used - 1];
  }

  @Override
  public Iterator<IndexValue> getNonZeroIterator(final int start) {
    if (used <= 0) {
      return Collections.EMPTY_LIST.iterator();
    }
    final int startPos;
    if (start <= indexes[0]) {
      startPos = 0;
    } else {
      final int tmpIndx = Arrays.binarySearch(indexes, 0, used, start);
      if (tmpIndx >= 0) {
        startPos = tmpIndx;
      } else {
        startPos = -tmpIndx - 1;
      }
    }
    final Iterator<IndexValue> itor = new Iterator<IndexValue>() {
      int curUsedPos = startPos;
      IndexValue indexValue = new IndexValue(-1, Double.NaN);

      @Override
      public boolean hasNext() {
        return curUsedPos < used;
      }

      @Override
      public IndexValue next() {
        indexValue.setIndex(indexes[curUsedPos]);
        indexValue.setValue(values[curUsedPos++]);
        return indexValue;
      }

      @Override
      public void remove() {
        throw new UnsupportedOperationException("Not supported yet.");
      }
    };
    return itor;
  }

  @Override
  public int hashCode() {
    int result = 1;

    for (int i = 0; i < used; i++) {
      final long bits = Double.doubleToLongBits(values[i]);
      result = 31 * result + (int) (bits ^ bits >>> 32);
      result = 31 * result + indexes[i];
    }

    return 31 * result + length;
  }

  /**
   * Increments the value at the given index by the given value.
   *
   * @param index
   *          the index of the value to alter
   * @param val
   *          the value to be added to the index
   */
  @Override
  public void increment(final int index, final double val) {
    if (index > length - 1 || index < 0) {
      throw new IndexOutOfBoundsException("Can not access an index larger then the vector or a negative index");
    }
    if (val == 0) {// donst want to insert a zero, and a zero changes nothing
      return;
    }
    final int location = Arrays.binarySearch(indexes, 0, used, index);
    if (location < 0) {
      insertValue(location, index, val);
    } else {
      values[location] += val;
      if (values[location] == 0.0) {
        removeNonZero(location);
      }
    }
  }

  /**
   * Takes the negative insert location value returned by
   * {@link Arrays#binarySearch(int[], int, int, int) } and adjust the vector to
   * add the given value into this location. Should only be called with negative
   * input returned by said method. Should never be called for an index that in
   * fact does already exist in this sparce vector.
   *
   * @param insertLocation
   *          the negative insertion index such that -(insertLocation+1) is the
   *          address that the value should have
   * @param index
   *          the index that is being added
   * @param val
   *          the value that is being added for the given index
   */
  private void insertValue(int insertLocation, final int index, final double val) {
    insertLocation = -(insertLocation + 1);// Convert from negative value to the
                                           // location is should be placed, see
                                           // JavaDoc of binarySearch
    if (used == indexes.length) // Full, expand
    {
      final int newIndexesSize = Math.max(Math.min(indexes.length * 3 / 2, Integer.MAX_VALUE), 8);
      indexes = Arrays.copyOf(indexes, newIndexesSize);
      values = Arrays.copyOf(values, newIndexesSize);
    }

    if (insertLocation < used) // Instead of moving indexes over manualy, set it
                               // up to use a native System call to move things
                               // out of the way
    {
      System.arraycopy(indexes, insertLocation, indexes, insertLocation + 1, used - insertLocation);
      System.arraycopy(values, insertLocation, values, insertLocation + 1, used - insertLocation);
    }

    indexes[insertLocation] = index;
    values[insertLocation] = val;
    used++;
  }

  @Override
  public boolean isSparse() {
    return true;
  }

  @Override
  public double kurtosis() {
    final double mean = mean();

    double tmp = 0;
    double var = 0;

    for (int i = 0; i < used; i++) {
      tmp += pow(values[i] - mean, 4);
      var += pow(values[i] - mean, 2);
    }

    // All the zero's we arent storing
    tmp += pow(-mean, 4) * (length - used);
    var += pow(-mean, 2) * (length - used);

    tmp /= length;
    var /= length;

    return tmp / pow(var, 2) - 3;
  }

  @Override
  public int length() {
    return length;
  }

  @Override
  public double max() {
    if (maxCache != null) {
      return maxCache;
    }

    double result = 0;
    for (int i = 0; i < used; i++) {
      result = Math.max(result, values[i]);
    }

    return maxCache = result;
  }

  @Override
  public double median() {
    if (used < length / 2) {// more than half zeros, so 0 must be the median
      return 0.0;
    } else {
      return super.median();
    }
  }

  @Override
  public double min() {
    if (minCache != null) {
      return minCache;
    }
    double result = 0;
    for (int i = 0; i < used; i++) {
      result = Math.min(result, values[i]);
    }

    return minCache = result;
  }

  @Override
  public void multiply(final double c, final Matrix A, final Vec b) {
    if (length() != A.rows()) {
      throw new ArithmeticException("Vector x Matrix dimensions do not agree");
    } else if (b.length() != A.cols()) {
      throw new ArithmeticException("Destination vector is not the right size");
    }

    for (int i = 0; i < used; i++) {
      final double val = c * values[i];
      final int index = indexes[i];
      for (int j = 0; j < A.cols(); j++) {
        b.increment(j, val * A.get(index, j));
      }
    }
  }

  @Override
  public void mutableAdd(final double c) {
    if (c == 0.0) {
      return;
    }
    clearCaches();
    /*
     * This NOT the most efficient way to implement this. But adding a constant
     * to every value in a sparce vector defeats its purpos.
     */
    for (int i = 0; i < length(); i++) {
      set(i, get(i) + c);
    }
  }

  @Override
  public void mutableAdd(final double c, final Vec v) {
    clearCaches();
    if (c == 0.0) {
      return;
    }
    if (v instanceof SparseVector) {
      final SparseVector b = (SparseVector) v;
      int p1 = 0, p2 = 0;
      while (p1 < used && p2 < b.used) {
        final int a1 = indexes[p1], a2 = b.indexes[p2];
        if (a1 == a2) {
          values[p1] += c * b.values[p2];
          p1++;
          p2++;
        } else if (a1 > a2) {
          // 0 + some value is that value, set it
          set(a2, c * b.values[p2]);
          /*
           * p2 must be increment becase were moving to the next value
           * 
           * p1 must be be incremented becase a2 was less thenn the current
           * index. So the inseration occured before p1, so for indexes[p1] to
           * == a1, p1 must be incremented
           *
           */
          p1++;
          p2++;
        } else// a1 < a2, thats adding 0 to this vector, nothing to do.
        {
          p1++;
        }
      }

      // One of them is now empty.
      // If b is not empty, we must add b to this. If b is empty, we would be
      // adding zeros to this [so we do nothing]
      while (p2 < b.used) {
        set(b.indexes[p2], c * b.values[p2++]);// TODO Can be done more
                                               // efficently
      }
    } else if (v.isSparse()) {
      if (v.nnz() == 0) {
        return;
      }
      int p1 = 0;
      final Iterator<IndexValue> iter = v.getNonZeroIterator();
      IndexValue iv = iter.next();
      while (p1 < used && iv != null) {
        final int a1 = indexes[p1];
        final int a2 = iv.getIndex();

        if (a1 == a2) {
          values[p1++] += c * iv.getValue();
          if (iter.hasNext()) {
            iv = iter.next();
          } else {
            break;
          }
        } else if (a1 > a2) {
          set(a2, c * iv.getValue());
          p1++;
          if (iter.hasNext()) {
            iv = iter.next();
          } else {
            break;
          }
        } else {
          p1++;
        }
      }
    } else {
      // Else it is dense
      for (int i = 0; i < length(); i++) {
        set(i, get(i) + c * v.get(i));
      }
    }

  }

  @Override
  public void mutableDivide(final double c) {
    clearCaches();
    if (c == 0 && used != length) {
      throw new ArithmeticException("Division by zero would occur");
    }
    for (int i = 0; i < used; i++) {
      values[i] /= c;
    }
  }

  @Override
  public void mutableMultiply(final double c) {
    clearCaches();
    if (c == 0.0) {
      zeroOut();
      return;
    }

    for (int i = 0; i < used; i++) {
      values[i] *= c;
    }
  }

  @Override
  public void mutablePairwiseDivide(final Vec b) {
    if (length() != b.length()) {
      throw new ArithmeticException("Vectors must have the same length");
    }
    clearCaches();

    for (int i = 0; i < used; i++) {
      values[i] /= b.get(indexes[i]);// zeros stay zero
    }
  }

  @Override
  public void mutablePairwiseMultiply(final Vec b) {
    if (length() != b.length()) {
      throw new ArithmeticException("Vectors must have the same length");
    }
    clearCaches();

    for (int i = 0; i < used; i++) {
      values[i] *= b.get(indexes[i]);// zeros stay zero
    }
  }

  @Override
  public int nnz() {
    return used;
  }

  @Override
  public void normalize() {
    double sum = 0;

    for (int i = 0; i < used; i++) {
      sum += values[i] * values[i];
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
      for (int i = 0; i < used; i++) {
        result += abs(values[i]);
      }
    } else if (p == 2) {
      for (int i = 0; i < used; i++) {
        result += values[i] * values[i];
      }
      result = Math.sqrt(result);
    } else if (Double.isInfinite(p)) {
      for (int i = 0; i < used; i++) {
        result = Math.max(result, abs(values[i]));
      }
    } else {
      for (int i = 0; i < used; i++) {
        result += Math.pow(Math.abs(values[i]), p);
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

    if (y instanceof SparseVector) {
      int p1 = 0, p2 = 0;
      final SparseVector b = (SparseVector) y;

      while (p1 < used && p2 < b.used) {
        final int a1 = indexes[p1], a2 = b.indexes[p2];
        if (a1 == a2) {
          norm += Math.pow(Math.abs(values[p1] - b.values[p2]), p);
          p1++;
          p2++;
        } else if (a1 > a2) {
          norm += Math.pow(Math.abs(b.values[p2++]), p);
        } else {
          // a1 < a2, this vec has a value, other does not
          norm += Math.pow(Math.abs(values[p1++]), p);
        }
      }
      // One of them is now empty.
      // So just sum up the rest of the elements
      while (p1 < used) {
        norm += Math.pow(Math.abs(values[p1++]), p);
      }
      while (p2 < b.used) {
        norm += Math.pow(Math.abs(b.values[p2++]), p);
      }
    } else {
      int z = 0;
      for (int i = 0; i < length(); i++) {
        // Move through until we hit our next non zero element
        while (z < used && indexes[z] > i) {
          norm += Math.pow(Math.abs(-y.get(i++)), p);
        }

        // We made it! (or are at the end). Is our non zero value the same?
        if (z < used && indexes[z] == i) {
          norm += Math.pow(Math.abs(values[z++] - y.get(i)), p);
        } else {
          // either we used a non zero of this in the loop or we are out of them
          norm += Math.pow(Math.abs(-y.get(i)), p);
        }
      }
    }
    return Math.pow(norm, 1.0 / p);
  }

  /**
   * Removes a non zero value by shifting everything to the right over by one
   *
   * @param nzIndex
   *          the index to remove (setting it to zero)
   */
  private void removeNonZero(final int nzIndex) {
    for (int i = nzIndex + 1; i < used; i++) {
      values[i - 1] = values[i];
      indexes[i - 1] = indexes[i];
    }
    used--;
  }

  @Override
  public void set(final int index, final double val) {
    if (index > length() - 1 || index < 0) {
      throw new IndexOutOfBoundsException(index + " does not fit in [0," + length + ")");
    }

    clearCaches();
    final int insertLocation = Arrays.binarySearch(indexes, 0, used, index);
    if (insertLocation >= 0) {
      if (val != 0) {// set it
        values[insertLocation] = val;
      } else// shift used count and everyone over
      {
        removeNonZero(insertLocation);
      }
    } else if (val != 0) {// dont insert 0s, that is stupid
      insertValue(insertLocation, index, val);
    }
  }

  /**
   * Because sparce vectors do not have most value set, they can have their
   * length increased, and sometimes decreased, without any effort. The length
   * can always be extended. The length can be reduced down to the size of the
   * largest non zero element.
   *
   * @param length
   *          the new length of this vector
   */
  public void setLength(final int length) {
    if (used > 0 && length < indexes[used - 1]) {
      throw new RuntimeException("Can not set the length to a value less then an index already in use");
    }
    this.length = length;
  }

  @Override
  public double skewness() {
    final double mean = mean();

    double numer = 0, denom = 0;

    for (int i = 0; i < used; i++) {
      numer += pow(values[i] - mean, 3);
      denom += pow(values[i] - mean, 2);
    }

    // All the zero's we arent storing
    numer += pow(-mean, 3) * (length - used);
    denom += pow(-mean, 2) * (length - used);

    numer /= length;
    denom /= length;

    final double s1 = numer / pow(denom, 3.0 / 2.0);

    if (length >= 3) {// We can use the bias corrected formula
      return sqrt(length * (length - 1)) / (length - 2) * s1;
    }

    return s1;
  }

  @Override
  public Vec sortedCopy() {
    final IndexTable it = new IndexTable(DoubleList.unmodifiableView(values, used));

    final double[] newValues = new double[used];
    final int[] newIndecies = new int[used];

    int lessThanZero = 0;
    for (int i = 0; i < used; i++) {
      final int origIndex = it.index(i);
      newValues[i] = values[origIndex];
      if (newValues[i] < 0) {
        lessThanZero++;
      }
      newIndecies[i] = i;
    }
    // all < 0 values are right, now correct > 0 values
    for (int i = lessThanZero; i < used; i++) {
      newIndecies[i] = length - (used - lessThanZero) + i - lessThanZero;
    }

    final SparseVector sv = new SparseVector(length);
    sv.used = used;
    sv.values = newValues;
    sv.indexes = newIndecies;
    return sv;
  }

  @Override
  public double sum() {
    if (sumCache != null) {
      return sumCache;
    }

    /*
     * Uses Kahan summation algorithm, which is more accurate then naively
     * summing the values in floating point. Though it does not guarenty the
     * best possible accuracy
     *
     * See: http://en.wikipedia.org/wiki/Kahan_summation_algorithm
     */
    double sum = 0;
    double c = 0;
    for (int i = 0; i < used; i++) {
      final double d = values[i];
      final double y = d - c;
      final double t = sum + y;
      c = t - sum - y;
      sum = t;
    }

    return sumCache = sum;
  }

  @Override
  public String toString() {
    final StringBuilder sb = new StringBuilder("[");

    int p = 0;
    for (int i = 0; i < length(); i++) {
      if (i != 0) {
        sb.append(", ");
      }

      if (p < used && indexes[p] == i) {
        sb.append(values[p++]);
      } else {
        sb.append("0.0");
      }
    }
    sb.append("]");

    return sb.toString();
  }

  @Override
  public double variance() {
    if (varianceCache != null) {
      return varianceCache;
    }

    final double mu = mean();
    double tmp = 0;

    final double N = length();

    for (int i = 0; i < used; i++) {
      tmp += Math.pow(values[i] - mu, 2);
    }
    // Now add all the zeros into it
    tmp += (length() - used) * Math.pow(0 - mu, 2);
    tmp /= N;

    return varianceCache = tmp;
  }

  @Override
  public void zeroOut() {
    used = 0;
  }
}
