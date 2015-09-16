package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.OnLineStatistics;

/**
 * This class transforms all numerical values into a specified range by a linear
 * scaling of all the data point values.
 *
 * @author Edward Raff
 */
public class LinearTransform implements InPlaceInvertibleTransform {

  /**
   * Factory for creating new {@link LinearTransform} transforms.
   */
  static public class LinearTransformFactory implements DataTransformFactory {

    private final Double A;
    private final Double B;

    /**
     * Creates a new Linear Transform factory for the range [0, 1]
     */
    public LinearTransformFactory() {
      this(0, 1);
    }

    /**
     * Creates a new Linear Transform factory
     *
     * @param A
     *          the maximum value for the transformed data set
     * @param B
     *          the minimum value for the transformed data set
     */
    public LinearTransformFactory(final double A, final double B) {
      this.A = A;
      this.B = B;
    }

    /**
     * Copy constructor
     *
     * @param toCopy
     *          the object to copy
     */
    public LinearTransformFactory(final LinearTransformFactory toCopy) {
      this(toCopy.A, toCopy.B);
    }

    @Override
    public LinearTransformFactory clone() {
      return new LinearTransformFactory(this);
    }

    @Override
    public DataTransform getTransform(final DataSet dataset) {
      return new LinearTransform(dataset, A, B);
    }
  }

  private static final long serialVersionUID = 5580283565080452022L;
  /**
   * The max value
   */
  private final double A;

  /**
   * The min value
   */
  private final double B;

  /**
   * The minimum observed value for each attribute
   */
  private Vec mins;

  /**
   * Represents
   *
   * A - B ----------- max - min
   */
  private Vec mutliplyConstants;

  /**
   * Creates a new Linear Transformation for the input data set so that all
   * values are in the [0, 1] range.
   *
   * @param dataSet
   *          the data set to learn the transform from
   */
  public LinearTransform(final DataSet dataSet) {
    this(dataSet, 1, 0);
  }

  /**
   * Creates a new Linear Transformation for the input data set.
   *
   * @param dataSet
   *          the data set to learn the transform from
   * @param A
   *          the maximum value for the transformed data set
   * @param B
   *          the minimum value for the transformed data set
   */
  public LinearTransform(final DataSet dataSet, double A, double B) {
    if (A == B) {
      throw new RuntimeException("Values must be different");
    } else if (B > A) {
      final double tmp = A;
      A = B;
      B = tmp;
    }
    this.A = A;
    this.B = B;

    mins = new DenseVector(dataSet.getNumNumericalVars());
    final Vec maxs = new DenseVector(mins.length());
    mutliplyConstants = new DenseVector(mins.length());

    final OnLineStatistics[] stats = dataSet.getOnlineColumnStats(false);

    for (int i = 0; i < mins.length(); i++) {
      final double min = stats[i].getMin();
      final double max = stats[i].getMax();
      if (max - min < 1e-6) // No change
      {
        mins.set(i, 0);
        maxs.set(i, 1);
        mutliplyConstants.set(i, 1.0);
      } else {
        mins.set(i, min);
        maxs.set(i, max);
        mutliplyConstants.set(i, A - B);
      }
    }

    /**
     * Now we set up the vectors to perform transformations
     *
     * if x := the variable to be transformed to the range [A, B] Then the
     * transformation we want is
     *
     * (A - B) B + --------- * (-min+x) max - min
     *
     * This middle constant will be placed in "maxs"
     *
     */
    maxs.mutableSubtract(mins);
    mutliplyConstants.mutablePairwiseDivide(maxs);

  }

  /**
   * Copy constructor
   *
   * @param other
   *          the transform to copy
   */
  private LinearTransform(final LinearTransform other) {
    A = other.A;
    B = other.B;
    if (other.mins != null) {
      mins = other.mins.clone();
    }
    if (other.mutliplyConstants != null) {
      mutliplyConstants = other.mutliplyConstants.clone();
    }
  }

  @Override
  public LinearTransform clone() {
    return new LinearTransform(this);
  }

  @Override
  public DataPoint inverse(final DataPoint dp) {
    final DataPoint toRet = dp.clone();
    mutableInverse(toRet);
    return toRet;
  }

  @Override
  public void mutableInverse(final DataPoint dp) {
    final Vec v = dp.getNumericalValues();
    v.mutableSubtract(B);
    v.mutablePairwiseDivide(mutliplyConstants);
    v.mutableAdd(mins);
  }

  @Override
  public void mutableTransform(final DataPoint dp) {
    final Vec v = dp.getNumericalValues();
    v.mutableSubtract(mins);
    v.mutablePairwiseMultiply(mutliplyConstants);
    v.mutableAdd(B);
  }

  @Override
  public boolean mutatesNominal() {
    return false;
  }

  @Override
  public DataPoint transform(final DataPoint dp) {
    final DataPoint toRet = dp.clone();
    mutableTransform(toRet);
    return toRet;
  }
}
