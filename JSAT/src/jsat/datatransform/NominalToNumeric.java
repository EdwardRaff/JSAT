package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.SparseVector;
import jsat.linear.Vec;

/**
 * This transform converts nominal feature values to numeric ones be adding a
 * new numeric feature for each possible categorical value for each nominal
 * feature. The numeric features will be all zeros, with only a single numeric
 * feature having a value of "1.0" for each nominal variable.
 *
 * @author Edward Raff
 */
public class NominalToNumeric implements DataTransform {

  /**
   * Factory for creating {@link NominalToNumeric} transforms
   */
  static public class NominalToNumericTransformFactory implements DataTransformFactory {

    public NominalToNumericTransformFactory() {
    }

    @Override
    public NominalToNumericTransformFactory clone() {
      return new NominalToNumericTransformFactory();
    }

    @Override
    public DataTransform getTransform(final DataSet dataset) {
      return new NominalToNumeric(dataset);
    }
  }

  private static final long serialVersionUID = -7765605678836464143L;
  private final int origNumericalCount;
  private final CategoricalData[] categoricalData;

  private int addedNumers;

  public NominalToNumeric(final DataSet dataSet) {
    this(dataSet.getNumNumericalVars(), dataSet.getCategories());
  }

  public NominalToNumeric(final int origNumericalCount, final CategoricalData[] categoricalData) {
    this.origNumericalCount = origNumericalCount;
    this.categoricalData = categoricalData;
    addedNumers = 0;

    for (final CategoricalData cd : categoricalData) {
      addedNumers += cd.getNumOfCategories();
    }
  }

  @Override
  public NominalToNumeric clone() {
    return new NominalToNumeric(origNumericalCount, categoricalData);
  }

  @Override
  public DataPoint transform(final DataPoint dp) {
    Vec v;

    // TODO we should detect if there are going to be so many sparce spaces
    // added by the categorical data that we should just choose a sparce vector
    // anyway
    if (dp.getNumericalValues().isSparse()) {
      v = new SparseVector(origNumericalCount + addedNumers);
    } else {
      v = new DenseVector(origNumericalCount + addedNumers);
    }

    final Vec oldV = dp.getNumericalValues();
    int i = 0;
    for (i = 0; i < origNumericalCount; i++) {
      v.set(i, oldV.get(i));
    }
    for (int j = 0; j < categoricalData.length; j++) {
      v.set(i + dp.getCategoricalValue(j), 1.0);
      i += categoricalData[j].getNumOfCategories();
    }

    return new DataPoint(v, new int[0], new CategoricalData[0]);
  }
}
