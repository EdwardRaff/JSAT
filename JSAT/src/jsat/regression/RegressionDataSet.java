package jsat.regression;

import java.util.ArrayList;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * A RegressionDataSet is a data set specifically for the task of performing
 * regression. Each data point is paired with s double value that indicates its
 * true regression value. An example of a regression problem would be mapping
 * the inputs of a function to its outputs, and attempting to learn the function
 * from the samples.
 *
 * @author Edward Raff
 */
public class RegressionDataSet extends DataSet<RegressionDataSet> {

  private static final int[] emptyInt = new int[0];

  public static RegressionDataSet comineAllBut(final List<RegressionDataSet> list, final int exception) {
    final int numer = list.get(exception).getNumNumericalVars();
    final CategoricalData[] categories = list.get(exception).getCategories();

    final RegressionDataSet rds = new RegressionDataSet(numer, categories);

    // The list of data sets
    for (int i = 0; i < list.size(); i++) {
      if (i == exception) {
      } else {
        rds.dataPoints.addAll(list.get(i).dataPoints);
      }
    }

    return rds;
  }

  /**
   * Creates a new data set that uses the given list as its backing list. No
   * copying is done, and changes to this list will be reflected in this data
   * set, and the other way.
   *
   * @param list
   *          the list of datapoint to back a new data set with
   * @return a new data set
   */
  public static RegressionDataSet usingDPPList(final List<DataPointPair<Double>> list) {
    final RegressionDataSet rds = new RegressionDataSet();
    rds.dataPoints = list;
    rds.numNumerVals = list.get(0).getDataPoint().numNumericalValues();
    rds.numericalVariableNames = new ArrayList<String>(rds.getNumNumericalVars());
    for (int i = 0; i < rds.getNumNumericalVars(); i++) {
      rds.numericalVariableNames.add("Numeric Input " + (i + 1));
    }
    rds.categories = CategoricalData.copyOf(list.get(0).getDataPoint().getCategoricalData());
    return rds;
  }

  /**
   * The list of all data points, paired with their true regression output
   */
  protected List<DataPointPair<Double>> dataPoints;

  private RegressionDataSet() {

  }

  /**
   * Creates a new empty data set for regression
   *
   * @param numerical
   *          the number of numerical attributes that will be used, excluding
   *          the regression value
   * @param categories
   *          an array of length equal to the number of categorical attributes,
   *          each object describing the attribute in question
   */
  public RegressionDataSet(final int numerical, final CategoricalData[] categories) {
    numNumerVals = numerical;
    this.categories = categories;
    dataPoints = new ArrayList<DataPointPair<Double>>();
    numericalVariableNames = new ArrayList<String>(getNumNumericalVars());
    setUpGenericNumericNames();
  }

  /**
   * Creates a new data set for the given list of data points. The data points
   * will be copied, changes in one will not effect the other.
   *
   * @param data
   *          the list of data point to create a data set from
   * @param predicting
   *          which of the numerical attributes is the regression target.
   *          Categorical attributes are ignored in the count of attributes for
   *          this value.
   */
  public RegressionDataSet(final List<DataPoint> data, final int predicting) {
    // Use the first data point to set up
    final DataPoint tmp = data.get(0);
    categories = new CategoricalData[tmp.numCategoricalValues()];
    System.arraycopy(tmp.getCategoricalData(), 0, categories, 0, categories.length);

    numNumerVals = tmp.numNumericalValues() - 1;

    dataPoints = new ArrayList<DataPointPair<Double>>(data.size());

    // Fill up data
    for (final DataPoint dp : data) {
      final DenseVector newVec = new DenseVector(numNumerVals);
      final Vec origVec = dp.getNumericalValues();
      // Set up the new vector
      for (int i = 0; i < origVec.length() - 1; i++) {
        if (i >= predicting) {
          newVec.set(i, origVec.get(i + 1));
        } else {
          newVec.set(i, origVec.get(i));
        }
      }

      final DataPoint newDp = new DataPoint(newVec, dp.getCategoricalValues(), categories);
      final DataPointPair<Double> dpp = new DataPointPair<Double>(newDp, origVec.get(predicting));

      dataPoints.add(dpp);
    }

    numericalVariableNames = new ArrayList<String>(getNumNumericalVars());
    setUpGenericNumericNames();
  }

  /**
   * Creates a new regression data set by copying all the data points in the
   * given list. Alterations to this list will not effect this DataSet.
   *
   * @param list
   *          source of data points to copy
   */
  public RegressionDataSet(final List<DataPointPair<Double>> list) {
    numNumerVals = list.get(0).getDataPoint().numNumericalValues();
    numericalVariableNames = new ArrayList<String>(getNumNumericalVars());
    setUpGenericNumericNames();
    categories = CategoricalData.copyOf(list.get(0).getDataPoint().getCategoricalData());
    dataPoints = new ArrayList<DataPointPair<Double>>(list.size());
    for (final DataPointPair<Double> dpp : list) {
      dataPoints.add(new DataPointPair<Double>(dpp.getDataPoint().clone(), dpp.getPair()));
    }
  }

  public void addDataPoint(final DataPoint dp, final double val) {
    if (dp.numNumericalValues() != getNumNumericalVars() || dp.numCategoricalValues() != getNumCategoricalVars()) {
      throw new RuntimeException(
          "The added data point does not match the number of values and categories for the data set");
    } else if (Double.isInfinite(val) || Double.isNaN(val)) {
      throw new ArithmeticException("Unregressiable value " + val + " given for regression");
    }

    final DataPointPair<Double> dpp = new DataPointPair<Double>(dp, val);
    dataPoints.add(dpp);
    columnVecCache.clear();
  }

  /**
   * Creates a new data point with no categorical variables to be added to the
   * data set. The arguments will be used directly, modifying them after will
   * effect the data set.
   *
   * @param numerical
   *          the numerical values for the data point
   * @throws IllegalArgumentException
   *           if the given values are inconsistent with the data this class
   *           stores.
   */
  public void addDataPoint(final Vec numerical, final double val) {
    addDataPoint(numerical, emptyInt, val);
  }

  /**
   * Creates a new data point to be added to the data set. The arguments will be
   * used directly, modifying them after will effect the data set.
   *
   * @param numerical
   *          the numerical values for the data point
   * @param categories
   *          the categorical values for the data point
   * @param val
   *          the target value to predict
   * @throws IllegalArgumentException
   *           if the given values are inconsistent with the data this class
   *           stores.
   */
  public void addDataPoint(final Vec numerical, final int[] categories, final double val) {
    if (numerical.length() != numNumerVals) {
      throw new RuntimeException("Data point does not contain enough numerical data points");
    }
    if (categories.length != categories.length) {
      throw new RuntimeException("Data point does not contain enough categorical data points");
    }

    for (int i = 0; i < categories.length; i++) {
      if (!this.categories[i].isValidCategory(categories[i])) {
        throw new RuntimeException("Categoriy value given is invalid");
      }
    }

    final DataPoint dp = new DataPoint(numerical, categories, this.categories);
    addDataPoint(dp, val);
  }

  public void addDataPointPair(final DataPointPair<Double> pair) {
    dataPoints.add(pair);
    columnVecCache.clear();
  }

  /**
   * Returns a new list containing copies of the data points in this data set,
   * paired with their regression target values. MModifications to the list or
   * data points will not effect this data set
   *
   * @return a list of copies of the data points in this set
   */
  public List<DataPointPair<Double>> getAsDPPList() {
    final ArrayList<DataPointPair<Double>> list = new ArrayList<DataPointPair<Double>>(dataPoints.size());
    for (final DataPointPair<Double> dpp : dataPoints) {
      list.add(new DataPointPair<Double>(dpp.getDataPoint().clone(), dpp.getPair()));
    }
    return list;
  }

  @Override
  public DataPoint getDataPoint(final int i) {
    return dataPoints.get(i).getDataPoint();
  }

  /**
   * Returns the i'th data point in the data set paired with its target
   * regressor value. Modifying the DataPointPair will effect the data set.
   *
   * @param i
   *          the index of the data point to obtain
   * @return the i'th DataPOintPair
   */
  public DataPointPair<Double> getDataPointPair(final int i) {
    return dataPoints.get(i);
  }

  /**
   * Returns a new list containing the data points in this data set, paired with
   * their regression target values. Modifications to the list will not effect
   * the data set, but modifying the points will. For a copy of the points, use
   * the {@link #getAsDPPList() } method.
   *
   * @return a list of the data points in this set
   */
  public List<DataPointPair<Double>> getDPPList() {
    final ArrayList<DataPointPair<Double>> list = new ArrayList<DataPointPair<Double>>(dataPoints);

    return list;
  }

  @Override
  public int getSampleSize() {
    return dataPoints.size();
  }

  @Override
  protected RegressionDataSet getSubset(final List<Integer> indicies) {
    final RegressionDataSet newData = new RegressionDataSet(numNumerVals, categories);
    for (final int i : indicies) {
      newData.addDataPoint(getDataPoint(i), getTargetValue(i));
    }
    return newData;
  }

  /**
   * Returns the target regression value for the <tt>i</tt>'th data point in the
   * data set.
   *
   * @param i
   *          the data point to get the regression value of
   * @return the target regression value
   */
  public double getTargetValue(final int i) {
    return dataPoints.get(i).getPair();
  }

  /**
   * Returns a vector containing the target regression values for each data
   * point. The vector is a copy, and modifications to it will not effect the
   * data set.
   *
   * @return a vector containing the target values for each data point
   */
  public Vec getTargetValues() {
    final DenseVector vals = new DenseVector(getSampleSize());

    for (int i = 0; i < getSampleSize(); i++) {
      vals.set(i, dataPoints.get(i).getPair());
    }

    return vals;
  }

  @Override
  public RegressionDataSet getTwiceShallowClone() {
    return (RegressionDataSet) super.getTwiceShallowClone(); // To change body
                                                             // of generated
                                                             // methods, choose
                                                             // Tools |
                                                             // Templates.
  }

  @Override
  public void setDataPoint(final int i, final DataPoint dp) {
    dataPoints.get(i).setDataPoint(dp);
    columnVecCache.clear();
  }

  /**
   * Sets the target regression value associated with a given data point
   *
   * @param i
   *          the index in the data set
   * @param val
   *          the new target value
   * @throws ArithmeticException
   *           if <tt>val</tt> is infinite or NaN
   */
  public void setTargetValue(final int i, final double val) {
    if (Double.isInfinite(val) || Double.isNaN(val)) {
      throw new ArithmeticException("Can not predict a " + val + " value");
    }
    dataPoints.get(i).setPair(val);
  }

  /**
   * Sets all the names of the numeric variables
   */
  private void setUpGenericNumericNames() {
    if (getNumNumericalVars() > 100) {
      return;
    }
    for (int i = 0; i < getNumNumericalVars(); i++) {
      numericalVariableNames.add("Numeric Input " + (i + 1));
    }
  }

  @Override
  public RegressionDataSet shallowClone() {
    final RegressionDataSet clone = new RegressionDataSet(numNumerVals, categories);
    for (final DataPointPair<Double> dpp : dataPoints) {
      clone.dataPoints.add(new DataPointPair<Double>(dpp.getDataPoint(), dpp.getPair()));
    }
    clone.columnVecCache.putAll(columnVecCache);
    return clone;
  }
}
