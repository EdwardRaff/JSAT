package jsat.regression.evaluation;

import jsat.utils.DoubleList;

/**
 * This abstract class provides the work for maintaining the history of
 * predictions and their true values.
 *
 * @author Edward Raff
 */
public abstract class TotalHistoryRegressionScore implements RegressionScore {

  private static final long serialVersionUID = -5262934560490160236L;
  /**
   * List of the true target values
   */
  protected DoubleList truths;
  /**
   * List of the predict values for each target
   */
  protected DoubleList predictions;
  /**
   * The weight of importance for each point
   */
  protected DoubleList weights;

  public TotalHistoryRegressionScore() {
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public TotalHistoryRegressionScore(final TotalHistoryRegressionScore toCopy) {
    if (toCopy.truths != null) {
      truths = new DoubleList(toCopy.truths);
      predictions = new DoubleList(toCopy.predictions);
      weights = new DoubleList(toCopy.weights);
    }
  }

  @Override
  public void addResult(final double prediction, final double trueValue, final double weight) {
    truths.add(trueValue);
    predictions.add(prediction);
    weights.add(weight);
  }

  @Override
  public void addResults(final RegressionScore other) {
    final TotalHistoryRegressionScore otherObj = (TotalHistoryRegressionScore) other;
    truths.addAll(otherObj.truths);
    predictions.addAll(otherObj.predictions);
    weights.addAll(otherObj.weights);
  }

  @Override
  public abstract TotalHistoryRegressionScore clone();

  @Override
  public void prepare() {
    truths = new DoubleList();
    predictions = new DoubleList();
    weights = new DoubleList();
  }

}
