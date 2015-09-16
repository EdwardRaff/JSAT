package jsat.regression;

import java.util.Collections;
import java.util.concurrent.ExecutorService;

import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * A base implementation of the UpdateableRegressor. null
 * {@link #train(jsat.regression.RegressionDataSet, java.util.concurrent.ExecutorService) }
 * will simply call {@link #train(jsat.regression.RegressionDataSet) }, which
 * will call null {@link #setUp(jsat.classifiers.CategoricalData[], int) } and
 * then call {@link #update(jsat.classifiers.DataPoint, double) } for each data
 * point in a random order.
 *
 * @author Edward Raff
 */
public abstract class BaseUpdateableRegressor implements UpdateableRegressor {

  private static final long serialVersionUID = -679467882721432240L;

  /**
   * Performs training on an updateable classifier by going over the whole data
   * set in random order one observation at a time, multiple times.
   *
   * @param dataSet
   *          the data set to train from
   * @param toTrain
   *          the classifier to train
   * @param epochs
   *          the number of passes through the data set
   */
  public static void trainEpochs(final RegressionDataSet dataSet, final UpdateableRegressor toTrain, final int epochs) {
    if (epochs < 1) {
      throw new IllegalArgumentException("epochs must be positive");
    }
    toTrain.setUp(dataSet.getCategories(), dataSet.getNumNumericalVars());
    final IntList randomOrder = new IntList(dataSet.getSampleSize());
    ListUtils.addRange(randomOrder, 0, dataSet.getSampleSize(), 1);
    for (int epoch = 0; epoch < epochs; epoch++) {
      Collections.shuffle(randomOrder);
      for (final int i : randomOrder) {
        toTrain.update(dataSet.getDataPoint(i), dataSet.getTargetValue(i));
      }
    }
  }

  private int epochs = 1;

  @Override
  abstract public UpdateableRegressor clone();

  /**
   * Returns the number of epochs used for training
   *
   * @return the number of epochs used for training
   */
  public int getEpochs() {
    return epochs;
  }

  /**
   * Sets the number of whole iterations through the training set that will be
   * performed for training
   *
   * @param epochs
   *          the number of whole iterations through the data set
   */
  public void setEpochs(final int epochs) {
    if (epochs < 1) {
      throw new IllegalArgumentException("epochs must be a positive value");
    }
    this.epochs = epochs;
  }

  @Override
  public void train(final RegressionDataSet dataSet) {
    trainEpochs(dataSet, this, epochs);
  }

  @Override
  public void train(final RegressionDataSet dataSet, final ExecutorService threadPool) {
    train(dataSet);
  }

}
