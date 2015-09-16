package jsat.regression;

import java.io.Serializable;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.DataPoint;

/**
 *
 * @author Edward Raff
 */
public interface Regressor extends Cloneable, Serializable {

  public Regressor clone();

  public double regress(DataPoint data);

  public boolean supportsWeightedData();

  public void train(RegressionDataSet dataSet);

  public void train(RegressionDataSet dataSet, ExecutorService threadPool);
}
