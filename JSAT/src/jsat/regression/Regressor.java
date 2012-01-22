
package jsat.regression;

import java.util.concurrent.ExecutorService;
import jsat.classifiers.DataPoint;

/**
 *
 * @author Edward Raff
 */
public interface Regressor extends Cloneable
{
    public double regress(DataPoint data);
    
    public void train(RegressionDataSet dataSet, ExecutorService threadPool);
    public void train(RegressionDataSet dataSet);
    
    public boolean supportsWeightedData();
    
    public Regressor clone();
}
