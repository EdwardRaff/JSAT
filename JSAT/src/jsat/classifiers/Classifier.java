
package jsat.classifiers;

import java.util.concurrent.ExecutorService;

/**
 *
 * @author Edward Raff
 */
public interface Classifier 
{
    public CategoricalResults classify(DataPoint data);
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool);
    public void trainC(ClassificationDataSet dataSet);
}
