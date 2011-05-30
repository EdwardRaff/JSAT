
package jsat.classifiers;

/**
 *
 * @author Edward Raff
 */
public interface Classifier 
{
    public CategoricalResults classify(DataPoint data);
    public void trainC(ClassificationDataSet dataSet);
}
