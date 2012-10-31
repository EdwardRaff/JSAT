
package jsat.classifiers;

import java.util.Collections;
import java.util.concurrent.ExecutorService;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * A base implementation of the UpdateableClassifier. 
 * {@link #trainC(jsat.classifiers.ClassificationDataSet, 
 * java.util.concurrent.ExecutorService) } will simply call 
 * {@link #trainC(jsat.classifiers.ClassificationDataSet) }, which will call 
 * {@link #setUp(jsat.classifiers.CategoricalData[], int, 
 * jsat.classifiers.CategoricalData) } and then call 
 * {@link #update(jsat.classifiers.DataPoint, int) } for each data point in a 
 * random order. 
 * 
 * @author Edward Raff
 */
public abstract class BaseUpdateableClassifier implements UpdateableClassifier
{

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        setUp(dataSet.getCategories(), dataSet.getNumNumericalVars(), 
                dataSet.getPredicting());
        IntList randomOrder = new IntList(dataSet.getSampleSize());
        ListUtils.addRange(randomOrder, 0, dataSet.getSampleSize(), 1);
        Collections.shuffle(randomOrder);
        for(int i : randomOrder)
            update(dataSet.getDataPoint(i), dataSet.getDataPointCategory(i));
    }

    @Override
    abstract public Classifier clone();
    
}
