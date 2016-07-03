
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

    private static final long serialVersionUID = 3138493999362400767L;
    private int epochs = 1;

    /**
     * Default constructor that does nothing
     */
    public BaseUpdateableClassifier()
    {
    }

    /**
     * Copy constructor
     * @param toCopy object to copy
     */
    public BaseUpdateableClassifier(BaseUpdateableClassifier toCopy)
    {
        this.epochs = toCopy.epochs;
    }

    /**
     * Sets the number of whole iterations through the training set that will be
     * performed for training
     * @param epochs the number of whole iterations through the data set
     */
    public void setEpochs(int epochs)
    {
        if(epochs < 1)
            throw new IllegalArgumentException("epochs must be a positive value");
        this.epochs = epochs;
    }

    /**
     * Returns the number of epochs used for training
     * @return the number of epochs used for training
     */
    public int getEpochs()
    {
        return epochs;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainEpochs(dataSet, this, epochs);
    }
    
    /**
     * Performs training on an updateable classifier by going over the whole
     * data set in random order one observation at a time, multiple times. 
     *
     * @param dataSet the data set to train from
     * @param toTrain the classifier to train
     * @param epochs the number of passes through the data set
     */
    public static void trainEpochs(ClassificationDataSet dataSet, UpdateableClassifier toTrain, int epochs)
    {
        if(epochs < 1)
            throw new IllegalArgumentException("epochs must be positive");
        toTrain.setUp(dataSet.getCategories(), dataSet.getNumNumericalVars(), 
                dataSet.getPredicting());
        IntList randomOrder = new IntList(dataSet.getSampleSize());
        ListUtils.addRange(randomOrder, 0, dataSet.getSampleSize(), 1);
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Collections.shuffle(randomOrder);
            for (int i : randomOrder)
                toTrain.update(dataSet.getDataPoint(i), dataSet.getDataPointCategory(i));
        }
    }

    @Override
    abstract public UpdateableClassifier clone();
    
}
