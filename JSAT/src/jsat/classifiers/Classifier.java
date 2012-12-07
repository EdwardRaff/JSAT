
package jsat.classifiers;

import java.io.Serializable;
import java.util.concurrent.ExecutorService;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.ModelMismatchException;
import jsat.exceptions.UntrainedModelException;

/**
 * A Classifier is used to predict the target class of new unseen data points. 
 * 
 * @author Edward Raff
 */
public interface Classifier extends Cloneable, Serializable
{
    /**
     * Performs classification on the given data point. 
     * @param data the data point to classify
     * @return the results of the classification. 
     * @throws UntrainedModelException if the method is called before the model has been trained
     * @throws ModelMismatchException if the given data point is incompatible with the model
     */
    public CategoricalResults classify(DataPoint data);
    /**
     * Trains the classifier and constructs a model for classification using the 
     * given data set. If the training method knows how, it will used the 
     * <tt>threadPool</tt> to conduct training in parallel. This method will 
     * block until the training has completed.
     * 
     * @param dataSet the data set to train on
     * @param threadPool the source of threads to use. 
     * @throws FailedToFitException if the model is unable to be constructed for some reason
     */
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool);
    /**
     * Trains the classifier and constructs a model for classification using the 
     * given data set.
     * 
     * @param dataSet the data set to train on
     * @throws FailedToFitException if the model is unable to be constructed for some reason
     */
    public void trainC(ClassificationDataSet dataSet);
    
    /**
     * Indicates whether the model knows how to train using weighted data points. If it 
     * does, the model will train assuming the weights. The values returned by this 
     * method may change depending on the parameters set for the model. 
     * @return <tt>true</tt> if the model supports weighted data, <tt>false</tt> otherwise
     */
    public boolean supportsWeightedData();
    
    public Classifier clone();
}
