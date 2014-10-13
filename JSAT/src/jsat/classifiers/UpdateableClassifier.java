package jsat.classifiers;

import jsat.exceptions.UntrainedModelException;

/**
 * UpdateableClassifier is an interface for one type of Online learner. The main
 * characteristic of an online learning is that new example points can be added 
 * incremental after the classifier was initially trained, or as part of its 
 * initial training. <br>
 * Some Online learners behave differently in when they are updated. The 
 * UpdateableClassifier is an online learner that specifically only performs 
 * additional learning when a new example is provided via the 
 * {@link #update(jsat.classifiers.DataPoint, int) } method. 
 * <br>
 * The standard behavior for an Updateable Classifier is that the user first 
 * calls {@link #trainC(jsat.classifiers.ClassificationDataSet) } to first train
 * the classifier, or {@link #setUp(jsat.classifiers.CategoricalData[], int, 
 * jsat.classifiers.CategoricalData) } to prepare for online updates. Once one 
 * of these is called, it should then be safe to call 
 * {@link #update(jsat.classifiers.DataPoint, int) } without getting a 
 * {@link UntrainedModelException}. Some online learners may require one of the 
 * train methods to be called first. 
 * 
 * @author Edward Raff
 */
public interface UpdateableClassifier extends Classifier
{
    /**
     * Prepares the classifier to begin learning from its 
     * {@link #update(jsat.classifiers.DataPoint, int) } method. 
     * 
     * @param categoricalAttributes an array containing the categorical 
     * attributes that will be in each data point
     * @param numericAttributes the number of numeric attributes that will be in
     * each data point
     * @param predicting the information for the target class that will be
     * predicted
     */
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting);
    
    /**
     * Updates the classifier by giving it a new data point to learn from. 
     * @param dataPoint the data point to learn
     * @param targetClass the target class of the data point
     */
    public void update(DataPoint dataPoint, int targetClass);

    @Override
    public UpdateableClassifier clone();
}
