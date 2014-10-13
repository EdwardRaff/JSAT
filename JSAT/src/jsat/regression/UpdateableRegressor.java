package jsat.regression;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.exceptions.FailedToFitException;

/**
 * UpdateableRegressor is an interface for one type of Online learner. The main
 * characteristic of an online learner is that new example points can be added 
 * incrementally after the classifier was initially trained, or as part of its 
 * initial training. <br>
 * Some Online learners behave differently in when they are updated. The 
 * UpdateableRegressor is an online learner that specifically only performs 
 * additional learning when a new example is provided via the 
 * {@link #update(jsat.classifiers.DataPoint, double)  } method. 
 * <br>
 * The standard behavior for an UpdateableRegressor is that the user first 
 * calls {@link #train(jsat.regression.RegressionDataSet)  } to first train
 * the classifier, or {@link #setUp(jsat.classifiers.CategoricalData[], int)  }
 * to prepare for online updates. Once one 
 * of these is called, it should then be safe to call 
 * {@link #update(jsat.classifiers.DataPoint, double)  } without getting a 
 * {@link FailedToFitException}. Some online learners may require one of the 
 * train methods to be called first. 
 * 
 * @author Edward Raff
 */
public interface UpdateableRegressor extends Regressor
{
    /**
     * Prepares the classifier to begin learning from its 
     * {@link #update(jsat.classifiers.DataPoint, double)  } method. 
     * 
     * @param categoricalAttributes an array containing the categorical 
     * attributes that will be in each data point
     * @param numericAttributes the number of numeric attributes that will be in
     * each data point
     */
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes);
    
    /**
     * Updates the classifier by giving it a new data point to learn from. 
     * @param dataPoint the data point to learn
     * @param targetValue the target value of the data point
     */
    public void update(DataPoint dataPoint, double targetValue);

    @Override
    public UpdateableRegressor clone();
}
