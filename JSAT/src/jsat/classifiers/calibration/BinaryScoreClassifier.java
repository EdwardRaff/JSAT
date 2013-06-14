
package jsat.classifiers.calibration;

import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;

/**
 * Many algorithms linear a binary separation between two classes <i>A</i> and 
 * <i>B</i> by representing the target labels with a {@code -1} ad {@code 1}. At
 * prediction, the output is a real valued number - where the sign indicates the
 * class label. This interface indicates that an algorithm conforms such 
 * behavior, and that the "0" class corresponds to the {@code -1} label, and the
 * "1" class corresponds to the {@code 1} label. <br>
 * 
 * @author Edward Raff
 */
public interface BinaryScoreClassifier extends Classifier
{
    /**
     * Returns the numeric score for predicting a class of a given data point,
     * where the sign of the value indicates which class the data point is 
     * predicted to belong to. 
     * 
     * @param dp the data point to predict the class label of
     * @return the score for the given data point
     */
    public double getScore(DataPoint dp);
    
    @Override
    public BinaryScoreClassifier clone();
}
