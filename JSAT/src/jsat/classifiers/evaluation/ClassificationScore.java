package jsat.classifiers.evaluation;

import java.io.Serializable;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;

/**
 * This interface defines the contract for evaluating or "scoring" the results 
 * on a classification problem. <br>
 * <br>
 * All classification scores must override the {@link #equals(java.lang.Object)}
 * and {@link #hashCode() } methods. If a score has parameters, different 
 * objects with different parameters must not be equal. However, different 
 * objects with the same parameters must be equal <i>even if their internal 
 * states are different</i>
 * 
 * @author Edward Raff
 */
public interface ClassificationScore extends Serializable
{
    /**
     * Prepares this score to predict on the given input
     * @param toPredict the class label information that will be evaluated
     */
    public void prepare(CategoricalData toPredict);
    
    /**
     * Adds the given result to the score
     * @param prediction the prediction for the data point 
     * @param trueLabel the true label for the data point
     * @param weight the weigh to assign to the data point
     */
    public void addResult(CategoricalResults prediction, int trueLabel, double weight);
    
    /**
     * The score contained in <i>this</i> object is augmented with the results 
     * already accumulated in the {@code other} object. This does not result in 
     * an averaging, but alters the current object to have the same score it 
     * would have had if all the results were originally inserted into <i>this
     * </i> object. <br>
     * <br>
     * This method is only required to work if {@code other} if of the same 
     * class as {@code this} object. 
     * 
     * @param other the object to add the results from
     */
    public void addResults(ClassificationScore other);
    
    /**
     * Computes the score for the results that have been enrolled via 
     * {@link #addResult(jsat.classifiers.CategoricalResults, int, double) } 
     * 
     * @return the score for the current results
     */
    public double getScore();
    
    /**
     * Returns {@code true} if a lower score is better, or {@code false} if a
     * higher score is better
     * @return {@code true} if a lower score is better
     */
    public boolean lowerIsBetter();
    
    @Override
    public boolean equals(Object obj);

    @Override
    public int hashCode();
    
    public ClassificationScore clone();
    
    /**
     * Returns the name to present for this score
     * @return the score name
     */
    public String getName();
}
