package jsat.regression.evaluation;

import java.io.Serializable;

/**
 * This interface defines the contract for evaluating or "scoring" the results 
 * on a regression problem. <br>
 * <br>
 * All regression scores must override the {@link #equals(java.lang.Object)}
 * and {@link #hashCode() } methods. If a score has parameters, different 
 * objects with different parameters must not be equal. However, different 
 * objects with the same parameters must be equal <i>even if their internal 
 * states are different</i>
 * 
 * @author Edward Raff
 */
public interface RegressionScore extends Serializable
{
    public void prepare();
    
    /**
     * Adds the given result to the score
     * @param prediction the prediction for the data point 
     * @param trueValue the true value for the data point
     * @param weight the weigh to assign to the data point
     */
    public void addResult(double prediction, double trueValue, double weight);
    
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
    public void addResults(RegressionScore other);
    
    /**
     * Computes the score for the results that have been enrolled via 
     * {@link #addResult(double, double, double)  } 
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
    
    public RegressionScore clone();
    
    /**
     * Returns the name to present for this score
     * @return the score name
     */
    public String getName();
}
