package jsat.linear.distancemetrics;

import jsat.linear.Vec;

/**
 * Many algorithms require computing the distances from a small set of points to
 * many other points. In these scenarios, if the small set of points contain 
 * dense vectors - and the large set contain sparse vectors, a large amount of 
 * unnecessary computation may be done. A {@link DistanceMetric} that implements 
 * this interface indicates that it supports more efficient computation of the 
 * distances in these scenarios. <br>
 * A distance metric that can efficiently handle dense to sparse distance 
 * computations has no reason to implement this interface. 
 * 
 * @author Edward Raff
 */
public interface DenseSparseMetric extends DistanceMetric
{
    /**
     * Computes a summary constant value for the vector that is based on the 
     * distance metric in use. This value will be used to perform efficient 
     * dense to sparse computations.
     * 
     * @param vec the vector that will be used in many distance computations
     * @return the summary value for the vector
     */
    public double getVectorConstant(Vec vec);
    
    /**
     * Efficiently computes the distance from one main vector that is used many 
     * times, to some sparse target vector. If the target vector dose not return 
     * true for {@link Vec#isSparse() }, the distance will be calculated using 
     * {@link #dist(jsat.linear.Vec, jsat.linear.Vec) } instead. 
     * 
     * @param summaryConst the summary constant for the main vector obtained 
     * with {@link #getVectorConstant(jsat.linear.Vec) }
     * @param main the main vector the summary constant is for
     * @param target the target vector to compute the distance to
     * @return the distance between the two vectors dist(main, target)
     */
    public double dist(double summaryConst, Vec main, Vec target);
}
