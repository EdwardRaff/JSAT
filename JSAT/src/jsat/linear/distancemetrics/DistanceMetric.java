
package jsat.linear.distancemetrics;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public interface DistanceMetric
{
    /**
     * Computes the distance between 2 vectors. 
     * The smaller the value, the closer, and there for,
     * more similar, the vectors are. 0 indicates the vectors are the same.
     * 
     * @param a the first vector
     * @param b the second vector
     * @return the distance between them
     */
    public double dist(Vec a, Vec b);
    
    /**
     * Returns true if this distance metric obeys the rule that, for any x, y, and z &isin; S <br>
     * d(x, y) = d(y, x) 
     * 
     * @return true if this distance metric is symmetric, false if it is not
     */
    public boolean isSymmetric();
    
    /**
     * Returns true if this distance metric obeys the rule that, for any x, y, and z &isin; S <br>
     * d(x, z) &le; d(x, y) + d(y, z) 
     * 
     * @return true if this distance metric supports the triangle inequality, false if it does not. 
     */
    public boolean isSubadditive();
    
    /**
     * Returns true if this distance metric obeys the rule that, for any x and y  &isin; S <br>
     * d(x, y) = 0 if and only if x = y
     * @return true if this distance metric is indicemible, false otherwise. 
     */
    public boolean isIndiscemible();
    
    /**
     * All metrics must return values greater than or equal to 0. 
     * The upper bound on the value returned is different for 
     * different metrics. This method returns the theoretical 
     * maximal value that could be returned by this distance 
     * metric. That means {@link Double#POSITIVE_INFINITY } 
     * is a valid return value. 
     * 
     * @return the maximal distance for any two points in that could exist by this distance metric. 
     */
    public double metricBound();
}
