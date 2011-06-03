
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
}
