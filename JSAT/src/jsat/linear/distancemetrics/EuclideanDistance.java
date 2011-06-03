
package jsat.linear.distancemetrics;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class EuclideanDistance implements DistanceMetric
{

    public double dist(Vec a, Vec b)
    {
        return a.pNormDist(2, b);
    }
    
}
