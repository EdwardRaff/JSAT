
package jsat.linear.distancemetrics;

import jsat.linear.Vec;

/**
 * Euclidean Distance is the L<sub>2</sub> norm. 
 * 
 * @author Edward Raff
 */
public class EuclideanDistance implements DistanceMetric
{

    @Override
    public double dist(Vec a, Vec b)
    {
        return a.pNormDist(2, b);
    }

    @Override
    public boolean isSymmetric()
    {
        return true;
    }

    @Override
    public boolean isSubadditive()
    {
        return true;
    }

    @Override
    public boolean isIndiscemible()
    {
        return true;
    }

    @Override
    public double metricBound()
    {
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public String toString()
    {
        return "Euclidean Distance";
    }

    @Override
    public EuclideanDistance clone()
    {
        return new EuclideanDistance();
    }
    
}
