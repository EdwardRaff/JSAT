
package jsat.linear.distancemetrics;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class ManhattanDistance implements DistanceMetric
{

    public double dist(Vec a, Vec b)
    {
        return a.pNormDist(1, b);
    }

    public boolean isSymmetric()
    {
        return true;
    }

    public boolean isSubadditive()
    {
        return true;
    }

    public boolean isIndiscemible()
    {
        return true;
    }

    public double metricBound()
    {
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public ManhattanDistance clone()
    {
        return new ManhattanDistance();
    }
    
}
