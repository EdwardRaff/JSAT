
package jsat.linear.distancemetrics;

import jsat.linear.Vec;

/**
 * Manhattan Distance is the L<sub>1</sub> norm. 
 * 
 * @author Edward Raff
 */
public class ManhattanDistance implements DistanceMetric
{

    @Override
    public double dist(Vec a, Vec b)
    {
        return a.pNormDist(1, b);
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
        return "Manhattan Distance";
    }
    
    @Override
    public ManhattanDistance clone()
    {
        return new ManhattanDistance();
    }
    
}
