
package jsat.linear.distancemetrics;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class ChebyshevDistance implements DistanceMetric
{

    public double dist(Vec a, Vec b)
    {
        if(a.length() != b.length())
            throw new ArithmeticException("Vectors must have the same length");
        double max = 0;
        
        for(int i = 0; i < a.length(); i++)
            max = Math.max(max, Math.abs(a.get(i)-b.get(i)));
        
        return max;
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
    public ChebyshevDistance clone()
    {
        return new ChebyshevDistance();
    }
    
}
