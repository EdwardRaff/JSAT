
package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;

/**
 * Chebyshev Distance is the L<sub>&#8734;</sub> norm. 
 * 
 * @author Edward Raff
 */
public class ChebyshevDistance implements DistanceMetric
{

    private static final long serialVersionUID = 2528153647402824790L;

    @Override
    public double dist(Vec a, Vec b)
    {
        if(a.length() != b.length())
            throw new ArithmeticException("Vectors must have the same length");
        double max = 0;
        
        for(int i = 0; i < a.length(); i++)
            max = Math.max(max, Math.abs(a.get(i)-b.get(i)));
        
        return max;
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
        return "Chebyshev Distance";
    }
    
    @Override
    public ChebyshevDistance clone()
    {
        return new ChebyshevDistance();
    }
    
}
