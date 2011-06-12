
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
    
}
