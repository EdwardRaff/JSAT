package jsat.linear.distancemetrics;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class MinkowskiDistance implements DistanceMetric
{
    
    private double p;

    public MinkowskiDistance(double p)
    {
        if(p <= 0)
            throw new ArithmeticException("The pNorm exists only for p > 0");
        else if(Double.isInfinite(p))
            throw new ArithmeticException("Infinity norm is a special case, use ChebyshevDistance for infinity norm");
        
        this.p = p;
    }
    
    public double dist(Vec a, Vec b)
    {
        return a.pNormDist(p, b);
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
    public MinkowskiDistance clone()
    {
        return new MinkowskiDistance(p);
    }
    
}
