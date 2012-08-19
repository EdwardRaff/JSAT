package jsat.linear.distancemetrics;

import jsat.linear.Vec;

/**
 * Minkowski Distance is the L<sub>p</sub> norm. 
 * 
 * @author Edward Raff
 */
public class MinkowskiDistance implements DistanceMetric
{
    
    private double p;

    public MinkowskiDistance(double p)
    {
        if(p <= 0 || Double.isNaN(p))
            throw new ArithmeticException("The pNorm exists only for p > 0");
        else if(Double.isInfinite(p))
            throw new ArithmeticException("Infinity norm is a special case, use ChebyshevDistance for infinity norm");
        
        this.p = p;
    }
    
    @Override
    public double dist(Vec a, Vec b)
    {
        return a.pNormDist(p, b);
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
        return "Minkowski Distance (p=" + p + ")";
    }

    @Override
    public MinkowskiDistance clone()
    {
        return new MinkowskiDistance(p);
    }
    
}
