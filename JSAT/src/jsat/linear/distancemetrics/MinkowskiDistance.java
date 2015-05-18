package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.linear.IndexValue;
import jsat.linear.Vec;

/**
 * Minkowski Distance is the L<sub>p</sub> norm. 
 * 
 * @author Edward Raff
 */
public class MinkowskiDistance implements DenseSparseMetric
{

    private static final long serialVersionUID = 8976696315441171045L;
    private double p;

    /**
     * 
     * @param p the norm to use as the distance
     */
    public MinkowskiDistance(double p)
    {
        if (p <= 0 || Double.isNaN(p))
            throw new ArithmeticException("The pNorm exists only for p > 0");
        else if (Double.isInfinite(p))
            throw new ArithmeticException("Infinity norm is a special case, use ChebyshevDistance for infinity norm");

        setP(p);
    }
    
    /**
     * 
     * @param p the norm to use for this metric
     */
    public void setP(double p)
    {
        if(p <= 0 || Double.isNaN(p) || Double.isInfinite(p))
            throw new IllegalArgumentException("p must be a positive value, not " + p);
        this.p = p;
    }

    /**
     * 
     * @return the norm to use for this metric. 
     */
    public double getP()
    {
        return p;
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

    @Override
    public double getVectorConstant(Vec vec)
    {
        return Math.pow(vec.pNorm(p), p);
    }

    @Override
    public double dist(double summaryConst, Vec main, Vec target)
    {
        if(!target.isSparse())
            return dist(main, target);
        /**
         * Summary contains the differences^p to the zero vec, only a few 
         * of the indices are actually non zero -  we correct those values
         */
        double addBack = 0.0;
        double takeOut = 0.0;
        for(IndexValue iv : target)
        {
            int i = iv.getIndex();
            double mainVal = main.get(i);
            takeOut += Math.pow(mainVal, p);
            addBack += Math.pow(mainVal-iv.getValue(), p);
        }
        return Math.pow(summaryConst-takeOut+addBack, 1/p);
    }

    @Override
    public boolean supportsAcceleration()
    {
        return false;
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> vecs)
    {
        return null;
    }

    @Override
    public double dist(int a, int b, List<? extends Vec> vecs, List<Double> cache)
    {
        return dist(vecs.get(a), vecs.get(b));
    }

    @Override
    public double dist(int a, Vec b, List<? extends Vec> vecs, List<Double> cache)
    {
        return dist(vecs.get(a), b);
    }

    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        return null;
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> vecs, ExecutorService threadpool)
    {
        return null;
    }
    
    @Override
    public double dist(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        return dist(vecs.get(a), b);
    }
    
}
