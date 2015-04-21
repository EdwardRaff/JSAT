
package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.linear.IndexValue;
import jsat.linear.Vec;

/**
 * Manhattan Distance is the L<sub>1</sub> norm. 
 * 
 * @author Edward Raff
 */
public class ManhattanDistance implements DenseSparseMetric
{


	private static final long serialVersionUID = 3028834823742743351L;

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

    @Override
    public double getVectorConstant(Vec vec)
    {
        return vec.pNorm(1);
    }

    @Override
    public double dist(double summaryConst, Vec main, Vec target)
    {
        if(!target.isSparse())
            return dist(main, target);
        /**
         * Summary contains the differences to the zero vec, only a few 
         * of the indices are actually non zero -  we correct those values
         */
        double takeOut = 0.0;
        for(IndexValue iv : target)
        {
            int i = iv.getIndex();
            double mainVal = main.get(i);
            takeOut += mainVal-Math.abs(mainVal-iv.getValue());
        }
        return summaryConst-takeOut;
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
