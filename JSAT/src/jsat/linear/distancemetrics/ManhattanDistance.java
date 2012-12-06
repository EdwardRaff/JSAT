
package jsat.linear.distancemetrics;

import jsat.linear.IndexValue;
import jsat.linear.Vec;

/**
 * Manhattan Distance is the L<sub>1</sub> norm. 
 * 
 * @author Edward Raff
 */
public class ManhattanDistance implements DenseSparseMetric
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
    
}
