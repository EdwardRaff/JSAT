
package jsat.linear.distancemetrics;

import jsat.linear.IndexValue;
import jsat.linear.Vec;

/**
 * Euclidean Distance is the L<sub>2</sub> norm. 
 * 
 * @author Edward Raff
 */
public class EuclideanDistance implements DenseSparseMetric
{

    @Override
    public double dist(Vec a, Vec b)
    {
        return a.pNormDist(2, b);
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
        return "Euclidean Distance";
    }

    @Override
    public EuclideanDistance clone()
    {
        return new EuclideanDistance();
    }

    @Override
    public double getVectorConstant(Vec vec)
    {
        /* Returns the sum of squarred differences if the other vec had been all 
         * zeros. That means this is one sqrt away from being the euclidean 
         * distance to the zero vector. 
         */
        return Math.pow(vec.pNorm(2), 2.0);
    }

    @Override
    public double dist(double summaryConst, Vec main, Vec target)
    {
        if(!target.isSparse())
            return dist(main, target);
        /**
         * Summary contains the squared differences to the zero vec, only a few 
         * of the indices are actually non zero -  we correct those values
         */
        double addBack = 0.0;
        double takeOut = 0.0;
        for(IndexValue iv : target)
        {
            int i = iv.getIndex();
            double mainVal = main.get(i);
            takeOut += Math.pow(main.get(i), 2);
            addBack += Math.pow(main.get(i)-iv.getValue(), 2.0);
        }
        return Math.sqrt(summaryConst-takeOut+addBack);
    }
    
}
