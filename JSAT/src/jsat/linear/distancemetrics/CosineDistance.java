
package jsat.linear.distancemetrics;

import jsat.linear.Vec;

/**
 * The Cosine Distance is a adaption of the Cosine similarity's range from 
 * [-1, 1] into the range [0, 2]. Where 0 means two vectors are the same, and 2 
 * means they are completly different. 
 * 
 * @author Edward Raff
 */
public class CosineDistance implements DistanceMetric
{

    @Override
    public double dist(Vec a, Vec b)
    {
        /*
         * a dot b / (2Norm(a) * 2Norm(b)) will return a value in the range -1 to 1
         * -1 means they are completly opposite
         * 1 means they are exactly the same
         * 
         * by returnin the result a 1 - val, we mak it so the value returns is in the range 2 to 0. 
         * 2 (1 - -1 = 2) means they are completly opposite
         * 0 ( 1 -1) means they are completly the same
         */
        double denom = a.pNorm(2) * b.pNorm(2);
        if(denom == 0)
            return 2.0;
        return 1 - a.dot(b) / denom;
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
        return 2;
    }

    @Override
    public String toString()
    {
        return "Cosine Distance";
    }

    @Override
    public CosineDistance clone()
    {
        return new CosineDistance();
    }
    
}
