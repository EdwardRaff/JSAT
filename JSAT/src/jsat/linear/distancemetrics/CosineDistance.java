
package jsat.linear.distancemetrics;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class CosineDistance implements DistanceMetric
{

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
        
        return 1 - a.dot(b) / (a.pNorm(2) * b.pNorm(2));
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
        return 2;
    }
    
}
