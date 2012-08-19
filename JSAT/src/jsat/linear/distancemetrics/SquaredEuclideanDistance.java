
package jsat.linear.distancemetrics;

import jsat.linear.SparceVector;
import jsat.linear.Vec;

/**
 * In many applications, the squared {@link EuclideanDistance} is used because it avoids an expensive {@link Math#sqrt(double) } operation. 
 * However, the Squared Euclidean Distance is not a truly valid metric, as it does not obey the {@link #isSubadditive() triangle inequality}.
 * 
 * @author Edward Raff
 */
public class SquaredEuclideanDistance implements DistanceMetric
{

    @Override
    public double dist(Vec a, Vec b)
    {
        if(a.length() != b.length())
            throw new ArithmeticException("Length miss match, vectors must have the same length");
        double d = 0;
        
        if( a instanceof SparceVector && b instanceof SparceVector)
        {
            //Just square the pNorm for now... not easy code to write, and the sparceness is more important
            return Math.pow(a.pNormDist(2, b), 2);
        }
        else
        {
            double tmp;
            for(int i = 0; i < a.length(); i++)
            {
                tmp = a.get(i) - b.get(i);
                d += tmp*tmp;
            }
        }
        
        return d;
    }

    @Override
    public boolean isSymmetric()
    {
        return true;
    }

    @Override
    public boolean isSubadditive()
    {
        return false;
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
        return "Squared Euclidean Distance";
    }
    
    @Override
    public SquaredEuclideanDistance clone()
    {
        return new SquaredEuclideanDistance();
    }
    
}
