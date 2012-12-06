package jsat.linear.distancemetrics;

import jsat.linear.Vec;

/**
 * This distance metric returns the same cosine distance as 
 * {@link CosineDistance}. This implementation assumes that all vectors being
 * passed in for distance computations have already been L2 normalized. This 
 * means the distance computation can be done more efficiently, but the results 
 * will be incorrect if the inputs have not already been normalized. <br>
 * The word Normalized is postfixed to the name to avoid confusion, as many 
 * might assume "Normalized-CosineDistance" would mean a cosine distance with 
 * some form of additional normalization. 
 * 
 * @author Edward Raff
 */
public class CosineDistanceNormalized implements DistanceMetric
{

    @Override
    public double dist(Vec a, Vec b)
    {
        return 1.0-a.dot(b);
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
        return "Cosine Distance (Normalized)";
    }

    @Override
    public CosineDistanceNormalized clone()
    {
        return new CosineDistanceNormalized();
    }
    
}
