package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;
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

    /*
     * NOTE: Math.min(val, 1) is used because numerical instability can cause 
     * slightly larger values than 1 when the values are extremly close to 
     * eachother. In this case, it would cause a negative value in the sqrt of 
     * the cosineToDinstance calculation, resulting in a NaN. So the max is used
     * to avoid this.
     */
    

	private static final long serialVersionUID = -4041803247001806577L;

	@Override
    public double dist(Vec a, Vec b)
    {
        return CosineDistance.cosineToDistance(Math.min(a.dot(b), 1));
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
        return 1;
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
