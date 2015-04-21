
package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.distributions.kernels.*;
import jsat.linear.Vec;

/**
 * Creates a distance metric from a given kernel trick. 
 * For the distance metric to be valid, the kernel used
 * must be positive definite. 
 * 
 * @author Edward Raff
 */
public class KernelDistance implements DistanceMetric
{

	private static final long serialVersionUID = -1553315486668768024L;
	private KernelTrick kf;

    /**
     * Creates a distane metric from the given kernel. For the metric to be valid, the kernel must be positive definite. This means that
     * <br><br>
     * &forall; c<sub>i</sub> &isin; &real; , x<sub>i</sub> &isin; &real;<sup>d</sup> <br>
     * &sum;<sub>i, j = 1</sub><sup>m</sup> c<sub>i</sub> c<sub>j</sub> K(x<sub>i</sub>, x<sub>j</sub>) &ge; 0
     * @param kf 
     */
    public KernelDistance(KernelTrick kf)
    {
        this.kf = kf;
    }
    
    /**
     * Returns the square of the distance function expanded as kernel methods. 
     * <br>
     * d<sup>2</sup>(x,y) = K(x,x) - 2*K(x,y) + K(y,y)
     * 
     * <br><br>
     * Special Notes:<br>
     * The use of {@link RBFKernel} or {@link PolynomialKernel} of degree 1 
     * in the {@link NearestNeighbour} classifier will degenerate into the
     * normal nearest neighbor algorithm. 
     * 
     * @param a the first vector
     * @param b the second vector
     * @return the distance metric based on a kernel function
     */
    @Override
    public double dist(Vec a, Vec b)
    {
        return kf.eval(a, a) - 2*kf.eval(a, b) + kf.eval(b, b);
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
        return "Kernel (" + kf +  ") Distance";
    }

    @Override
    public KernelDistance clone()
    {
        return new KernelDistance(kf);
    }

    //TODO modify CacheAccerleratedKernel to work with lists...
    
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
