
package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.linear.Vec;
import jsat.linear.VecOps;
import jsat.math.MathTricks;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.ParallelUtils;

/**
 * Implements the weighted Euclidean distance such that d(a, b) =
 * <big>&sum;</big><sub>&forall; i &isin; |w|</sub> w<sub>i</sub> 
 * (x<sub>i</sub>-y<sub>i</sub>)<sup>2</sup> <br>
 * When used with a weight vector of ones, it degenerates into 
 * the {@link EuclideanDistance}. 
 * 
 * @author Edward Raff
 */
public class WeightedEuclideanDistance implements DistanceMetric
{

	private static final long serialVersionUID = 2959997330647828673L;
	private Vec w;

    /**
     * Creates a new weighted Euclidean distance metric using the 
     * given set of weights.
     * @param w the weight to apply to each variable
     */
    public WeightedEuclideanDistance(Vec w)
    {
        setWeight(w);
    }

    /**
     * Returns the weight vector used by this object. Altering the returned 
     * vector is visible to this object, so there is no need to set it again
     * using {@link #setWeight(jsat.linear.Vec) }. If you do not want to 
     * alter it, you will need to clone the returned object and modify that. 
     * @return the weight vector used by this object
     */
    public Vec getWeight()
    {
        return w;
    }

    /**
     * Sets the weight vector to use for the distance function
     * @param w the weight vector to use
     * @throws NullPointerException if {@code w} is null
     */
    public void setWeight(Vec w)
    {
        if(w == null)
            throw new NullPointerException();
        this.w = w;
    }

    @Override
    public double dist(Vec a, Vec b)
    {
        return Math.sqrt(VecOps.accumulateSum(w, a, b, MathTricks.sqrdFunc));
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
    public WeightedEuclideanDistance clone()
    {
        return new WeightedEuclideanDistance(w.clone());
    }

    /*
     * Using: w_i (x_i - y_i)^2 = w_i x_i^2 - 2 w_i x_i y_i + w_i y_i^2 
     * Dots are a little weight, then use Vec ops for weighted dot
     * 
     * also use : w_i x_i^2 = w_i (x_i x_i) 
     * 
     */
    
    @Override
    public boolean supportsAcceleration()
    {
        return true;
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> vecs)
    {
        DoubleList cache = new DoubleList(vecs.size());
        
        for(Vec v : vecs)
            cache.add(VecOps.weightedDot(w, v, v));
        
        return cache;
    }
    
    @Override
    public List<Double> getAccelerationCache(final List<? extends Vec> vecs, ExecutorService threadpool)
    {
        if(threadpool == null || threadpool instanceof FakeExecutor)
            return getAccelerationCache(vecs);
        final double[] cache = new double[vecs.size()];
        
        final int P = Math.min(SystemInfo.LogicalCores, vecs.size());
        final CountDownLatch latch = new CountDownLatch(P);

        for(int ID = 0; ID < P; ID++)
        {
            final int start = ParallelUtils.getStartBlock(cache.length, ID, P);
            final int end = ParallelUtils.getEndBlock(cache.length, ID, P);
            threadpool.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    for(int i = start; i < end; i++)
                        cache[i] = VecOps.weightedDot(w, vecs.get(i), vecs.get(i));
                    latch.countDown();
                }
            });
        }
        
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(WeightedEuclideanDistance.class.getName()).log(Level.SEVERE, null, ex);
        }

        return DoubleList.view(cache, cache.length);
    }

    @Override
    public double dist(int a, int b, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), vecs.get(b));
        
        return Math.sqrt(cache.get(a)+cache.get(b)-2*VecOps.weightedDot(w, vecs.get(a), vecs.get(b)));
    }

    @Override
    public double dist(int a, Vec b, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), b);
        
        return Math.sqrt(cache.get(a)+VecOps.weightedDot(w, b, b)-2*VecOps.weightedDot(w, vecs.get(a), b));
    }

    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        DoubleList qi = new DoubleList(1);
        qi.add(VecOps.weightedDot(w, q, q));
        return qi;
    }

    @Override
    public double dist(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), b);
        
        return Math.sqrt(cache.get(a)+qi.get(0)-2*VecOps.weightedDot(w, vecs.get(a), b));
    }

}
