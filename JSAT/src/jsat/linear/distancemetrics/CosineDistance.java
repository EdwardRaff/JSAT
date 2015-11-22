
package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.linear.Vec;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.ParallelUtils;

/**
 * The Cosine Distance is a adaption of the Cosine Similarity's range from 
 * [-1, 1] into the range [0, 1]. Where 0 means two vectors are the same, and 1 
 * means they are completely different. 
 * 
 * @author Edward Raff
 */
public class CosineDistance implements DistanceMetric
{
    /*
     * NOTE: Math.min(val, 1) is used because numerical instability can cause 
     * slightly larger values than 1 when the values are extremly close to 
     * eachother. In this case, it would cause a negative value in the sqrt of 
     * the cosineToDinstance calculation, resulting in a NaN. So the max is used
     * to avoid this.
     */

    private static final long serialVersionUID = -6475546704095989078L;

    @Override
    public double dist(Vec a, Vec b)
    {
        if(a.length() != b.length())
            throw new ArithmeticException("vectors a and b are of differeing legnths " + a.length() + " and " + b.length());
        /*
         * a dot b / (2Norm(a) * 2Norm(b)) will return a value in the range -1 to 1
         * -1 means they are completly opposite
         */
        double denom = a.pNorm(2) * b.pNorm(2);
        if(denom == 0)
            return cosineToDistance(-1);
        return cosineToDistance(Math.min(a.dot(b) / denom, 1));
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
        return "Cosine Distance";
    }

    @Override
    public CosineDistance clone()
    {
        return new CosineDistance();
    }

    @Override
    public boolean supportsAcceleration()
    {
        return true;
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> vecs)
    {
        //Store the pnorms in the cache
        DoubleList cache = new DoubleList(vecs.size());
        for(Vec v : vecs)
            cache.add(v.pNorm(2));
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
                        cache[i] = vecs.get(i).pNorm(2);
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
            Logger.getLogger(CosineDistance.class.getName()).log(Level.SEVERE, null, ex);
        }

        return DoubleList.view(cache, cache.length);
    }

    @Override
    public double dist(int a, int b, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), vecs.get(b));
        
        double denom = cache.get(a)*cache.get(b);
        if(denom == 0)
            return cosineToDistance(-1);
        return cosineToDistance(Math.min(vecs.get(a).dot(vecs.get(b)) / denom, 1));
    }

    @Override
    public double dist(int a, Vec b, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), b);
        
        double denom = cache.get(a)*b.pNorm(2);
        if(denom == 0)
            return cosineToDistance(-1);
        return cosineToDistance(Math.min(vecs.get(a).dot(b) / denom, 1));
    }

    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        DoubleList qi = new DoubleList(1);
        qi.add(q.pNorm(2));
        return qi;
    }

    @Override
    public double dist(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), b);
        
        double denom = cache.get(a)*qi.get(0);
        if(denom == 0)
            return cosineToDistance(-1);
        return cosineToDistance(Math.min(vecs.get(a).dot(b) / denom, 1));
    }
    
    /**
     * This method converts the cosine distance in [-1, 1] to a valid distance 
     * metric in the range [0, 1]
     * @param cosAngle the cosine similarity in [-1, 1]
     * @return the distance metric for the cosine value
     */
    public static double cosineToDistance(double cosAngle)
    {
        return Math.sqrt(0.5*(1-cosAngle));
    }
    
    /**
     * This method converts the distance obtained with 
     * {@link #cosineToDistance(double) } back into the cosine angle
     * @param dist the distance value in [0, 1]
     * @return the cosine angle
     */
    public static double distanceToCosine(double dist)
    {
        return 1-2*(dist*dist);
    }
    
}
