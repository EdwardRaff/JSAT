
package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.ParallelUtils;

/**
 * Euclidean Distance is the L<sub>2</sub> norm. 
 * 
 * @author Edward Raff
 */
public class EuclideanDistance implements DenseSparseMetric
{


	private static final long serialVersionUID = 8155062933851345574L;

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
        return Math.sqrt(Math.max(summaryConst-takeOut+addBack, 0));//Max incase of numerical issues
    }

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
            cache.add(v.dot(v));
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
                        cache[i] = vecs.get(i).dot(vecs.get(i));
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
            Logger.getLogger(EuclideanDistance.class.getName()).log(Level.SEVERE, null, ex);
        }

        return DoubleList.view(cache, cache.length);
    }

    @Override
    public double dist(int a, int b, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), vecs.get(b));
        
        return Math.sqrt(Math.max(cache.get(a)+cache.get(b)-2*vecs.get(a).dot(vecs.get(b)), 0));//Max incase of numerical issues
    }

    @Override
    public double dist(int a, Vec b, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), b);
        
        return Math.sqrt(Math.max(cache.get(a)+b.dot(b)-2*vecs.get(a).dot(b), 0));//Max incase of numerical issues
    }

    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        DoubleList qi = new DoubleList(1);
        qi.add(q.dot(q));
        return qi;
    }

    @Override
    public double dist(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), b);
        
        return Math.sqrt(Math.max(cache.get(a)+qi.get(0)-2*vecs.get(a).dot(b), 0));//Max incase of numerical issues
    }
    
}
