
package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.ParallelUtils;

/**
 * In many applications, the squared {@link EuclideanDistance} is used because it avoids an expensive {@link Math#sqrt(double) } operation. 
 * However, the Squared Euclidean Distance is not a truly valid metric, as it does not obey the {@link #isSubadditive() triangle inequality}.
 * 
 * @author Edward Raff
 */
public class SquaredEuclideanDistance implements DistanceMetric
{


	private static final long serialVersionUID = 2966818558802484702L;

	@Override
    public double dist(Vec a, Vec b)
    {
        if(a.length() != b.length())
            throw new ArithmeticException("Length miss match, vectors must have the same length");
        double d = 0;
        
        if( a instanceof SparseVector && b instanceof SparseVector)
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
            Logger.getLogger(SquaredEuclideanDistance.class.getName()).log(Level.SEVERE, null, ex);
        }

        return DoubleList.view(cache, cache.length);
    }
    
    @Override
    public double dist(int a, int b, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), vecs.get(b));
        
        return (cache.get(a)+cache.get(b)-2*vecs.get(a).dot(vecs.get(b)));
    }

    @Override
    public double dist(int a, Vec b, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), b);
        
        return (cache.get(a)+b.dot(b)-2*vecs.get(a).dot(b));
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
        
        return (cache.get(a)+qi.get(0)-2*vecs.get(a).dot(b));
    }
}
