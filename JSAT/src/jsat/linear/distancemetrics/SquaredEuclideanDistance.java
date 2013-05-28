
package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.utils.DoubleList;
import jsat.utils.SystemInfo;

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
        final double[] cache = new double[vecs.size()];
   
        final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
        final int blockSize = cache.length / SystemInfo.LogicalCores;
        int extra = cache.length % SystemInfo.LogicalCores;
        int start = 0;

        while (start < cache.length)
        {
            final int S = start;
            final int end;
            if (extra-- > 0)
                end = start + blockSize + 1;
            else
                end = start + blockSize;
            threadpool.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    for(int i = S; i < end; i++)
                        cache[i] = vecs.get(i).dot(vecs.get(i));
                    latch.countDown();
                }
            });
            start = end;
        }

        return DoubleList.unmodifiableView(cache, cache.length);
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
