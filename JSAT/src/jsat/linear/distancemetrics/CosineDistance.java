
package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;
import jsat.utils.DoubleList;
import jsat.utils.SystemInfo;

/**
 * The Cosine Distance is a adaption of the Cosine similarity's range from 
 * [-1, 1] into the range [0, 2]. Where 0 means two vectors are the same, and 2 
 * means they are completly different. 
 * 
 * @author Edward Raff
 */
public class CosineDistance implements DistanceMetric
{

    @Override
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
        double denom = a.pNorm(2) * b.pNorm(2);
        if(denom == 0)
            return 2.0;
        return 1 - a.dot(b) / denom;
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
                        cache[i] = vecs.get(i).pNorm(2);
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
        
        double denom = cache.get(a)*cache.get(b);
        if(denom == 0)
            return 2.0;
        return 1 - vecs.get(a).dot(vecs.get(b)) / denom;
    }

    @Override
    public double dist(int a, Vec b, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), b);
        
        double denom = cache.get(a)*b.pNorm(2);
        if(denom == 0)
            return 2.0;
        return 1 - vecs.get(a).dot(b) / denom;
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
            return 2.0;
        return 1 - vecs.get(a).dot(b) / denom;
    }
    
}
