package jsat.linear.vectorcollection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.VecPairedComparable;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.*;

/**
 * Provides a specialized collections that only supports the 
 * {@link EuclideanDistance} by brute force search. It does this more 
 * efficiently by decomposing the squared distance into<br>
 * d(x, y)<sup>2</sup> = || x - y||<sup>2</sup> = &lt;x, x&gt; + &lt;y, y&gt; - 2 &lt;x, y&gt;
 * <br>
 * The shortest distances can then be obtained using only 1 dot product per 
 * element, as the input and collection values are cached. This avoids redundant
 * calculations present in the naive approach, and can result in drastic 
 * performance increases when using sparse vectors or a mix of dense and sparse
 * vectors.
 * 
 * @author Edward Raff
 */
public class EuclideanCollection<V extends Vec> implements VectorCollection<V>
{

	private static final long serialVersionUID = 3544832051605265927L;
	private List<V> source;
    /**
     * Cache of y dot y values
     */
    private double[] dotCache;

    /**
     * Creates a new Vector Collection that does an efficient brute force search
     * for only the Euclidean distance. 
     * @param source the set of vectors to form the collection
     */
    public EuclideanCollection(List<V> source)
    {
        this(source, new FakeExecutor());
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public EuclideanCollection(EuclideanCollection toCopy)
    {
        this.source = new ArrayList<V>(toCopy.source);
        this.dotCache = Arrays.copyOf(toCopy.dotCache, toCopy.dotCache.length);
    }
    
    /**
     * Creates a new Vector Collection that does an efficient brute force search
     * for only the Euclidean distance. 
     * @param source the set of vectors to form the collection
     * @param threadpool the source of threads for parallel construction
     */
    public EuclideanCollection(final List<V> source, ExecutorService threadpool)
    {
        this.source = source;
        dotCache = new double[source.size()];
        
        final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
        int start = 0;
        for(int id = 0; id < SystemInfo.LogicalCores; id++)
        {
            final int S = start;
            final int E = id == SystemInfo.LogicalCores-1 ? dotCache.length : start + dotCache.length/SystemInfo.LogicalCores;
            start = E;
            threadpool.submit(new Runnable() 
            {
                @Override
                public void run()
                {
                    for(int i = S; i < E; i++)
                    {
                        Vec c = source.get(i);
                        dotCache[i] = c.dot(c);
                    }
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
            Logger.getLogger(EuclideanCollection.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, double range)
    {
        List<VecPairedComparable<V, Double>> list = new ArrayList<VecPairedComparable<V, Double>>();
        //get the sacled and shifted value of range for fast comparison for d(x, y)^2
        final double xx = query.dot(query);
        final double cmpRange = range*range-xx;
        for(int i = 0; i < dotCache.length; i++)
        {
            double v = dotCache[i]-2*query.dot(source.get(i));
            if(v <= cmpRange)
                list.add(new VecPairedComparable<V, Double>(source.get(i), Math.sqrt(xx+v)));
        }
        Collections.sort(list);
        return list;
    }

    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, int neighbors)
    {
        BoundedSortedList<ProbailityMatch<V>> boundedList= new BoundedSortedList<ProbailityMatch<V>>(neighbors, neighbors);
        for(int i = 0; i < dotCache.length; i++)
        {
            double v = dotCache[i]-2*query.dot(source.get(i));
            if(boundedList.size() < neighbors || v < boundedList.get(neighbors-1).getProbability())
                boundedList.add(new ProbailityMatch<V>(v, source.get(i)));
        }
        
        double xx = query.dot(query);
        List<VecPaired<V, Double>> list = new ArrayList<VecPaired<V, Double>>(boundedList.size());
        for(ProbailityMatch<V> pm : boundedList)
            list.add(new VecPaired<V, Double>(pm.getMatch(), Math.sqrt(xx+pm.getProbability())));
        return list;
    }

    @Override
    public int size()
    {
        return dotCache.length;
    }

    @Override
    public EuclideanCollection<V> clone()
    {
        return new EuclideanCollection<V>(this);
    }
    
    public static class EuclideanCollectionFactory<V extends Vec> implements VectorCollectionFactory<V>
    {
        /**
		 * 
		 */
		private static final long serialVersionUID = 4838578403165658320L;

		@Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric)
        {
            if(!(distanceMetric instanceof EuclideanDistance))
                throw new IllegalArgumentException("EuclideanCollection only supports Euclidean Distanse");
            return new EuclideanCollection<V>(source);
        }

        @Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric, ExecutorService threadpool)
        {
            if(!(distanceMetric instanceof EuclideanDistance))
                throw new IllegalArgumentException("EuclideanCollection only supports Euclidean Distanse");
            return new EuclideanCollection<V>(source, threadpool);
        }

        @Override
        public EuclideanCollectionFactory<V> clone()
        {
            return new EuclideanCollectionFactory<V>();
        }
    }
}
