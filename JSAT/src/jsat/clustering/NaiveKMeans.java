package jsat.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DenseSparseMetric;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;

/**
 * An implementation of Lloyd's K-Means clustering algorithm using the 
 * naive algorithm. This implementation exists mostly for comparison as
 * a base line and educational reasons. For efficient exact k-Means,
 * use {@link KMeans}<br>
 * <br>
 * This implementation is parallel, but does not support any of the 
 * clustering methods that do not specify the number of clusters. 
 * 
 * @author Edward Raff
 */
public class NaiveKMeans extends KClustererBase
{
    private DistanceMetric dm;
    private SeedSelectionMethods.SeedSelection seedSelection;
    
    private boolean storeMeans = true;
    private List<Vec> means;

    /**
     * Creates a new naive k-Means cluster using 
     * {@link SeedSelection#KPP k-means++} for the 
     * seed selection and the {@link EuclideanDistance}
     */
    public NaiveKMeans()
    {
        this(new EuclideanDistance());
    }

    /**
     * Creates a new naive k-Means cluster using 
     * {@link SeedSelection#KPP k-means++} for the seed selection.
     * @param dm the distance function to use
     */
    public NaiveKMeans(DistanceMetric dm)
    {
        this(dm, SeedSelectionMethods.SeedSelection.KPP);
    }

    /**
     * Creates a new naive k-Means cluster
     * @param dm the distance function to use
     * @param seedSelection the method of selecting the initial seeds
     */
    public NaiveKMeans(DistanceMetric dm, SeedSelection seedSelection)
    {
        this.dm = dm;
        this.seedSelection = seedSelection;
    }
    
    /**
     * If set to {@code true} the computed means will be stored after clustering
     * is completed, and can then be retrieved using {@link #getMeans() }. 
     * @param storeMeans {@code true} if the means should be stored for later, 
     * {@code false} to discard them once clustering is complete. 
     */
    public void setStoreMeans(boolean storeMeans)
    {
        this.storeMeans = storeMeans;
    }

    /**
     * Returns the raw list of means that were used for each class. 
     * @return the list of means for each class
     */
    public List<Vec> getMeans()
    {
        return means;
    }
    
    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int[] cluster(final DataSet dataSet, final int clusters, ExecutorService threadpool, int[] designations)
    {
        final int[] des;
        if (designations == null)
            des = new int[dataSet.getSampleSize()];
        else
            des = designations;
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadpool);
        
        final int blockSize = dataSet.getSampleSize() / SystemInfo.LogicalCores;
        final List<Vec> X = dataSet.getDataVectors();
        final List<Double> accelCache;
        if(threadpool == null || threadpool instanceof  FakeExecutor)
            accelCache = dm.getAccelerationCache(X);
        else
            accelCache = dm.getAccelerationCache(X, threadpool);
        
        means = SeedSelectionMethods.selectIntialPoints(dataSet, clusters, dm, accelCache, new Random(), seedSelection, threadpool);
        
        final List<List<Double>> meanQIs = new ArrayList<List<Double>>(clusters);
        
        //Use dense mean objects
        for(int i = 0; i < means.size(); i++)
        {
            if(dm.supportsAcceleration())
                meanQIs.add(dm.getQueryInfo(means.get(i)));
            else
                meanQIs.add(Collections.EMPTY_LIST);
            
            if(means.get(i).isSparse())
                means.set(i, new DenseVector(means.get(i)));
        }
        
        final List<Vec> meanSum = new ArrayList<Vec>(means.size());
        final AtomicIntegerArray meanCounts = new AtomicIntegerArray(means.size());
        for(int i = 0; i < clusters; i++)
            meanSum.add(new DenseVector(means.get(0).length()));
        final AtomicInteger changes = new AtomicInteger();
        
        //used to store local changes to the means and accumulated at the end
        final ThreadLocal<Vec[]> localMeanDeltas = new ThreadLocal<Vec[]>()
        {
            @Override
            protected Vec[] initialValue()
            {
                Vec[] deltas = new Vec[clusters];
                for(int i = 0; i < clusters; i++)
                    deltas[i] = new DenseVector(means.get(0).length());
                return deltas;
            }
        };
        
        Arrays.fill(des, -1);
        do
        {
            changes.set(0);
            int extra = dataSet.getSampleSize() % SystemInfo.LogicalCores;
            int start = 0;
            final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
            while(start < dataSet.getSampleSize())
            {
                final int s = start;
                final int end = start + blockSize + (extra-- > 0 ? 1 : 0);
                threadpool.submit(new Runnable()
                {
                    @Override
                    public void run()
                    {
                        Vec[] deltas = localMeanDeltas.get();
                        double tmp;
                        for (int i = s; i < end; i++)
                        {
                            final Vec x = X.get(i);
                            double minDist = Double.POSITIVE_INFINITY;
                            int min = -1;
                            for (int j = 0; j < means.size(); j++)
                            {
                                tmp = dm.dist(i, means.get(j), meanQIs.get(j), X, accelCache);
                                if (tmp < minDist)
                                {
                                    minDist = tmp;
                                    min = j;
                                }
                            }
                            if(des[i] == min)
                                continue;
                            
                            //add change
                            deltas[min].mutableAdd(x);
                            meanCounts.incrementAndGet(min);
                            //remove from prev owner
                            if(des[i] >= 0)
                            {
                                deltas[des[i]].mutableSubtract(x);
                                meanCounts.getAndDecrement(des[i]);
                            }
                            des[i] = min;
                            changes.incrementAndGet();
                        }
                        
                        //accumulate deltas into globals
                        for(int i = 0; i < deltas.length; i++)
                            synchronized(meanSum.get(i))
                            {
                                meanSum.get(i).mutableAdd(deltas[i]);
                                deltas[i].zeroOut();
                            }
                        
                        latch.countDown();
                    }
                });
                
                start = end;
            }
            
            try
            {
                latch.await();
                if(changes.get() == 0)
                    break;
                for(int i = 0; i < clusters; i++)
                {
                    meanSum.get(i).copyTo(means.get(i));
                    means.get(i).mutableDivide(meanCounts.get(i));
                    if(dm.supportsAcceleration())
                        meanQIs.set(i, dm.getQueryInfo(means.get(i)));
                }
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(NaiveKMeans.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        while(changes.get() > 0);
        
        if(!storeMeans)
            means = null;

        return des;
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, int[] designations)
    {
        return cluster(dataSet, clusters, new FakeExecutor(), designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
}
