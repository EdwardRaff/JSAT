package jsat.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
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
    public int[] cluster(final DataSet dataSet, int clusters, ExecutorService threadpool, int[] designations)
    {
        final int[] des;
        if (designations == null)
            des = new int[dataSet.getSampleSize()];
        else
            des = designations;
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadpool);
        
        final int blockSize = dataSet.getSampleSize() / SystemInfo.LogicalCores;
        
        final List<Vec> means = SeedSelectionMethods.selectIntialPoints(dataSet, clusters, dm, new Random(), seedSelection, threadpool);
        final AtomicInteger changes = new AtomicInteger();
        
        final DenseSparseMetric dsm = dm instanceof DenseSparseMetric ? (DenseSparseMetric) dm : null;
        final double[] smc = dsm == null ? null : new double[clusters];
        if(smc != null)
            for(int i = 0; i < means.size(); i++)
                smc[i] = dsm.getVectorConstant(means.get(i));
        
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
                        double tmp;
                        for (int i = s; i < end; i++)
                        {
                            Vec x = dataSet.getDataPoint(i).getNumericalValues();
                            double minDist = Double.POSITIVE_INFINITY;
                            int min = -1;
                            for (int j = 0; j < means.size(); j++)
                            {
                                if(dsm != null)
                                    tmp = dsm.dist(smc[j], means.get(j), x);
                                else
                                    tmp = dm.dist(means.get(j), x);
                                if (tmp < minDist)
                                {
                                    minDist = tmp;
                                    min = j;
                                }
                            }
                            if(des[i] == min)
                                continue;
                            des[i] = min;
                            changes.incrementAndGet();
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
                //Recalc means
                int[] finalCounts = new int[clusters];
                List<Future<MeanComputer>> futures = new ArrayList<Future<MeanComputer>>(SystemInfo.LogicalCores);

                extra = dataSet.getSampleSize() % SystemInfo.LogicalCores;
                start = 0;

                while(start < dataSet.getSampleSize())
                {
                    final int end = start + blockSize + (extra-- > 0 ? 1 : 0);
                    futures.add(threadpool.submit(new MeanComputer(start, end, clusters, dataSet, des)));
                    start = end;
                }
                
                for(Vec mean : means)
                    mean.zeroOut();

                for(Future<MeanComputer> fmc : futures)
                {
                    MeanComputer mc = fmc.get();
                    for(int i = 0; i < clusters; i++)
                    {
                        finalCounts[i] += mc.meanCount[i];
                        means.get(i).mutableAdd(mc.newMeans.get(i));
                    }
                }
                
                for(int i = 0; i < clusters; i++)
                {
                    means.get(i).mutableDivide(finalCounts[i]);
                    if(dsm != null)
                        smc[i] = dsm.getVectorConstant(means.get(i));
                }
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(NaiveKMeans.class.getName()).log(Level.SEVERE, null, ex);
            }
            catch (ExecutionException ex)
            {
                Logger.getLogger(NaiveKMeans.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        while(changes.get() > 0);

        return des;
    }

    private class MeanComputer implements Callable<MeanComputer>
    {
        int start, end, k;
        DataSet dataSet;
        int[] des;
        List<Vec> newMeans;
        int[] meanCount;
        
        public MeanComputer(int start, int end, int k, DataSet dataSet, int[] des)
        {
            this.start = start;
            this.end = end;
            this.k = k;
            this.dataSet = dataSet;
            this.des = des;
            newMeans = new ArrayList<Vec>(k);
            for(int i = 0; i < k; i++)
                newMeans.add(new DenseVector(dataSet.getNumNumericalVars()));
            meanCount = new int[k];
        }
        
        
        @Override
        public MeanComputer call() throws Exception
        {
            for(int i = start; i < end; i++)
            {
                Vec x = dataSet.getDataPoint(i).getNumericalValues();
                int c = des[i];
                meanCount[c]++;
                newMeans.get(c).mutableAdd(x);
            }
            
            return this;
        }
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
