package jsat.clustering.kmeans;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.clustering.KClustererBase;
import jsat.clustering.SeedSelectionMethods;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.*;
import jsat.utils.*;
import jsat.utils.random.RandomUtil;

/**
 * Implements the mini-batch algorithms for k-means. This is a stochastic algorithm, 
 * so it does not find the global solution. This implementation is parallel, but 
 * only the methods that specify the exact number of clusters are supported. <br>
 * <br>
 * See: Sculley, D. (2010). <i>Web-scale k-means clustering</i>. Proceedings of the 
 * 19th international conference on World wide web (pp. 1177–1178). 
 * New York, New York, USA: ACM Press. doi:10.1145/1772690.1772862
 * 
 * @author Edward Raff
 */
public class MiniBatchKMeans extends KClustererBase
{

	private static final long serialVersionUID = 412553399508594014L;
	private int batchSize;
    private int iterations;
    private DistanceMetric dm;
    private SeedSelectionMethods.SeedSelection seedSelection;
    
    private boolean storeMeans = true;
    private List<Vec> means;

    /**
     * Creates a new Mini-Batch k-Means object that uses 
     * {@link SeedSelection#KPP k-means++} for seed selection
     * and uses the {@link EuclideanDistance}. 
     * 
     * @param batchSize the mini-batch size
     * @param iterations the number of mini batches to perform
     */
    public MiniBatchKMeans(int batchSize, int iterations)
    {
        this(new EuclideanDistance(), batchSize, iterations);
    }

    /**
     * Creates a new Mini-Batch k-Means object that uses 
     * {@link SeedSelection#KPP k-means++} for seed selection. 
     * 
     * @param dm the distance metric to use
     * @param batchSize the mini-batch size
     * @param iterations the number of mini batches to perform
     */
    public MiniBatchKMeans(DistanceMetric dm, int batchSize, int iterations)
    {
        this(dm, batchSize, iterations, SeedSelectionMethods.SeedSelection.KPP);
    }
    
    /**
     * Creates a new Mini-Batch k-Means object
     * @param dm the distance metric to use
     * @param batchSize the mini-batch size
     * @param iterations the number of mini batches to perform
     * @param seedSelection the seed selection algorithm to initiate clustering
     */
    public MiniBatchKMeans(DistanceMetric dm, int batchSize, int iterations, SeedSelection seedSelection)
    {
        setBatchSize(batchSize);
        setIterations(iterations);
        setDistanceMetric(dm);
        setSeedSelection(seedSelection);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public MiniBatchKMeans(MiniBatchKMeans toCopy)
    {
        this.batchSize = toCopy.batchSize;
        this.iterations = toCopy.iterations;
        this.dm = toCopy.dm.clone();
        this.seedSelection = toCopy.seedSelection;
        this.storeMeans = toCopy.storeMeans;
        if(toCopy.means != null)
        {
            this.means = new ArrayList<Vec>();
            for(Vec v : toCopy.means)
                this.means.add(v.clone());
        }
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

    /**
     * Sets the distance metric used for determining the nearest cluster center
     * @param dm the distance metric to use
     */
    public void setDistanceMetric(DistanceMetric dm)
    {
        this.dm = dm;
    }

    /**
     * Returns the distance metric used for determining the nearest cluster center
     * @return the distance metric in use
     */
    public DistanceMetric getDistanceMetric()
    {
        return dm;
    }

    /**
     * Sets the batch size to use at each iteration. Increasing the 
     * batch size can improve the resulting clustering, but increases
     * computational cost at each iteration. <br>
     * If the batch size is set equal to or larger than data set size, 
     * it reduces to the {@link NaiveKMeans naive k-means} algorithm.
     * @param batchSize the number of points to use at each iteration
     */
    public void setBatchSize(int batchSize)
    {
        if(batchSize < 1)
            throw new ArithmeticException("Batch size must be a positive value, not " + batchSize);
        this.batchSize = batchSize;
    }

    /**
     * Returns the batch size used at each iteration
     * @return the batch size in use
     */
    public int getBatchSize()
    {
        return batchSize;
    }

    /**
     * Sets the number of mini-batch iterations to perform
     * @param iterations the number of algorithm iterations to perform
     */
    public void setIterations(int iterations)
    {
        if(iterations < 1)
            throw new ArithmeticException("Iterations must be a positive value, not " + iterations);
        this.iterations = iterations;
    }

    /**
     * Returns the number of mini-batch iterations used
     * @return the number of algorithm iterations that will be used
     */
    public int getIterations()
    {
        return iterations;
    }

    /**
     * Sets the method of selecting the initial data points to 
     * seed the clustering algorithm. 
     * @param seedSelection the seed selection algorithm to use
     */
    public void setSeedSelection(SeedSelection seedSelection)
    {
        this.seedSelection = seedSelection;
    }

    /**
     * Returns the method of seed selection to use
     * @return the method of seed selection to use
     */
    public SeedSelection getSeedSelection()
    {
        return seedSelection;
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
    public int[] cluster(DataSet dataSet, int clusters, ExecutorService threadpool, int[] designations) 
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadpool);
        
        final List<Vec> source = dataSet.getDataVectors();
        final List<Double> distCache;
        if(threadpool == null || threadpool instanceof FakeExecutor)
            distCache = dm.getAccelerationCache(source);
        else
            distCache = dm.getAccelerationCache(source, threadpool);
        
        means = SeedSelectionMethods.selectIntialPoints(dataSet, clusters, dm, distCache, RandomUtil.getRandom(), seedSelection, threadpool);
        
        final List<List<Double>> meanQIs = new ArrayList<List<Double>>(means.size());
        for (int i = 0; i < means.size(); i++)
            if (dm.supportsAcceleration())
                meanQIs.add(dm.getQueryInfo(means.get(i)));
            else
                meanQIs.add(Collections.EMPTY_LIST);

        final int[] v = new int[means.size()];
        
        final int usedBatchSize = Math.min(batchSize, dataSet.getSampleSize());
        
        /**
         * Store the indices of the sampled points instead of sampling, that 
         * way we can use the distance acceleration cache. 
         */
        final List<Integer> M = new IntList(usedBatchSize);
        final List<Integer> allIndx = new IntList(source.size());
        ListUtils.addRange(allIndx, 0, source.size(), 1);
        final int[] nearestCenter = new int[usedBatchSize];
        
                
        for(int iter = 0; iter < iterations; iter++)
        {
            M.clear();
            ListUtils.randomSample(allIndx, M, usedBatchSize);
            
            {//compute centers
                final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
                final int blockSize = usedBatchSize / SystemInfo.LogicalCores;
                int extra = usedBatchSize % SystemInfo.LogicalCores;
                int start = 0;
                while (start < usedBatchSize)
                {
                    final int s = start;
                    final int end = start + blockSize + (extra-- > 0 ? 1 : 0);
                    start = end;
                    threadpool.submit(new Runnable()
                    {
                        @Override
                        public void run()
                        {
                            double tmp;
                            for (int i = s; i < end; i++)
                            {
                                double minDist = Double.POSITIVE_INFINITY;
                                int min = -1;
                                
                                for (int j = 0; j < means.size(); j++)
                                {
                                    tmp = dm.dist(M.get(i), means.get(j), meanQIs.get(j), source, distCache);
                                    
                                    if (tmp < minDist)
                                    {
                                        minDist = tmp;
                                        min = j;
                                    }
                                }
                                nearestCenter[i] = min;
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
                    Logger.getLogger(MiniBatchKMeans.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            
            //Update centers
            for(int j = 0; j < M.size(); j++)
            {
                int c_i = nearestCenter[j];
                double eta = 1.0/(++v[c_i]);
                Vec c = means.get(c_i);
                c.mutableMultiply(1-eta);
                c.mutableAdd(eta, source.get(M.get(j)));
            }
            
            //update mean caches
            if(dm.supportsAcceleration())
                for(int i = 0; i < means.size(); i++)
                    meanQIs.set(i, dm.getQueryInfo(means.get(i)));
        }
        
        //Stochastic travel complete, calculate all
        List<Future<Double>> futures = new ArrayList<Future<Double>>(SystemInfo.LogicalCores);//Getting the objective function
        final int blockSize = dataSet.getSampleSize() / SystemInfo.LogicalCores;
        int extra = dataSet.getSampleSize() % SystemInfo.LogicalCores;
        int start = 0;

        final int[] des = designations;


        while (start < dataSet.getSampleSize())
        {
            final int s = start;
            final int end = start + blockSize + (extra-- > 0 ? 1 : 0);
            start = end;

            futures.add(threadpool.submit(new Callable<Double>()
            {
                @Override
                public Double call() throws Exception
                {
                    double dists = 0;
                    double tmp;
                    for (int i = s; i < end; i++)
                    {
                        double minDist = Double.POSITIVE_INFINITY;
                        int min = -1;
                        for (int j = 0; j < means.size(); j++)
                        {
                            tmp = dm.dist(i, means.get(j), meanQIs.get(j), source, distCache);

                            if (tmp < minDist)
                            {
                                minDist = tmp;
                                min = j;
                            }
                        }
                        
                        des[i] = min;
                        dists += minDist*minDist;
                    }
                    return dists;
                }
            }));
        }

        double sumErr = 0;

        try
        {
            for (Future<Double> future : futures)
                sumErr += future.get();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(MiniBatchKMeans.class.getName()).log(Level.SEVERE, null, ex);
        }
        catch (ExecutionException ex)
        {
            Logger.getLogger(MiniBatchKMeans.class.getName()).log(Level.SEVERE, null, ex);
        }
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

    @Override
    public MiniBatchKMeans clone()
    {
        return new MiniBatchKMeans(this);
    }
    
}
