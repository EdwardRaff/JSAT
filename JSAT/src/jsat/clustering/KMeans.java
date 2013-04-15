
package jsat.clustering;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import static jsat.clustering.SeedSelectionMethods.selectIntialPoints;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.*;
import jsat.math.OnLineStatistics;
import jsat.utils.*;

/**
 * An efficient implementation of the K-Means algorithm. This implementation uses
 * the triangle inequality to accelerate computation while maintaining the exact
 * same solution. This requires that the {@link DistanceMetric} used support 
 * {@link DistanceMetric#isSubadditive() }. 
 * <br>
 * Implementation based on the paper: Using the Triangle Inequality to Accelerate k-Means, by Charles Elkan
 * 
 * @author Edward Raff
 */
public class KMeans extends KClustererBase
{
    /**
     * This is the default seed selection method used in KMeans. When used with 
     * the {@link EuclideanDistance}, it selects seeds that are log optimal with
     * a high probability. 
     */
    public static final SeedSelection DEFAULT_SEED_SELECTION = SeedSelection.KPP;
    
    private DistanceMetric dm;
    private DenseSparseMetric dmds;
    private Random rand;
    private SeedSelection seedSelection;
    
    private boolean storeMeans = true;
    private boolean useDenseSparse = false;
    private List<Vec> means;
    
    /**
     * Control the maximum number of iterations to perform. 
     */
    protected int MaxIterLimit = Integer.MAX_VALUE;

    /**
     * Creates a new KMeans instance. 
     * @param dm the distance metric to use, must support {@link DistanceMetric#isSubadditive() }. 
     * @param rand the random number generator to use during seed selection
     * @param seedSelection the method of seed selection to use
     */
    public KMeans(DistanceMetric dm, Random rand, SeedSelection seedSelection)
    {
        if(!dm.isSubadditive())
            throw new ClusterFailureException("KMeans implementation requires the triangle inequality");
        this.dm = dm;
        this.rand = rand;
        this.seedSelection = seedSelection;
    }

    /**
     * Creates a new KMeans instance
     * @param dm the distance metric to use, must support {@link DistanceMetric#isSubadditive() }.  
     * @param rand the random number generator to use during seed selection
     */
    public KMeans(DistanceMetric dm, Random rand)
    {
        this(dm, rand, DEFAULT_SEED_SELECTION);
    }

    /**
     * Creates a new KMeans instance
     * @param dm the distance metric to use, must support {@link DistanceMetric#isSubadditive() }.  
     */
    public KMeans(DistanceMetric dm)
    {
        this(dm, new Random());
    }

    /**
     * Creates a new KMeans instance. The {@link EuclideanDistance} will be used by default. 
     */
    public KMeans()
    {
        this(new EuclideanDistance());
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
     * Sets the maximum number of iterations allowed
     * @param iterLimit the nex maximum number of iterations of the KMeans algorithm 
     */
    public void setIterationLimit(int iterLimit)
    {
        this.MaxIterLimit = iterLimit;
    }

    /**
     * Returns the maximum number of iterations of the KMeans algorithm that will be performed. 
     * @return the maximum number of iterations of the KMeans algorithm that will be performed. 
     */
    public int getIterationLimit()
    {
        return MaxIterLimit;
    }

    /**
     * Sets the method of seed selection to use for this algorithm. {@link SeedSelection#KPP} is recommended for this algorithm in particular. 
     * @param seedSelection the method of seed selection to use
     */
    public void setSeedSelection(SeedSelection seedSelection)
    {
        this.seedSelection = seedSelection;
    }

    /**
     * 
     * @return the method of seed selection used
     */
    public SeedSelection getSeedSelection()
    {
        return seedSelection;
    }

    /**
     * Sets whether or not to use {@link DenseSparseMetric } when computing. 
     * This may or may not provide a speed increase. 
     * @param useDenseSparse whether or not to compute the distance from dense
     * mean vectors to sparse ones using acceleration
     */
    public void setUseDenseSparse(boolean useDenseSparse)
    {
        this.useDenseSparse = useDenseSparse;
    }

    /**
     * Returns if Dense Sparse acceleration will be used if available
     * @return if Dense Sparse acceleration will be used if available
     */
    public boolean isUseDenseSparse()
    {
        return useDenseSparse;
    }
    
    /**
     * This is a helper method where the actual cluster is performed. This is because there
     * are multiple strategies for modifying kmeans, but all of them require this step. 
     * <br>
     * The distance metric used is trained if needed
     * 
     * @param dataSet The set of data points to perform clustering on
     * @param means the initial points to use as the means. Its
     * length is the number of means that will be searched for. 
     * These means will be altered, and should contain deep copies
     * of the points they were drawn from. 
     * @param assignment an empty temp space to store the clustering 
     * classifications. Should be the same length as the number of data points
     * @param exactTotal determines how the objective function (return value) 
     * will be computed. If true, extra work will be done to compute the exact 
     * distance from each data point to its cluster. If false, an upper bound 
     * approximation will be used. 
     * 
     * @return the sum of squares distances from each data point to its closest cluster
     */
    protected double cluster(final DataSet dataSet, final List<Vec> means, final int[] assignment, boolean exactTotal)
    {
        return cluster(dataSet, means, assignment, exactTotal, null);
    }
    
    /*
     * IMPLEMENTATION NOTE: Means are updates as a set of sums via deltas. Deltas are 
     * computed locally withing a thread local object. Then to avoid dropping updates,
     * every thread that was working must apply its deltas itself, as other threads 
     * can not access another's thread locals. 
     */
    
    /**
     *  This is a helper method where the actual cluster is performed. This is because there
     * are multiple strategies for modifying kmeans, but all of them require this step. 
     * <br>
     * The distance metric used is trained if needed
     * 
     * @param dataSet The set of data points to perform clustering on
     * @param means the initial points to use as the means. Its
     * length is the number of means that will be searched for. 
     * These means will be altered, and should contain deep copies
     * of the points they were drawn from. 
     * @param assignment an empty temp space to store the clustering 
     * classifications. Should be the same length as the number of data points
     * @param exactTotal determines how the objective function (return value) 
     * will be computed. If true, extra work will be done to compute the exact 
     * distance from each data point to its cluster. If false, an upper bound 
     * approximation will be used. 
     * @param threadpool the source of threads for parallel computation. If <tt>null</tt>, single threaded execution will occur
     * @return the sum of squares distances from each data point to its closest cluster
     */
    protected double cluster(final DataSet dataSet, final List<Vec> means, final int[] assignment, boolean exactTotal, ExecutorService threadpool)
    {
        try
        {
            //Make our means dense
            for(int i = 0; i < means.size(); i++)
                if(means.get(i).isSparse())
                    means.set(i, new DenseVector(means.get(i)));
            /**
             * K clusters
             */
            final int k = means.size();
            /**
             * N data points
             */
            final int N = dataSet.getSampleSize();
            final int D = dataSet.getNumNumericalVars();
            if(N < k)//Not enough points
                throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");

            TrainableDistanceMetric.trainIfNeeded(dm, dataSet);
            
            final double[][] lowerBound = new double[N][k];
            final double[] upperBound = new double[N];

            /**
             * Distances between centroid i and all other centroids
             */
            final double[][] centroidSelfDistances = new double[k][k];
            final double[] sC = new double[k];
            calculateCentroidDistances(k, centroidSelfDistances, means, sC, null);
            final AtomicLongArray meanCount = new AtomicLongArray(k);
            Vec[] oldMeans = new Vec[k];//The means fromt he current step are needed when computing the new means
            final Vec[] meanSums = new Vec[k];
            for (int i = 0; i < k; i++)
            {
                oldMeans[i] = means.get(i).clone();//This way the new vectors are of the same implementation
                meanSums[i] = new DenseVector(D);
            }
            
            if(dm instanceof DenseSparseMetric && useDenseSparse)
                dmds = (DenseSparseMetric) dm;
            final double[] meanSummaryConsts = dmds != null ? new double[means.size()] : null;
            
            int atLeast = 2;//Used to performan an extra round (first round does not assign)
            final AtomicBoolean changeOccurred = new AtomicBoolean(true);
            final boolean[] r = new boolean[N];//Default value of a boolean is false, which is what we want
            
            final ThreadLocal<Vec[]> localDeltas = new ThreadLocal<Vec[]>()
            {
                @Override
                protected Vec[] initialValue()
                {
                    Vec[] toRet = new Vec[k];
                    for(int i = 0; i < toRet.length; i++)
                        toRet[i] = new DenseVector(D);
                    return toRet;
                }
            };
            
            if (threadpool == null)
                initialClusterSetUp(k, N, dataSet, means, lowerBound, upperBound, centroidSelfDistances, assignment, meanCount, meanSums);
            else
                initialClusterSetUp(k, N, dataSet, means, lowerBound, upperBound, centroidSelfDistances, assignment, meanCount, meanSums, localDeltas, threadpool);

            final ArrayBlockingQueue<Runnable> runnableList = threadpool == null ? null : new ArrayBlockingQueue<Runnable>(4 * SystemInfo.LogicalCores, false);

            int iterLimit = MaxIterLimit;
            while ((changeOccurred.get() || atLeast > 0) && iterLimit-- >= 0)
            {
                atLeast--;
                changeOccurred.set(false);
                //Step 1 
                calculateCentroidDistances(k, centroidSelfDistances, means, sC, meanSummaryConsts);
                
                final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);

                //Create readers to run jobs 
                if (threadpool != null)
                    for (int i = 0; i < SystemInfo.LogicalCores; i++)
                        threadpool.submit(new RunnableConsumer(runnableList));

                //Step 2 / 3
                for (int q = 0; q < N; q++)
                {
                    //Step 2, skip those that u(v) < s(c(v))
                    if (upperBound[q] <= sC[assignment[q]])
                        continue;
                    
                    final Vec v = dataSet.getDataPoint(q).getNumericalValues();
                        
                    if(threadpool == null)
                    {
                        for (int c = 0; c < k; c++)
                            if (c != assignment[q] && upperBound[q] > lowerBound[q][c] && upperBound[q] > centroidSelfDistances[assignment[q]][c] * 0.5)
                            {
                                step3aBoundsUpdate(r, q, v, means, assignment, upperBound, lowerBound, meanSummaryConsts);
                                step3bUpdate(upperBound, q, lowerBound, c, centroidSelfDistances, assignment, v, means, localDeltas, meanCount, changeOccurred, meanSummaryConsts);
                            }
                    }
                    else
                    {
                        final int qq = q;
                        Runnable run = new Runnable() 
                        {
                            @Override
                            public void run()
                            {
                                for (int c = 0; c < k; c++)
                                    if (c != assignment[qq] && upperBound[qq] > lowerBound[qq][c] && upperBound[qq] > centroidSelfDistances[assignment[qq]][c] * 0.5)
                                    {
                                        step3aBoundsUpdate(r, qq, v, means, assignment, upperBound, lowerBound, meanSummaryConsts);
                                        step3bUpdate(upperBound, qq, lowerBound, c, centroidSelfDistances, assignment, v, means, localDeltas, meanCount,  changeOccurred, meanSummaryConsts);
                                    }
                            }
                        };
                        runnableList.put(run);
                    }
                }
                
                //Step 4
                //Re compute centroids
                for (int i = 0; i < k; i++)
                    means.get(i).copyTo(oldMeans[i]);
              
                if (threadpool != null)
                {
                    //Pop extra off
                    for (int i = 0; i < SystemInfo.LogicalCores; i++)
                        runnableList.put(new PoisonRunnable(new Runnable()
                        {
                            @Override
                            public void run()
                            {
                                step4UpdateCentroids(meanSums, localDeltas);

                                latch.countDown();
                            }
                        }));

                    //Still need to wait for everyone to finish, some threasd may still be easting jobs from the que 
                    try
                    {
                        latch.await();
                    }
                    catch (InterruptedException ex)
                    {
                        throw new ClusterFailureException("Clustering failed");
                    }
                }
                else
                    step4UpdateCentroids(meanSums, localDeltas);
                
                //normalize 
                for (int i = 0; i < k; i++)
                {
                    meanSums[i].copyTo(means.get(i));
                    long count = meanCount.get(i);
                    if(count == 0)
                        means.get(i).zeroOut();
                    else
                        means.get(i).mutableDivide(meanCount.get(i));
                }
                
                step5_6_distanceMovedBoundsUpdate(k, oldMeans, means, N, lowerBound, upperBound, assignment, r);
            }

            double totalDistance = 0.0;

            if (exactTotal == true)
                for (int i = 0; i < N; i++)
                    totalDistance += Math.pow(dm.dist(dataSet.getDataPoint(i).getNumericalValues(), means.get(assignment[i])), 2);
            else
                for (int i = 0; i < N; i++)
                    totalDistance += Math.pow(upperBound[i], 2);

            return totalDistance;
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(KMeans.class.getName()).log(Level.SEVERE, null, ex);
        }
        return Double.MAX_VALUE;
    }

    private void initialClusterSetUp(final int k, final int N, final DataSet dataSet, final List<Vec> means, final double[][] lowerBound, 
            final double[] upperBound, final double[][] centroidSelfDistances, final int[] assignment, AtomicLongArray meanCount,
            final Vec[] meanSums)
    {
        //Skip markers
        final boolean[] skip = new boolean[k];
        for (int q = 0; q < N; q++)
        {
            Vec v = dataSet.getDataPoint(q).getNumericalValues();
            double minDistance = Double.MAX_VALUE;
            int index = -1;
            //Default value is false, we cant skip anything yet
            Arrays.fill(skip, false);
            for (int i = 0; i < k; i++)
            {
                if (skip[i])
                    continue;
                double d = dm.dist(v, means.get(i));
                lowerBound[q][i] = d;

                if (d < minDistance)
                {
                    minDistance = upperBound[q] = d;
                    index = i;
                    //We now have some information, use lemma 1 to see if we can skip anything
                    for (int z = i + 1; z < k; z++)
                        if (centroidSelfDistances[i][z] >= 2 * d)
                            skip[z] = true;
                }
            }

            assignment[q] = index;
            meanCount.incrementAndGet(index);
            meanSums[index].mutableAdd(v);
        }
    }
    
    private void initialClusterSetUp(final int k, final int N, final DataSet dataSet, final List<Vec> means, final double[][] lowerBound, 
            final double[] upperBound, final double[][] centroidSelfDistances, final int[] assignment, final AtomicLongArray meanCount, 
            final Vec[] meanSums, final ThreadLocal<Vec[]> localDeltas, ExecutorService threadpool)
    {
        final int blockSize = N / SystemInfo.LogicalCores;
        int extra = N % SystemInfo.LogicalCores;
        int pos = 0;
        final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
        while (pos < N)
        {
            final int from = pos;
            final int to = pos + blockSize + (extra-- > 0 ? 1 : 0);
            pos = to;

            threadpool.submit(new Runnable()
            {

                @Override
                public void run()
                {
                    Vec[] deltas = localDeltas.get();
                    final boolean[] skip = new boolean[k];
                    for (int q = from; q < to; q++)
                    {
                        Vec v = dataSet.getDataPoint(q).getNumericalValues();
                        double minDistance = Double.MAX_VALUE;
                        int index = -1;
                        //Default value is false, we cant skip anything yet
                        Arrays.fill(skip, false);
                        for (int i = 0; i < k; i++)
                        {
                            if (skip[i])
                                continue;
                            double d = dm.dist(v, means.get(i));
                            lowerBound[q][i] = d;

                            if (d < minDistance)
                            {
                                minDistance = upperBound[q] = d;
                                index = i;
                                //We now have some information, use lemma 1 to see if we can skip anything
                                for (int z = i + 1; z < k; z++)
                                    if (centroidSelfDistances[i][z] >= 2 * d)
                                        skip[z] = true;
                            }
                        }

                        assignment[q] = index;
                        meanCount.incrementAndGet(index);
                        deltas[index].mutableAdd(v);
                    }
                    for (int i = 0; i < deltas.length; i++)
                    {
                        synchronized (meanSums[i])
                        {
                            meanSums[i].mutableAdd(deltas[i]);
                        }
                        deltas[i].zeroOut();
                    }

                    latch.countDown();
                }
            });
        }
        while(pos++ < SystemInfo.LogicalCores)
            latch.countDown();
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(KMeans.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void step4UpdateCentroids(Vec[] meanSums, ThreadLocal<Vec[]> localDeltas)
    {
        Vec[] deltas = localDeltas.get();
        for(int i = 0; i < deltas.length; i++)
        {
            if(deltas[i].nnz() == 0)
                continue;
            synchronized(meanSums[i])
            {
                meanSums[i].mutableAdd(deltas[i]);
            }
            deltas[i].zeroOut();
        }
    }

    private void step5_6_distanceMovedBoundsUpdate(final int k, Vec[] oldMeans, final List<Vec> means, final int N, final double[][] lowerBound, final double[] upperBound, final int[] assignment, final boolean[] r)
    {
        double[] distancesMoved = new double[k];
        for(int i = 0; i < k; i++)
            distancesMoved[i] = dm.dist(oldMeans[i], means.get(i));

        //Step 5
        for(int c = 0; c < k; c++)
            for(int q = 0; q < N; q++)
                lowerBound[q][c] = Math.max(lowerBound[q][c] - distancesMoved[c], 0);

        //Step 6
        for(int q = 0; q < N; q++)
        {
            upperBound[q] +=  distancesMoved[assignment[q]];
            r[q] = true; 
        }
    }

    private void step3aBoundsUpdate(boolean[] r, int q, Vec v, final List<Vec> means, final int[] assignment, double[] upperBound, double[][] lowerBound, double[] meanSummaryConsts)
    {
        //3(a)
        if (r[q])
        {
            r[q] = false;
            double d;
            int meanIndx = assignment[q];
            if(dmds == null)
                d = dm.dist(v, means.get(meanIndx));
            else
                d = dmds.dist(meanSummaryConsts[meanIndx], means.get(meanIndx), v);
            lowerBound[q][meanIndx] = d;///Not sure if this is supposed to be here
            upperBound[q] = d;
        }
    }

    private void step3bUpdate(double[] upperBound, final int q, double[][] lowerBound, final int c, double[][] centroidSelfDistances, 
            final int[] assignment, Vec v, final List<Vec> means, final ThreadLocal<Vec[]> localDeltas, AtomicLongArray meanCount, 
            final AtomicBoolean changeOccurred, double[] meanSummaryConsts)
    {
        //3(b)
        if (upperBound[q] > lowerBound[q][c] || upperBound[q] > centroidSelfDistances[assignment[q]][c] / 2)
        {
            double d;
            if(dmds == null)
                d = dm.dist(v, means.get(c));
            else
                d = dmds.dist(meanSummaryConsts[c], means.get(c), v);
            lowerBound[q][c] = d;
            if (d < upperBound[q])
            {
                Vec[] deltas = localDeltas.get();
                deltas[assignment[q]].mutableSubtract(v);
                meanCount.decrementAndGet(assignment[q]);
                
                deltas[c].mutableAdd(v);
                meanCount.incrementAndGet(c);
                
                assignment[q] = c;
                upperBound[q] = d;
                
                changeOccurred.set(true);
            }
        }
    }

    private void calculateCentroidDistances(final int k, double[][] centroidSelfDistances, final List<Vec> means, double[] sC, double[] meanSummaryConsts)
    {
        for (int i = 0; i < k; i++)
        {
            double sCmin = Double.MAX_VALUE;
            for (int z = 0; z < k; z++)
            {
                if (i == z)//Distance to self is zero
                    centroidSelfDistances[i][z] = 0;
                else
                {
                    centroidSelfDistances[i][z] = dm.dist(means.get(i), means.get(z));
                    sCmin = Math.min(sCmin, centroidSelfDistances[i][z]);
                }
            }
            sC[i] = sCmin / 2.0;
            if(meanSummaryConsts != null)
                meanSummaryConsts[i] = dmds.getVectorConstant(means.get(i));
        }
    }

    static protected List<List<DataPoint>> getListOfLists(int k)
    {
        List<List<DataPoint>> ks = new ArrayList<List<DataPoint>>(k);
        for(int i = 0; i < k; i++)
            ks.add(new ArrayList<DataPoint>());
        return ks;
    }

    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()/2), designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()/2), threadpool, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, ExecutorService threadpool, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        if(dataSet.getSampleSize() < clusters)
            throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");
        
        means = selectIntialPoints(dataSet, clusters, dm, rand, seedSelection, threadpool);
        cluster(dataSet, means, designations, false, threadpool);
        if(!storeMeans)
            means = null;
        return designations;
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        if(dataSet.getSampleSize() < clusters)
            throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");
        
        means = selectIntialPoints(dataSet, clusters, dm, rand, seedSelection);
        cluster(dataSet, means, designations, false);
        if(!storeMeans)
            means = null;
        
        return designations;
    }
    
    //We use the object itself to return the k 
    private class ClusterWorker implements Runnable
    {
        private DataSet dataSet;
        private int k;
        int[] clusterIDs;
        private Random rand;
        private volatile double result = -1;
        private volatile BlockingQueue<ClusterWorker> putSelf;


        public ClusterWorker(DataSet dataSet, int k, BlockingQueue<ClusterWorker> que)
        {
            this.dataSet = dataSet;
            this.k = k;
            this.putSelf = que;
            clusterIDs = new int[dataSet.getSampleSize()];
            rand = new Random();
        }

        public ClusterWorker(DataSet dataSet, BlockingQueue<ClusterWorker> que)
        {
            this(dataSet, 2, que);
        }

        public void setK(int k)
        {
            this.k = k;
        }

        public int getK()
        {
            return k;
        }

        public double getResult()
        {
            return result;
        }

        public void run()
        {
            result = cluster(dataSet, selectIntialPoints(dataSet, k, dm, rand, seedSelection), clusterIDs, true);
            putSelf.add(this);
        }
        
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
        if(dataSet.getSampleSize() < highK)
            throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");
        
        double[] totDistances = new double[highK-lowK+1];
        
        BlockingQueue<ClusterWorker> workerQue = new ArrayBlockingQueue<ClusterWorker>(SystemInfo.LogicalCores);
        for(int i = 0; i < SystemInfo.LogicalCores; i++)
            workerQue.add(new ClusterWorker(dataSet, workerQue));
        
        int k = lowK;
        int received = 0;
        while(received < totDistances.length)
        {
            try
            {
                ClusterWorker worker = workerQue.take();
                if(worker.getResult() != -1)//-1 means not really in use
                {
                    totDistances[worker.getK() - lowK] = worker.getResult();
                    received++;
                }
                if(k <= highK)
                {
                    worker.setK(k++);
                    threadpool.submit(worker);
                }
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(PAM.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        //Now we process the distance changes
        /**
         * Keep track of the changes
         */
        OnLineStatistics stats = new OnLineStatistics();
        
        double maxChange = Double.MIN_VALUE;
        int maxChangeK = lowK;
        
        for(int i = 1; i < totDistances.length; i++)
        {
            double change = Math.abs(totDistances[i] - totDistances[i-1]);
            stats.add(change);
            if (change > maxChange)
            {
                maxChange = change;
                maxChangeK = i+lowK;
            }
        }
        
        if(maxChange < stats.getStandardDeviation()*2+stats.getMean())
            maxChangeK = lowK;        
        
        return cluster(dataSet, maxChangeK, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        /**
         * Stores the cluster ids associated with each data point
         */
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        if(dataSet.getSampleSize() < highK)
            throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");

        List<List<DataPoint>> ks = new ArrayList<List<DataPoint>>(highK);
        for (int i = 0; i < ks.size(); i++)
            ks.add(new ArrayList<DataPoint>());
        
        double[] totDistances = new double[highK-lowK+1];
        /**
         * Keep track of the changes
         */
        OnLineStatistics stats = new OnLineStatistics();
        
        double maxChange = Double.MIN_VALUE;
        int maxChangeK = lowK;

        for(int i = lowK; i <= highK; i++)
        {
            double totDist = cluster(dataSet, selectIntialPoints(dataSet, i, dm, rand, seedSelection), designations, true);
            totDistances[i-lowK] = totDist;
            if(i > lowK)
            {
                double change = Math.abs(totDist-totDistances[i-lowK-1]);
                stats.add(change);
                if(change > maxChange)
                {
                    maxChange = change;
                    maxChangeK = i;
                }
            }
        }
        
        double changeMean = stats.getMean();
        double changeDev = stats.getStandardDeviation();
        
        //If we havent had any huge drops in total distance, assume that there are onlu to clusts
        if(maxChange < changeDev*2+changeMean)
            maxChangeK = lowK;
        else
        {
            double tmp;
            for(int i = 1; i < totDistances.length; i++)
            {
                if( (tmp = Math.abs(totDistances[i]-totDistances[i-1])) < maxChange )
                {
                    maxChange = tmp;
                    maxChangeK = i+lowK;
                    break;
                }
            }
        }
        
        
        return cluster(dataSet, maxChangeK, designations);
    }
}
