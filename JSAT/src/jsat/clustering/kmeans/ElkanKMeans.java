
package jsat.clustering.kmeans;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.clustering.ClusterFailureException;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import static jsat.clustering.SeedSelectionMethods.selectIntialPoints;
import jsat.linear.ConstantVector;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.*;
import jsat.utils.*;
import jsat.utils.concurrent.AtomicDoubleArray;
import jsat.utils.random.RandomUtil;

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
public class ElkanKMeans extends KMeans
{

    private static final long serialVersionUID = -1629432283103273051L;

    private DenseSparseMetric dmds;

    private boolean useDenseSparse = false;
    
    /**
     * Creates a new KMeans instance. 
     * @param dm the distance metric to use, must support {@link DistanceMetric#isSubadditive() }. 
     * @param rand the random number generator to use during seed selection
     * @param seedSelection the method of seed selection to use
     */
    public ElkanKMeans(DistanceMetric dm, Random rand, SeedSelection seedSelection)
    {
        super(dm, seedSelection, rand);
        if(!dm.isSubadditive())
            throw new ClusterFailureException("KMeans implementation requires the triangle inequality");
    }

    /**
     * Creates a new KMeans instance
     * @param dm the distance metric to use, must support {@link DistanceMetric#isSubadditive() }.  
     * @param rand the random number generator to use during seed selection
     */
    public ElkanKMeans(DistanceMetric dm, Random rand)
    {
        this(dm, rand, DEFAULT_SEED_SELECTION);
    }

    /**
     * Creates a new KMeans instance
     * @param dm the distance metric to use, must support {@link DistanceMetric#isSubadditive() }.  
     */
    public ElkanKMeans(DistanceMetric dm)
    {
        this(dm, RandomUtil.getRandom());
    }

    /**
     * Creates a new KMeans instance. The {@link EuclideanDistance} will be used by default. 
     */
    public ElkanKMeans()
    {
        this(new EuclideanDistance());
    }

    public ElkanKMeans(ElkanKMeans toCopy)
    {
        super(toCopy);
        if(toCopy.dmds != null)
            this.dmds = (DenseSparseMetric) toCopy.dmds.clone();
        this.useDenseSparse = toCopy.useDenseSparse;
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
    
    /*
     * IMPLEMENTATION NOTE: Means are updates as a set of sums via deltas. Deltas are 
     * computed locally withing a thread local object. Then to avoid dropping updates,
     * every thread that was working must apply its deltas itself, as other threads 
     * can not access another's thread locals. 
     */
    
    @Override
    protected double cluster(final DataSet dataSet, List<Double> accelCache, final int k, final List<Vec> means, final int[] assignment, boolean exactTotal, ExecutorService threadpool, boolean returnError, Vec dataPointWeights)
    {
        try
        {
            /**
             * N data points
             */
            final int N = dataSet.getSampleSize();
            final int D = dataSet.getNumNumericalVars();
            if(N < k)//Not enough points
                throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");
            /**
             * Weights for each data point
             */
            final Vec W;
            if(dataPointWeights == null)
                W = dataSet.getDataWeights();
            else
                W = dataPointWeights;

            TrainableDistanceMetric.trainIfNeeded(dm, dataSet);
            final List<Vec> X = dataSet.getDataVectors();
            
            //Distance computation acceleration
            final List<Double> distAccelCache;
            final List<List<Double>> meanQIs = new ArrayList<List<Double>>(k);;
            //done a wonky way b/c we want this as a final object for convinence, otherwise we may be stuck with null accel when we dont need to be
            if(accelCache == null)
            {
                if(threadpool == null || threadpool instanceof FakeExecutor)
                    distAccelCache = dm.getAccelerationCache(X);
                else
                    distAccelCache = dm.getAccelerationCache(X, threadpool);
            }
            else
                distAccelCache = accelCache;
            
            if(means.size() != k)
            {
                means.clear();
                if(threadpool == null || threadpool instanceof FakeExecutor)
                    means.addAll(selectIntialPoints(dataSet, k, dm, distAccelCache, rand, seedSelection));
                else
                    means.addAll(selectIntialPoints(dataSet, k, dm, distAccelCache, rand, seedSelection, threadpool));
            }
            
            //Make our means dense
            for(int i = 0; i < means.size(); i++)
                if(means.get(i).isSparse())
                    means.set(i, new DenseVector(means.get(i)));
            
            final double[][] lowerBound = new double[N][k];
            final double[] upperBound = new double[N];

            /**
             * Distances between centroid i and all other centroids
             */
            final double[][] centroidSelfDistances = new double[k][k];
            final double[] sC = new double[k];
            calculateCentroidDistances(k, centroidSelfDistances, means, sC, null, threadpool);
            final AtomicDoubleArray meanCount = new AtomicDoubleArray(k);
            Vec[] oldMeans = new Vec[k];//The means fromt he current step are needed when computing the new means
            final Vec[] meanSums = new Vec[k];
            for (int i = 0; i < k; i++)
            {
                oldMeans[i] = means.get(i).clone();//This way the new vectors are of the same implementation
                if(dm.supportsAcceleration())
                    meanQIs.add(dm.getQueryInfo(means.get(i)));
                else
                    meanQIs.add(Collections.EMPTY_LIST);//Avoid null pointers
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
                initialClusterSetUp(k, N, X, means, lowerBound, upperBound, centroidSelfDistances, assignment, meanCount, meanSums, distAccelCache, meanQIs, W);
            else
                initialClusterSetUp(k, N, X, means, lowerBound, upperBound, centroidSelfDistances, assignment, meanCount, meanSums, distAccelCache, meanQIs, localDeltas, threadpool, W);

            int iterLimit = MaxIterLimit;
            while ((changeOccurred.get() || atLeast > 0) && iterLimit-- >= 0)
            {
                atLeast--;
                changeOccurred.set(false);
                //Step 1 
                if(iterLimit < MaxIterLimit-1)//we already did this on before iteration
                    calculateCentroidDistances(k, centroidSelfDistances, means, sC, meanSummaryConsts, threadpool);
                
                final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);

                //Step 2 / 3
                if (threadpool == null)
                {
                    for (int q = 0; q < N; q++)
                    {
                        //Step 2, skip those that u(v) < s(c(v))
                        if (upperBound[q] <= sC[assignment[q]])
                            continue;

                        final Vec v = X.get(q);

                        for (int c = 0; c < k; c++)
                            if (c != assignment[q] && upperBound[q] > lowerBound[q][c] && upperBound[q] > centroidSelfDistances[assignment[q]][c] * 0.5)
                            {
                                step3aBoundsUpdate(X, r, q, v, means, assignment, upperBound, lowerBound, meanSummaryConsts, distAccelCache, meanQIs);
                                step3bUpdate(X, upperBound, q, lowerBound, c, centroidSelfDistances, assignment, v, means, localDeltas, meanCount, changeOccurred, meanSummaryConsts, distAccelCache, meanQIs, W);
                            }
                    }
                }
                else
                {
                    for(int id = 0; id < SystemInfo.LogicalCores; id++)
                    {
                        final int ID = id;
                        threadpool.submit(new Runnable() {

                            @Override
                            public void run()
                            {
                                for (int q = ID; q < N; q += SystemInfo.LogicalCores)
                                {
                                    //Step 2, skip those that u(v) < s(c(v))
                                    if (upperBound[q] <= sC[assignment[q]])
                                        continue;

                                    final Vec v = dataSet.getDataPoint(q).getNumericalValues();

                                    for (int c = 0; c < k; c++)
                                        if (c != assignment[q] && upperBound[q] > lowerBound[q][c] && upperBound[q] > centroidSelfDistances[assignment[q]][c] * 0.5)
                                        {
                                            step3aBoundsUpdate(X, r, q, v, means, assignment, upperBound, lowerBound, meanSummaryConsts, distAccelCache, meanQIs);
                                            step3bUpdate(X, upperBound, q, lowerBound, c, centroidSelfDistances, assignment, v, means, localDeltas, meanCount, changeOccurred, meanSummaryConsts, distAccelCache, meanQIs, W);
                                        }
                                }

                                step4UpdateCentroids(meanSums, localDeltas);
                                latch.countDown();
                            }
                        });
                    }
                }
              
                if (threadpool != null)
                {
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
                
                step5_6_distanceMovedBoundsUpdate(k, oldMeans, means, meanSums, meanCount, N, lowerBound, upperBound, assignment, r, meanQIs, threadpool);
            }

            double totalDistance = 0.0;

            if(returnError)
            {
                if(saveCentroidDistance)
                    nearestCentroidDist = new double[N];
                else
                    nearestCentroidDist = null;

                if (exactTotal == true)
                    for (int i = 0; i < N; i++)
                    {
                        double dist = dm.dist(i, means.get(assignment[i]), meanQIs.get(assignment[i]), X, distAccelCache);
                        totalDistance += Math.pow(dist, 2);
                        if(saveCentroidDistance)
                            nearestCentroidDist[i] = dist;
                    }
                else
                    for (int i = 0; i < N; i++)
                    {
                        totalDistance += Math.pow(upperBound[i], 2);
                        if(saveCentroidDistance)
                            nearestCentroidDist[i] = upperBound[i];
                    }
            }
            
            return totalDistance;
        }
        catch (Exception ex)
        {
            Logger.getLogger(ElkanKMeans.class.getName()).log(Level.SEVERE, null, ex);
        }
        return Double.MAX_VALUE;
    }

    private void initialClusterSetUp(final int k, final int N, final List<Vec> dataSet, 
            final List<Vec> means, final double[][] lowerBound, final double[] upperBound, 
            final double[][] centroidSelfDistances, final int[] assignment, 
            AtomicDoubleArray meanCount, final Vec[] meanSums, 
            List<Double> distAccelCache, List<List<Double>> meanQIs, Vec W)
    {
        //Skip markers
        final boolean[] skip = new boolean[k];
        for (int q = 0; q < N; q++)
        {
            Vec v = dataSet.get(q);
            double minDistance = Double.MAX_VALUE;
            int index = -1;
            //Default value is false, we cant skip anything yet
            Arrays.fill(skip, false);
            for (int i = 0; i < k; i++)
            {
                if (skip[i])
                    continue;
                double d = dm.dist(q, means.get(i), meanQIs.get(i), dataSet, distAccelCache);
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
            final double weight = W.get(q);
            meanCount.addAndGet(index, weight);
            meanSums[index].mutableAdd(weight, v);
        }
    }
    
    private void initialClusterSetUp(final int k, final int N, final List<Vec> dataSet, final List<Vec> means, final double[][] lowerBound, 
            final double[] upperBound, final double[][] centroidSelfDistances, final int[] assignment, final AtomicDoubleArray meanCount, 
            final Vec[] meanSums, final List<Double> distAccelCache, final List<List<Double>> meanQIs, 
            final ThreadLocal<Vec[]> localDeltas, ExecutorService threadpool, final Vec W)
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
                        Vec v = dataSet.get(q);
                        double minDistance = Double.MAX_VALUE;
                        int index = -1;
                        //Default value is false, we cant skip anything yet
                        Arrays.fill(skip, false);
                        for (int i = 0; i < k; i++)
                        {
                            if (skip[i])
                                continue;
                            double d = dm.dist(q, means.get(i), meanQIs.get(i), dataSet, distAccelCache);
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
                        final double weight = W.get(q);
                        meanCount.addAndGet(index, weight);
                        deltas[index].mutableAdd(weight, v);
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
            Logger.getLogger(ElkanKMeans.class.getName()).log(Level.SEVERE, null, ex);
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

    private void step5_6_distanceMovedBoundsUpdate(final int k, final Vec[] oldMeans, final List<Vec> means, final Vec[] meanSums, 
            final AtomicDoubleArray meanCount, final int N, final double[][] lowerBound, final double[] upperBound, 
            final int[] assignment, final boolean[] r, final List<List<Double>> meanQIs, ExecutorService threadpool)
    {
        final double[] distancesMoved = new double[k];
        
        if(threadpool != null)
        {
            try
            {
                final CountDownLatch latch1 = new CountDownLatch(k);
                
                //Step 5
                for (int i = 0; i < k; i++)
                {
                    final int c = i;
                    threadpool.submit(new Runnable()
                    {
                        @Override
                        public void run()
                        {
                            means.get(c).copyTo(oldMeans[c]);
                            
                            meanSums[c].copyTo(means.get(c));
                            double count = meanCount.get(c);
                            if (count <= 1e-14)
                                means.get(c).zeroOut();
                            else
                                means.get(c).mutableDivide(meanCount.get(c));
                            
                            distancesMoved[c] = dm.dist(oldMeans[c], means.get(c));
                            
                            if(dm.supportsAcceleration())
                                meanQIs.set(c, dm.getQueryInfo(means.get(c)));
                            
                            for (int q = 0; q < N; q++)
                                lowerBound[q][c] = Math.max(lowerBound[q][c] - distancesMoved[c], 0);
                            latch1.countDown();
                        }
                    });
                }
                latch1.await();
                
                //Step 6
                final CountDownLatch latch2 = new CountDownLatch(SystemInfo.LogicalCores);
                final int blockSize = N/SystemInfo.LogicalCores;
                for(int id = 0; id < SystemInfo.LogicalCores; id++)
                {
                    final int start = id*blockSize;
                    final int end = (id == SystemInfo.LogicalCores-1 ? N : start+blockSize);
                    threadpool.submit(new Runnable() 
                    {
                        @Override
                        public void run()
                        {
                            for(int q = start; q < end; q++)
                            {
                                upperBound[q] +=  distancesMoved[assignment[q]];
                                r[q] = true; 
                            }
                            latch2.countDown();
                        }
                    });
                }
                latch2.await();
                return;
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(ElkanKMeans.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        //Re compute centroids
        for (int i = 0; i < k; i++)
            means.get(i).copyTo(oldMeans[i]);

        //normalize 
        for (int i = 0; i < k; i++)
        {
            meanSums[i].copyTo(means.get(i));
            double count = meanCount.get(i);
            if (count <= 1e-14)
                means.get(i).zeroOut();
            else
                means.get(i).mutableDivide(meanCount.get(i));
        }

        for (int i = 0; i < k; i++)
        {
            distancesMoved[i] = dm.dist(oldMeans[i], means.get(i));

            if (dm.supportsAcceleration())
                meanQIs.set(i, dm.getQueryInfo(means.get(i)));
        }

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

    private void step3aBoundsUpdate(List<Vec> X, boolean[] r, int q, Vec v, final List<Vec> means, final int[] assignment, double[] upperBound, double[][] lowerBound, double[] meanSummaryConsts, List<Double> distAccelCache, List<List<Double>> meanQIs)
    {
        //3(a)
        if (r[q])
        {
            r[q] = false;
            double d;
            int meanIndx = assignment[q];
            if(dmds == null)
                d = dm.dist(q, means.get(meanIndx), meanQIs.get(meanIndx), X, distAccelCache);
            else
                d = dmds.dist(meanSummaryConsts[meanIndx], means.get(meanIndx), v);
            lowerBound[q][meanIndx] = d;///Not sure if this is supposed to be here
            upperBound[q] = d;
        }
    }

    private void step3bUpdate(List<Vec> X, double[] upperBound, final int q, double[][] lowerBound, final int c, double[][] centroidSelfDistances, 
            final int[] assignment, Vec v, final List<Vec> means, final ThreadLocal<Vec[]> localDeltas, AtomicDoubleArray meanCount, 
            final AtomicBoolean changeOccurred, double[] meanSummaryConsts, List<Double> distAccelCache, List<List<Double>> meanQIs,
            final Vec W)
    {
        //3(b)
        if (upperBound[q] > lowerBound[q][c] || upperBound[q] > centroidSelfDistances[assignment[q]][c] / 2)
        {
            double d;
            if(dmds == null)
                d = dm.dist(q, means.get(c), meanQIs.get(c), X, distAccelCache);
            else
                d = dmds.dist(meanSummaryConsts[c], means.get(c), v);
            lowerBound[q][c] = d;
            if (d < upperBound[q])
            {
                Vec[] deltas = localDeltas.get();
                final double weight = W.get(q);
                deltas[assignment[q]].mutableSubtract(weight, v);
                meanCount.addAndGet(assignment[q], -weight);
                
                deltas[c].mutableAdd(weight, v);
                meanCount.addAndGet(c, weight);
                
                assignment[q] = c;
                upperBound[q] = d;
                
                changeOccurred.set(true);
            }
        }
    }

    private void calculateCentroidDistances(final int k, final double[][] centroidSelfDistances, final List<Vec> means, final double[] sC, final double[] meanSummaryConsts, ExecutorService threadpool)
    {
        final List<Double> meanAccelCache = dm.supportsAcceleration() ? dm.getAccelerationCache(means) : null;
        
        if(threadpool != null)
        {
            //# of items in the upper triangle of a matrix excluding diagonal is (1+k)*k/2-k
            int jobs = (1+k)*k/2-k;
            final CountDownLatch latch = new CountDownLatch(jobs);
            for (int i = 0; i < k; i++)
            {
                final int ii = i;
                for (int z = i + 1; z < k; z++)
                {
                    final int zz = z;
                    threadpool.submit(new Runnable()
                    {
                        @Override
                        public void run()
                        {
                            centroidSelfDistances[ii][zz] = dm.dist(ii, zz, means, meanAccelCache);
                            if (meanSummaryConsts != null)
                                meanSummaryConsts[ii] = dmds.getVectorConstant(means.get(ii));
                            latch.countDown();
                        }
                    });
                }
            }
            try
            {
                latch.await();
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(ElkanKMeans.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        else
        {
            for (int i = 0; i < k; i++)
            {
                for (int z = i + 1; z < k; z++)
                    centroidSelfDistances[z][i] = centroidSelfDistances[i][z] = dm.dist(i, z, means, meanAccelCache);;

                if (meanSummaryConsts != null)
                    meanSummaryConsts[i] = dmds.getVectorConstant(means.get(i));
            }
        }
        //final step quickly figure out sCmin
        for (int i = 0; i < k; i++)
        {
            double sCmin = Double.MAX_VALUE;
            for (int z = 0; z < k; z++)
                if (z != i)
                    sCmin = Math.min(sCmin, centroidSelfDistances[i][z]);
            sC[i] = sCmin / 2.0;
        }
    }

    @Override
    public ElkanKMeans clone()
    {
        return new ElkanKMeans(this);
    }
}
