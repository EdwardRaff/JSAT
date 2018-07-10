
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
import jsat.utils.concurrent.ParallelUtils;
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
    protected double cluster(final DataSet dataSet, List<Double> accelCache, final int k, final List<Vec> means, final int[] assignment, boolean exactTotal, boolean parallel, boolean returnError, Vec dataPointWeights)
    {
        try
        {
            /**
             * N data points
             */
            final int N = dataSet.size();
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
            final List<List<Double>> meanQIs = new ArrayList<>(k);
            //done a wonky way b/c we want this as a final object for convinence, otherwise we may be stuck with null accel when we dont need to be
            if(accelCache == null)
                distAccelCache = dm.getAccelerationCache(X, parallel);
            else
                distAccelCache = accelCache;
            
            if(means.size() != k)
            {
                means.clear();
                means.addAll(selectIntialPoints(dataSet, k, dm, distAccelCache, rand, seedSelection, parallel));
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
            calculateCentroidDistances(k, centroidSelfDistances, means, sC, null, parallel);
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
            
            initialClusterSetUp(k, N, X, means, lowerBound, upperBound, centroidSelfDistances, assignment, meanCount, meanSums, distAccelCache, meanQIs, localDeltas, parallel, W);

            int iterLimit = MaxIterLimit;
            while ((changeOccurred.get() || atLeast > 0) && iterLimit-- >= 0)
            {
                atLeast--;
                changeOccurred.set(false);
                //Step 1 
                if(iterLimit < MaxIterLimit-1)//we already did this on before iteration
                    calculateCentroidDistances(k, centroidSelfDistances, means, sC, meanSummaryConsts, parallel);
                
                final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);

                //Step 2 / 3
                ParallelUtils.run(parallel, N, (q)->
                {
                    //Step 2, skip those that u(v) < s(c(v))
                    if (upperBound[q] <= sC[assignment[q]])
                        return;

                    final Vec v = X.get(q);

                    for (int c = 0; c < k; c++)
                        if (c != assignment[q] && upperBound[q] > lowerBound[q][c] && upperBound[q] > centroidSelfDistances[assignment[q]][c] * 0.5)
                        {
                            step3aBoundsUpdate(X, r, q, v, means, assignment, upperBound, lowerBound, meanSummaryConsts, distAccelCache, meanQIs);
                            step3bUpdate(X, upperBound, q, lowerBound, c, centroidSelfDistances, assignment, v, means, localDeltas, meanCount, changeOccurred, meanSummaryConsts, distAccelCache, meanQIs, W);
                        }
                    step4UpdateCentroids(meanSums, localDeltas);
                });
                
                step5_6_distanceMovedBoundsUpdate(k, oldMeans, means, meanSums, meanCount, N, lowerBound, upperBound, assignment, r, meanQIs, parallel);
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
    
    private void initialClusterSetUp(final int k, final int N, final List<Vec> dataSet, final List<Vec> means, final double[][] lowerBound, 
            final double[] upperBound, final double[][] centroidSelfDistances, final int[] assignment, final AtomicDoubleArray meanCount, 
            final Vec[] meanSums, final List<Double> distAccelCache, final List<List<Double>> meanQIs, 
            final ThreadLocal<Vec[]> localDeltas, boolean parallel, final Vec W)
    {
        ParallelUtils.run(parallel, N, (from, to)->
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
        });
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
            final int[] assignment, final boolean[] r, final List<List<Double>> meanQIs, boolean parallel)
    {
        final double[] distancesMoved = new double[k];
        
        ParallelUtils.run(parallel, k, (i)->
        {
            //Re compute centroids
            means.get(i).copyTo(oldMeans[i]);

            //normalize 
            meanSums[i].copyTo(means.get(i));
            double count = meanCount.get(i);
            if (count <= 1e-14)
                means.get(i).zeroOut();
            else
                means.get(i).mutableDivide(meanCount.get(i));

            distancesMoved[i] = dm.dist(oldMeans[i], means.get(i));

            if(dm.supportsAcceleration())
                meanQIs.set(i, dm.getQueryInfo(means.get(i)));

            //Step 5
            for (int q = 0; q < N; q++)
                lowerBound[q][i] = Math.max(lowerBound[q][i] - distancesMoved[i], 0);
        });
        //Step 6
        ParallelUtils.run(parallel, N, (start, end) ->
        {
            for(int q = start; q < end; q++)
            {
                upperBound[q] +=  distancesMoved[assignment[q]];
                r[q] = true;
            }
        });
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

    private void calculateCentroidDistances(final int k, final double[][] centroidSelfDistances, final List<Vec> means, final double[] sC, final double[] meanSummaryConsts, boolean parallel)
    {
        final List<Double> meanAccelCache = dm.supportsAcceleration() ? dm.getAccelerationCache(means) : null;
        
        //TODO can improve parallel performance for when k ~<= # cores
        ParallelUtils.run(parallel, k, (i)->
        {
            for (int z = i + 1; z < k; z++)
                centroidSelfDistances[z][i] = centroidSelfDistances[i][z] = dm.dist(i, z, means, meanAccelCache);;

            if (meanSummaryConsts != null)
                meanSummaryConsts[i] = dmds.getVectorConstant(means.get(i));
        });
        
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
