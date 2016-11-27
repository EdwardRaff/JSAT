
package jsat.clustering.kmeans;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.clustering.ClusterFailureException;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.FailedToFitException;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.*;
import jsat.utils.*;
import jsat.utils.concurrent.ParallelUtils;

/**
 * An efficient implementation of the K-Means algorithm. This implementation uses
 * the triangle inequality to accelerate computation while maintaining the exact
 * same solution. This requires that the {@link DistanceMetric} used support 
 * {@link DistanceMetric#isSubadditive() }. <br>
 * <br>
 * See: Elkan, C. (2003). <i>Using the Triangle Inequality to Accelerate 
 * k-Means.</i> In Proceedings of the Twentieth International Conference on 
 * Machine Learning (ICML-2003) (pp. 147–153). AAAI Press.
 * 
 * @author Edward Raff
 */
public class ElkanKernelKMeans extends KernelKMeans
{

    private static final long serialVersionUID = 4998832201379993827L;
    
    /**
     * Distances between centroid i and all other centroids
     */
    private double[][] centroidSelfDistances;
    
    /**
     * un-normalized dot product between centroid i and all other centroids. 
     */
    private double[][] centroidPairDots;

    /**
     * Creates a new Kernel K Means object
     * @param kernel the kernel to use
     */
    public ElkanKernelKMeans(KernelTrick kernel)
    {
        super(kernel);
    }

    public ElkanKernelKMeans(ElkanKernelKMeans toCopy)
    {
        super(toCopy);
    }

    @Override
    public int findClosestCluster(Vec x, List<Double> qi)
    {
        double min = Double.MAX_VALUE;
        int min_indx = -1;
        //Use triangle inequality to prune out clusters!
        boolean[] pruned = new boolean[meanSqrdNorms.length];
        Arrays.fill(pruned, false);
        for(int i  = 0; i < meanSqrdNorms.length; i++)
        {
            if(ownes[i] <= 1e-15 || pruned[i])
                continue;
            
            double dist = distance(x, qi, i);
            if(dist < min)
            {
                min = dist;
                min_indx = i;
            }
            
            //we now know d(x, c_i), see who we can prune based off this
            for(int j = i+1; j < meanSqrdNorms.length; j++)
            {
                if(centroidSelfDistances[i][j] >= 2*dist)
                    pruned[j] = true;
            }
        }
        
        return min_indx;
    }
    
    /**
     * Updates the un-normalized dot product between cluster centers, which is
     * stored in {@link #centroidPairDots}. Using this method avoids redundant
     * calculation, only adjusting for the values that have changed.
     *
     * @param prev_assignments what the previous assignment of data points to
     * cluster center looked like
     * @param new_assignments the new assignment of data points to cluster
     * centers
     * @param ex source of threads for parallel computation. May be null.
     */
    private void update_centroid_pair_dots(final int[] prev_assignments, final int[] new_assignments, ExecutorService ex)
    {
        final int N = X.size();
        
        final int cores_to_use;
        if(ex == null || ex instanceof FakeExecutor)
        {
            ex = new FakeExecutor();
            cores_to_use = 1;
        }
        else
             cores_to_use = SystemInfo.LogicalCores;

        List<Future<double[][]>> futurePairDotChanges = new ArrayList<Future<double[][]>>(cores_to_use);
        for(int id = 0; id < cores_to_use; id++)
        {
            final int ID = id;
            Future<double[][]> future = ex.submit(new Callable<double[][]>()
            {
                @Override
                public double[][] call() throws Exception
                {
                    double[][] localChanges = new double[centroidPairDots.length][centroidPairDots.length];
                    
                    for(int i = ID; i < N; i+=cores_to_use)
                    {
                        final double w_i = W.get(i);
                        int old_c_i = prev_assignments[i];
                        int new_c_i = new_assignments[i];

                        for(int j = i; j < N; j++)
                        {
                            int old_c_j = prev_assignments[j];
                            int new_c_j = new_assignments[j];

                            if(old_c_i == new_c_i && old_c_j == new_c_j)//no class changes
                                continue;//so we can skip it

                            final double w_j = W.get(j);
                            double K_ij = w_i * w_j * kernel.eval(i, j, X, accel);

                            if(old_c_i >= 0 && old_c_j >= 0)
                            {
                                localChanges[old_c_i][old_c_j] -= K_ij;
                                localChanges[old_c_j][old_c_i] -= K_ij;
                            }

                            localChanges[new_c_i][new_c_j] += K_ij;
                            localChanges[new_c_j][new_c_i] += K_ij;
                        }
                    }
                    
                    return localChanges;
                }
            });
            futurePairDotChanges.add(future);
        }
        
        for(Future<double[][]> f : futurePairDotChanges)
        {
            try
            {
                double[][] changes = f.get();
                for(int i = 0; i < changes.length; i++)
                    for(int j = 0; j < changes[i].length; j++)
                        centroidPairDots[i][j] += changes[i][j];
            }
            catch (InterruptedException ex1)
            {
                throw new FailedToFitException(ex1);
            }
            catch (ExecutionException ex1)
            {
                throw new FailedToFitException(ex1);
            }
        }
    }
    
    /**
     *  This is a helper method where the actual cluster is performed. This is because there
     * are multiple strategies for modifying kmeans, but all of them require this step. 
     * <br>
     * The distance metric used is trained if needed
     * 
     * @param dataSet The set of data points to perform clustering on
     * @param k the number of clusters 
     * @param assignment an empty temp space to store the clustering 
     * classifications. Should be the same length as the number of data points
     * @param exactTotal determines how the objective function (return value) 
     * will be computed. If true, extra work will be done to compute the exact 
     * distance from each data point to its cluster. If false, an upper bound 
     * approximation will be used. 
     * @param threadpool the source of threads for parallel computation. If <tt>null</tt>, single threaded execution will occur
     * @return the sum of squares distances from each data point to its closest cluster
     */
    protected double cluster(final DataSet dataSet, final int k, final int[] assignment, boolean exactTotal, ExecutorService threadpool)
    {
        try
        {
            /**
             * N data points
             */
            final int N = dataSet.getSampleSize();
            if(N < k)//Not enough points
                throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");
            
            X = dataSet.getDataVectors();
            setup(k, assignment, dataSet.getDataWeights());
                        
            final double[][] lowerBound = new double[N][k];
            final double[] upperBound = new double[N];

            /**
             * Distances between centroid i and all other centroids
             */
            centroidSelfDistances = new double[k][k];
            centroidPairDots = new double[k][k];
            final double[] sC = new double[k];
            calculateCentroidDistances(k, centroidSelfDistances, sC, assignment, null, threadpool);
            final int[] prev_assignment = new int[N];
                    
            int atLeast = 2;//Used to performan an extra round (first round does not assign)
            final AtomicBoolean changeOccurred = new AtomicBoolean(true);
            final boolean[] r = new boolean[N];//Default value of a boolean is false, which is what we want

            
            if (threadpool == null)
                initialClusterSetUp(k, N, lowerBound, upperBound, centroidSelfDistances, assignment);
            else
                initialClusterSetUp(k, N, lowerBound, upperBound, centroidSelfDistances, assignment, threadpool);

            int iterLimit = maximumIterations;
            while ((changeOccurred.get() || atLeast > 0) && iterLimit-- >= 0)
            {
                atLeast--;
                changeOccurred.set(false);
                //Step 1 
                if(iterLimit < maximumIterations-1)//we already did this on before iteration
                    calculateCentroidDistances(k, centroidSelfDistances, sC, assignment, prev_assignment, threadpool);
                
                //Save current assignent for update next iteration
                System.arraycopy(assignment, 0, prev_assignment, 0, assignment.length);
                
                final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);

                //Step 2 / 3
                if (threadpool == null)
                {
                    for (int q = 0; q < N; q++)
                    {
                        //Step 2, skip those that u(v) < s(c(v))
                        if (upperBound[q] <= sC[assignment[q]])
                            continue;

                        for (int c = 0; c < k; c++)
                            if (c != assignment[q] && upperBound[q] > lowerBound[q][c] && upperBound[q] > centroidSelfDistances[assignment[q]][c] * 0.5)
                            {
                                step3aBoundsUpdate(r, q, assignment, upperBound, lowerBound);
                                step3bUpdate(upperBound, q, lowerBound, c, centroidSelfDistances, assignment, changeOccurred);
                            }
                    }
                }
                else
                {
                    for(int id = 0; id < SystemInfo.LogicalCores; id++)
                    {
                        final int ID = id;
                        threadpool.submit(new Runnable() 
                        {

                            @Override
                            public void run()
                            {
                                for (int q = ID; q < N; q += SystemInfo.LogicalCores)
                                {
                                    //Step 2, skip those that u(v) < s(c(v))
                                    if (upperBound[q] <= sC[assignment[q]])
                                        continue;

                                    for (int c = 0; c < k; c++)
                                        if (c != assignment[q] && upperBound[q] > lowerBound[q][c] && upperBound[q] > centroidSelfDistances[assignment[q]][c] * 0.5)
                                        {
                                            step3aBoundsUpdate(r, q, assignment, upperBound, lowerBound);
                                            step3bUpdate(upperBound, q, lowerBound, c, centroidSelfDistances, assignment, changeOccurred);
                                        }
                                }

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
                
                int moved = step4_5_6_distanceMovedBoundsUpdate(k, N, lowerBound, upperBound, assignment, r, threadpool);
            }

            double totalDistance = 0.0;

            //TODO do I realy want to keep this around for the kernel version?
            if (exactTotal == true)
                for (int i = 0; i < N; i++)
                    totalDistance += Math.pow(upperBound[i], 2);//TODO this isn't exact any more
            else
                for (int i = 0; i < N; i++)
                    totalDistance += Math.pow(upperBound[i], 2);

            return totalDistance;
        }
        catch (Exception ex)
        {
            Logger.getLogger(ElkanKernelKMeans.class.getName()).log(Level.SEVERE, null, ex);
        }
        return Double.MAX_VALUE;
    }

    private void initialClusterSetUp(final int k, final int N, final double[][] lowerBound, 
            final double[] upperBound, final double[][] centroidSelfDistances, final int[] assignment)
    {
        //Skip markers
        final boolean[] skip = new boolean[k];
        for (int q = 0; q < N; q++)
        {
            double minDistance = Double.MAX_VALUE;
            int index = -1;
            //Default value is false, we cant skip anything yet
            Arrays.fill(skip, false);
            for (int i = 0; i < k; i++)
            {
                if (skip[i])
                    continue;
                double d = distance(q, i, assignment);
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

            newDesignations[q] = index;
        }
    }
    
    private void initialClusterSetUp(final int k, final int N, final double[][] lowerBound, 
            final double[] upperBound, final double[][] centroidSelfDistances, final int[] assignment,
            ExecutorService threadpool)
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
                    final boolean[] skip = new boolean[k];
                    
                    for (int q = from; q < to; q++)
                    {
                        double minDistance = Double.MAX_VALUE;
                        int index = -1;
                        //Default value is false, we cant skip anything yet
                        Arrays.fill(skip, false);
                        for (int i = 0; i < k; i++)
                        {
                            if (skip[i])
                                continue;
                            double d = distance(q, i, assignment);
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

                        newDesignations[q] = index;
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
            Logger.getLogger(ElkanKernelKMeans.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private int step4_5_6_distanceMovedBoundsUpdate(final int k, final int N, 
            final double[][] lowerBound, final double[] upperBound, 
            final int[] assignment, final boolean[] r, ExecutorService threadpool)
    {
        final double[] distancesMoved = new double[k];
        //copy the originonal sqrd norms b/c we need them to compute the distance means moved
        final double[] oldSqrdNorms = new double[meanSqrdNorms.length];
        for(int i = 0; i < meanSqrdNorms.length; i++)
            oldSqrdNorms[i] = meanSqrdNorms[i]*normConsts[i];
        int moved = 0;
        if(threadpool != null)
        {
            try
            {
                //first we need to do assignment movement updated, otherwise the sqrdNorms will be wrong and we will get incorrect values for cluster movemnt
                List<Future<Integer>> futureChanges = new ArrayList<Future<Integer>>(SystemInfo.LogicalCores);
                for(int id = 0; id < SystemInfo.LogicalCores; id++)
                {
                    final int start = ParallelUtils.getStartBlock(N, id, SystemInfo.LogicalCores);
                    final int end = ParallelUtils.getEndBlock(N, id, SystemInfo.LogicalCores);
                    futureChanges.add(threadpool.submit(new Callable<Integer>()
                    {
                        @Override
                        public Integer call()
                        {
                            double[] sqrdChange = new double[k];
                            double[] ownerChange = new double[k];
                            int localChange = 0;
                            for(int q = start; q < end; q++)
                                localChange += updateMeansFromChange(q, assignment, sqrdChange, ownerChange);
                            synchronized(assignment)
                            {
                                applyMeanUpdates(sqrdChange, ownerChange);
                            }
                            return localChange;
                        }
                    }));
                }
                
                try
                {
                    for (Future<Integer> f : futureChanges)
                        moved += f.get();
                }
                catch (ExecutionException ex)
                {
                    Logger.getLogger(ElkanKernelKMeans.class.getName()).log(Level.SEVERE, null, ex);
                }
                updateNormConsts();
                
                //now do cluster movement
                final CountDownLatch latch2 = new CountDownLatch(k);
                //Step 5
                for (int i = 0; i < k; i++)
                    distancesMoved[i] = meanToMeanDistance(i, i, newDesignations, assignment, oldSqrdNorms[i], threadpool);
                for (int i = 0; i < k; i++)
                {
                    final int c = i;
                    threadpool.submit(new Runnable()
                    {
                        @Override
                        public void run()
                        {
                            for (int q = 0; q < N; q++)
                                lowerBound[q][c] = Math.max(lowerBound[q][c] - distancesMoved[c], 0);
                            latch2.countDown();
                        }
                    });
                }
                latch2.await();
                //now we can move the assignments over
                System.arraycopy(newDesignations, 0, assignment, 0, N);
                
                //Step 6
                final CountDownLatch latch3 = new CountDownLatch(SystemInfo.LogicalCores);
                for(int id = 0; id < SystemInfo.LogicalCores; id++)
                {
                    final int start = ParallelUtils.getStartBlock(N, id, SystemInfo.LogicalCores);
                    final int end = ParallelUtils.getEndBlock(N, id, SystemInfo.LogicalCores);
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
                            latch3.countDown();
                        }
                    });
                }
                latch3.await();
                
                return moved;
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(ElkanKernelKMeans.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        //else, single threaded case
        
        //Re compute centroids. Hold off on copying newDesignations into assignment until after we do needed movement calculations
        for (int i = 0; i < N; i++)
            moved += updateMeansFromChange(i, assignment);
        updateNormConsts();
        
        //compute how far each cluster moved
        for (int i = 0; i < k; i++)
            distancesMoved[i] = meanToMeanDistance(i, i, newDesignations, assignment, oldSqrdNorms[i]);
        //now we can move the assignments over
        System.arraycopy(newDesignations, 0, assignment, 0, N);

        //Step 5
        for(int c = 0; c < k; c++)
            for(int q = 0; q < N; q++)
                lowerBound[q][c] = Math.max(lowerBound[q][c] - distancesMoved[c], 0);

        //Step 6
        for(int q = 0; q < N; q++)
        {
            upperBound[q] += distancesMoved[assignment[q]];
            r[q] = true; 
        }
        
        return moved;
    }

    private void step3aBoundsUpdate(boolean[] r, int q, final int[] assignment, double[] upperBound, double[][] lowerBound)
    {
        //3(a)
        if (r[q])
        {
            r[q] = false;
            int meanIndx = assignment[q];
            double d = distance(q, meanIndx, assignment);
            lowerBound[q][meanIndx] = d;///Not sure if this is supposed to be here
            upperBound[q] = d;
        }
    }

    private void step3bUpdate(double[] upperBound, final int q, double[][] lowerBound,
            final int c, double[][] centroidSelfDistances, final int[] assignment, 
            final AtomicBoolean changeOccurred)
    {
        //3(b)
        if (upperBound[q] > lowerBound[q][c] || upperBound[q] > centroidSelfDistances[assignment[q]][c] / 2)
        {
            double d = distance(q, c, assignment);
            lowerBound[q][c] = d;
            if (d < upperBound[q])
            {
                newDesignations[q] = c;
                
                upperBound[q] = d;
                
                changeOccurred.set(true);
            }
        }
    }

    private void calculateCentroidDistances(final int k, final double[][] centroidSelfDistances, final double[] sC, final int[] curAssignments, int[] prev_assignments, final ExecutorService threadpool)
    {
        if(prev_assignments == null)
        {
            prev_assignments = new int[curAssignments.length];
            Arrays.fill(prev_assignments, -1);
        }
        final int[] prev_assing = prev_assignments;
        
        //compute self dot-products
        update_centroid_pair_dots(prev_assing, curAssignments, threadpool);
        
        
        double[] weight_per_cluster = new double[k];
        for (int i = 0; i < curAssignments.length; i++)
            weight_per_cluster[curAssignments[i]] += W.get(i);
        //normalize dot products and make distances
        for (int i = 0; i < k; i++)
            for (int z = i + 1; z < k; z++)
            {
                double dot = centroidPairDots[i][z];
                dot /= (weight_per_cluster[i] * weight_per_cluster[z]);

                double d = meanSqrdNorms[i] * normConsts[i] + meanSqrdNorms[z] * normConsts[z] - 2 * dot;
                centroidSelfDistances[z][i] = centroidSelfDistances[i][z] = Math.sqrt(Math.max(0, d));//Avoid rare cases wehre 2*dot might be slightly larger
            }
        
        //update sC
        for (int i = 0; i < k; i++)
        {
            double sCmin = Double.MAX_VALUE;
            for (int z = 0; z < k; z++)
                if(i != z)
                    sCmin = Math.min(sCmin, centroidSelfDistances[i][z]);
            sC[i] = sCmin / 2.0;
        }
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, ExecutorService threadpool, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        if(dataSet.getSampleSize() < clusters)
            throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");
        
        cluster(dataSet, clusters, designations, false, threadpool);
        return designations;
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        if(dataSet.getSampleSize() < clusters)
            throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");

        cluster(dataSet, clusters, designations, false, null);
        
        return designations;
    }

    @Override
    public ElkanKernelKMeans clone()
    {
        return new ElkanKernelKMeans(this);
    }
}
