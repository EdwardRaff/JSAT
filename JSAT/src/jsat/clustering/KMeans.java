
package jsat.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import static jsat.clustering.SeedSelectionMethods.selectIntialPoints;
import static jsat.clustering.SeedSelectionMethods.selectIntialPoints;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.math.OnLineStatistics;
import jsat.utils.ModifiableCountDownLatch;
import jsat.utils.PoisonRunnable;
import jsat.utils.SystemInfo;

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
    private DistanceMetric dm;
    private Random rand;
    private SeedSelection seedSelection;
    
    /**
     * Control the maximum number of iterations to perform. 
     */
    protected int MaxIterLimit = 100;

    public KMeans(DistanceMetric dm, Random rand, SeedSelection seedSelection)
    {
        if(!dm.isSubadditive())
            throw new ArithmeticException("KMeans implementation requires the triangle inequality");
        this.dm = dm;
        this.rand = rand;
        this.seedSelection = seedSelection;
    }

    public KMeans(DistanceMetric dm, Random rand)
    {
        this(dm, rand, SeedSelection.KPP);
    }

    public KMeans(DistanceMetric dm)
    {
        this(dm, new Random(2));
    }

    public KMeans()
    {
        this(new EuclideanDistance());
    }


    /**
     * Sets the maximum number of iterations allowed
     * @param MaxIterLimit 
     */
    public void setIterationLimit(int iterLimit)
    {
        this.MaxIterLimit = iterLimit;
    }

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
            /**
             * K clusters
             */
            final int k = means.size();
            /**
             * N data points
             */
            final int N = dataSet.getSampleSize();

            TrainableDistanceMetric.trainIfNeeded(dm, dataSet);

            final double[][] lowerBound = new double[N][k];
            final double[] upperBound = new double[N];

            /**
             * Distances between centroid i and all other centroids
             */
            final double[][] centroidSelfDistances = new double[k][k];
            final double[] sC = new double[k];
            calculateCentroidDistances(k, centroidSelfDistances, means, sC);
            if (threadpool == null)
                initialClusterSetUp(k, N, dataSet, means, lowerBound, upperBound, centroidSelfDistances, assignment);
            else
                initialClusterSetUp(k, N, dataSet, means, lowerBound, upperBound, centroidSelfDistances, assignment, threadpool);

            int atLeast = 2;//Used to performan an extra round (first round does not assign)
            final AtomicInteger changes = new AtomicInteger(N);//Atomic int so each thread can update safely & lock free
            final boolean[] r = new boolean[N];//Default value of a boolean is false, which is what we want
            Vec[] oldMeans = new Vec[k];//The means fromt he current step are needed when computing the new means
            for (int i = 0; i < k; i++)
                oldMeans[i] = means.get(0).clone();//This way the new vectors are of the same implementation

            final ArrayBlockingQueue<Runnable> runnableList = threadpool == null ? null : new ArrayBlockingQueue<Runnable>(4 * SystemInfo.LogicalCores, false);

            int iterLimit = MaxIterLimit;
            while ((changes.intValue() > 0 || atLeast > 0) && iterLimit-- >= 0)
            {
                atLeast--;
                changes.set(0);
                //Step 1 
                calculateCentroidDistances(k, centroidSelfDistances, means, sC);

                final ModifiableCountDownLatch latch = new ModifiableCountDownLatch(1);

                //Create readers to run jobs 
                if (threadpool != null)
                    for (int i = 0; i < SystemInfo.LogicalCores; i++)
                        threadpool.submit(new Runnable()
                        {

                            @Override
                            public void run()
                            {
                                while (true)
                                {
                                    try
                                    {
                                        Runnable r = runnableList.take();
                                        if (r instanceof PoisonRunnable)
                                            return;
                                        r.run();
                                    }
                                    catch (InterruptedException ex)
                                    {
                                        Logger.getLogger(KMeans.class.getName()).log(Level.SEVERE, null, ex);
                                    }
                                }
                            }
                        });

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
                                step3aBoundsUpdate(r, q, v, means, assignment, upperBound);
                                step3bUpdate(upperBound, q, lowerBound, c, centroidSelfDistances, assignment, v, means, changes);
                            }
                    }
                    else
                    {
                        final int qq = q;
                        Runnable run = new Runnable() {

                            public void run()
                            {
                                for (int c = 0; c < k; c++)
                                    if (c != assignment[qq] && upperBound[qq] > lowerBound[qq][c] && upperBound[qq] > centroidSelfDistances[assignment[qq]][c] * 0.5)
                                    {
                                        step3aBoundsUpdate(r, qq, v, means, assignment, upperBound);
                                        step3bUpdate(upperBound, qq, lowerBound, c, centroidSelfDistances, assignment, v, means, changes);
                                    }
                                latch.countDown();
                            }
                        };
                        latch.countUp();
                        runnableList.put(run);
                    }
                }
                
                if (threadpool != null)
                {
                    //Pop extra off
                    for (int i = 0; i < SystemInfo.LogicalCores; i++)
                        runnableList.put(new PoisonRunnable());
                    //Still need to wait for everyone to finish, some threasd may still be easting jobs from the que 
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
                step4UpdateCentroids(k, means, oldMeans, N, assignment, dataSet);

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

    private void initialClusterSetUp(final int k, final int N, final DataSet dataSet, final List<Vec> means, final double[][] lowerBound, final double[] upperBound, final double[][] centroidSelfDistances, final int[] assignment)
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
        }
    }
    
    private void initialClusterSetUp(final int k, final int N, final DataSet dataSet, final List<Vec> means, final double[][] lowerBound, final double[] upperBound, final double[][] centroidSelfDistances, final int[] assignment, ExecutorService threadpool)
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
            Logger.getLogger(KMeans.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void step4UpdateCentroids(final int k, final List<Vec> means, Vec[] oldMeans, final int N, final int[] assignment, final DataSet dataSet)
    {
        //Step 4
        //Re compute centroids
        for (int i = 0; i < k; i++)
        {
            means.get(i).copyTo(oldMeans[i]);
            means.get(i).zeroOut();
        }

        int[] bucketCount = new int[k];

        for (int q = 0; q < N; q++)
        {
            bucketCount[assignment[q]]++;
            means.get(assignment[q]).mutableAdd(dataSet.getDataPoint(q).getNumericalValues());
        }

        for (int i = 0; i < k; i++)
            means.get(i).mutableDivide(bucketCount[i]);
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

    private void step3aBoundsUpdate(boolean[] r, int q, Vec v, final List<Vec> means, final int[] assignment, double[] upperBound)
    {
        //3(a)
        if (r[q])
        {
            r[q] = false;
            double d = dm.dist(v, means.get(assignment[q]));
            //lowerBound[q][assignment[q]] = d;///Not sure if this is supposed to be here
            upperBound[q] = d;
        }
    }

    private void step3bUpdate(double[] upperBound, final int q, double[][] lowerBound, final int c, double[][] centroidSelfDistances, 
            final int[] assignment, Vec v, final List<Vec> means, final AtomicInteger changes)
    {
        //3(b)
        if (upperBound[q] > lowerBound[q][c] || upperBound[q] > centroidSelfDistances[assignment[q]][c] / 2)
        {
            double d = dm.dist(v, means.get(c));
            lowerBound[q][c] = d;
            if (d < upperBound[q])
            {
                changes.incrementAndGet();
                assignment[q] = c;
                upperBound[q] = d;
            }
        }
    }

    private void calculateCentroidDistances(final int k, double[][] centroidSelfDistances, final List<Vec> means, double[] sC)
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
        }
    }

    static protected List<List<DataPoint>> getListOfLists(int k)
    {
        List<List<DataPoint>> ks = new ArrayList<List<DataPoint>>(k);
        for(int i = 0; i < k; i++)
            ks.add(new ArrayList<DataPoint>());
        return ks;
    }

    public int[] cluster(DataSet dataSet, int[] designations)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()/2), designations);
    }

    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()/2), threadpool, designations);
    }

    public int[] cluster(DataSet dataSet, int clusters, ExecutorService threadpool, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        
        cluster(dataSet, selectIntialPoints(dataSet, clusters, dm, rand, seedSelection, threadpool), designations, false, threadpool);
        
        return designations;
    }

    public int[] cluster(DataSet dataSet, int clusters, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        
        cluster(dataSet, selectIntialPoints(dataSet, clusters, dm, rand, seedSelection), designations, false);
        
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

    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
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
            double tmp = 0;
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
