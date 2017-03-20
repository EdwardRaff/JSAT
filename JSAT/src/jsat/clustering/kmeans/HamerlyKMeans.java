package jsat.clustering.kmeans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.clustering.SeedSelectionMethods;
import static jsat.clustering.SeedSelectionMethods.selectIntialPoints;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.AtomicDoubleArray;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * An efficient implementation of the K-Means algorithm. This implementation uses
 * the triangle inequality to accelerate computation while maintaining the exact
 * same solution. This requires that the {@link DistanceMetric} used support 
 * {@link DistanceMetric#isSubadditive() }. It uses only O(n) extra memory. <br>
 * <br>
 * See: Hamerly, G. (2010). <i>Making k-means even faster</i>. SIAM 
 * International Conference on Data Mining (SDM) (pp. 130–140). Retrieved from 
 * <a href="http://72.32.205.185/proceedings/datamining/2010/dm10_012_hamerlyg.pdf">here</a>
 * 
 * @author Edward Raff
 */
public class HamerlyKMeans extends KMeans
{

	private static final long serialVersionUID = -4960453870335145091L;

	/**
     * Creates a new k-Means object 
     * @param dm the distance metric to use for clustering
     * @param seedSelection the method of initial seed selection
     * @param rand the source of randomnes to use
     */
    public HamerlyKMeans(DistanceMetric dm, SeedSelectionMethods.SeedSelection seedSelection, Random rand)
    {
        super(dm, seedSelection, rand);
    }
    
    /**
     * Creates a new k-Means object 
     * @param dm the distance metric to use for clustering
     * @param seedSelection the method of initial seed selection
     */
    public HamerlyKMeans(DistanceMetric dm, SeedSelectionMethods.SeedSelection seedSelection)
    {
        this(dm, seedSelection, RandomUtil.getRandom());
    }
    
    /**
     * Creates a new k-Means object 
     */
    public HamerlyKMeans()
    {
        this(new EuclideanDistance(), SeedSelectionMethods.SeedSelection.KPP);
    }

    public HamerlyKMeans(HamerlyKMeans toCopy)
    {
        super(toCopy);
    }
    
    //TODO reduce some code duplication in the methods bellow 
    
    @Override
    protected double cluster(final DataSet dataSet, List<Double> accelCache, final int k, final List<Vec> means, final int[] assignment, final boolean exactTotal, ExecutorService threadpool, boolean returnError, Vec dataPointWeights)
    {
        final int N = dataSet.getSampleSize();
        final int D = dataSet.getNumNumericalVars();
        
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadpool);
        /**
         * Weights for each data point
         */
        final Vec W;
        if (dataPointWeights == null)
            W = dataSet.getDataWeights();
        else
            W = dataPointWeights;
        final List<Vec> X = dataSet.getDataVectors();
        final List<Double> distAccel;//used like htis b/c we want it final for convinence, but input may be null
        if(accelCache == null)
        {
            if(threadpool == null || threadpool instanceof FakeExecutor)
                distAccel = dm.getAccelerationCache(X);
            else
                distAccel = dm.getAccelerationCache(X, threadpool);
        }
        else
            distAccel = accelCache;
        
        final List<List<Double>> meanQI = new ArrayList<List<Double>>(k);
        
        if (means.size() != k)
        {
            means.clear();
            means.addAll(selectIntialPoints(dataSet, k, dm, distAccel, rand, seedSelection, threadpool));
        }

        //Make our means dense
        for (int i = 0; i < means.size(); i++)
            if (means.get(i).isSparse())
                means.set(i, new DenseVector(means.get(i)));

        /**
         * vector sum of all points in cluster j <br>
         * denoted c'(j)
         */
        final Vec[] cP = new Vec[k];
        final Vec[] tmpVecs = new Vec[k];
        /**
         * weighted number of points assigned to cluster j,<br>
         * denoted q(j)
         */
        final AtomicDoubleArray q = new AtomicDoubleArray(k);
        /**
         * distance that c(j) last moved <br>
         * denoted p(j)
         */
        final double[] p = new double[k];
        /**
         * distance from c(j) to its closest other center.<br>
         * denoted s(j)
         */
        final double[] s = new double[k];
        
        //index of the center to which x(i) is assigned
        //use assignment array
        
        /**
         * upper bound on the distance between x(i) and its assigned center c(a(i)) <br>
         * denoted u(i)
         */
        final double[] u = new double[N];
        /**
         * lower bound on the distance between x(i) and its second closest 
         * center – that is, the closest center to x(i) that is not c(a(i)) <br>
         * denoted l(i)
         */
        final double[] l = new double[N];
        
        final ThreadLocal<Vec[]> localDeltas = new ThreadLocal<Vec[]>()
        {
            @Override
            protected Vec[] initialValue()
            {
                Vec[] toRet = new Vec[means.size()];
                for(int i = 0; i < k; i++)
                    toRet[i] = new DenseVector(D);
                return toRet;
            }
        };
        
        //Start of algo
        Initialize(dataSet, q, means, tmpVecs, cP, u, l, assignment, threadpool, localDeltas, X, distAccel, meanQI, W);
        //Use dense mean objects
        for(int i = 0; i < means.size(); i++)
            if(means.get(i).isSparse())
                means.set(i, new DenseVector(means.get(i)));
        final AtomicInteger updates = new AtomicInteger(N);
        while(updates.get() > 0)
        {
            moveCenters(means, tmpVecs, cP, q, p, meanQI);
            UpdateBounds(p, assignment, u, l);
            updates.set(0);
            updateS(s, means, threadpool, meanQI);
            
            if(threadpool == null)
            {
                int localUpdates = 0;
                for(int i = 0; i < N; i++)
                {
                    localUpdates += mainLoopWork(dataSet, i, s, assignment, u, l, q, cP, X, distAccel, means, meanQI, W);
                }
                updates.set(localUpdates);
            }
            else
            {
                final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
                for(int id = 0; id < SystemInfo.LogicalCores; id++)
                {
                    final int ID = id;
                    threadpool.submit(new Runnable() 
                    {
                        @Override
                        public void run()
                        {
                            Vec[] deltas = localDeltas.get();
                            int localUpdates = 0;
                            for(int i = ID; i < N; i+=SystemInfo.LogicalCores)
                            {
                                localUpdates += mainLoopWork(dataSet, i, s, assignment, u, l, q, deltas, X, distAccel, means, meanQI, W);
                            }
                            //collect deltas
                            if(localUpdates > 0)
                            {
                                updates.getAndAdd(localUpdates);
                                for(int i = 0; i < cP.length; i++)
                                {
                                    synchronized(cP[i])
                                    {
                                        cP[i].mutableAdd(deltas[i]);
                                    }
                                    deltas[i].zeroOut();
                                }
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
                    Logger.getLogger(HamerlyKMeans.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
        
        if (returnError)
        {
            double totalDistance = 0;
            
            if (saveCentroidDistance)
                nearestCentroidDist = new double[N];
            else
                nearestCentroidDist = null;

            if (exactTotal == true)
                for (int i = 0; i < N; i++)
                {
                    double dist = dm.dist(i, means.get(assignment[i]), meanQI.get(assignment[i]), X, distAccel);
                    totalDistance += Math.pow(dist, 2);
                    if (saveCentroidDistance)
                        nearestCentroidDist[i] = dist;
                }
            else
                for (int i = 0; i < N; i++)
                {
                    totalDistance += Math.pow(u[i], 2);
                    if (saveCentroidDistance)
                        nearestCentroidDist[i] = u[i];
                }

            return totalDistance;
        }
        else
            return 0;//who cares
    }

    /**
     * 
     * @param dataSet data set
     * @param i the index to do the work for
     * @param s the centroid centroid nearest distance
     * @param assignment the array assignments are stored in
     * @param u the "u" array of the algo
     * @param l the "l" array of the algo
     * @param q the "q" array of the algo (cluster counts)
     * @param deltas the location to store the computed delta if one occurs 
     * @return 0 if no changes in assignment were made, 1 if a change in assignment was made
     */
    private int mainLoopWork(DataSet dataSet, int i, double[] s, int[] assignment, double[] u, 
            double[] l, AtomicDoubleArray q, Vec[] deltas, final List<Vec> X, final List<Double> distAccel, final List<Vec> means, final List<List<Double>> meanQI, final Vec W)
    {
        final int a_i = assignment[i];
        double m = Math.max(s[a_i] / 2, l[i]);
        if (u[i] > m)//first bound test
        {
            Vec x = X.get(i);
            u[i] = dm.dist(i, means.get(a_i), meanQI.get(a_i), X, distAccel);//tighten upper bound
            if (u[i] > m)  //second bound test
            {
                final int new_a_i = PointAllCtrs(x, i, means, assignment, u, l, X, distAccel, meanQI);
                if (a_i != new_a_i)
                {
                    double w = W.get(i);
                    q.addAndGet(a_i, -w);
                    q.addAndGet(new_a_i, w);
                    deltas[a_i].mutableSubtract(w, x);
                    deltas[new_a_i].mutableAdd(w, x);
                    return 1;//1 change in ownership
                }
            }
        }
        return 0;//no change
    }
    
    private void updateS(final double[] s, final List<Vec> means, final ExecutorService threadpool, final List<List<Double>> meanQIs)
    {
        final int tasks = means.size();
        final CountDownLatch latch = new CountDownLatch(tasks);
        Arrays.fill(s, Double.MAX_VALUE);
        //TODO temp object for puting all the query info into a cache, should probably be cleaned up - or change original code to have one massive list and then use sub lits to get the QIs individualy 
        final DoubleList meanCache = meanQIs.get(0).isEmpty() ? null : new DoubleList(meanQIs.size());
        if (meanCache != null)
            for (List<Double> qi : meanQIs)
                meanCache.addAll(qi);

        for (int j = 0; j < means.size(); j++)
        {
            if(threadpool == null)
            {
                double tmp;
                double min = Double.POSITIVE_INFINITY;
                int otherIndx = Integer.MAX_VALUE;
                for(int jp = j+1; jp < means.size(); jp++)
                    if((tmp = dm.dist(j, jp, means, meanCache)) < min)
                    {
                        min = tmp;
                        otherIndx = jp;
                    }
                s[j] = Math.min(min, s[j]);
                //trick to avoid computing twice as many distances as needed
                if(otherIndx < s.length)//if index i is our min, we may be their min too
                    s[otherIndx] = Math.min(s[otherIndx], s[j]);
            }
            else
            {
                final int J = j;
                threadpool.submit(new Runnable() 
                {
                    @Override
                    public void run()
                    {
                        double tmp;
                        double min = Double.POSITIVE_INFINITY;
                        int otherIndx = Integer.MAX_VALUE;
                        for (int jp = J+1; jp < means.size(); jp++)
                            if ((tmp = dm.dist(J, jp, means, meanCache)) < min)
                            {
                                min = tmp;
                                otherIndx = jp;
                            }

                        synchronized (s)
                        {
                            min = s[J] = Math.min(min, s[J]);
                            if (otherIndx < s.length)
                                s[otherIndx] = Math.min(min, s[otherIndx]);
                        }
                        latch.countDown();
                    }
                });
            }
        }
        
        if (threadpool != null)
            try
            {
                latch.await();
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(HamerlyKMeans.class.getName()).log(Level.SEVERE, null, ex);
            }
    }
    
    private void Initialize(final DataSet d, final AtomicDoubleArray q, final List<Vec> means, final Vec[] tmp, final Vec[] cP, final double[] u, final double[] l, final int[] a, ExecutorService threadpool, final ThreadLocal<Vec[]> localDeltas, final List<Vec> X, final List<Double> distAccel, final List<List<Double>> meanQI, final Vec W)
    {
        for(int j = 0; j < means.size(); j++)
        {
            //q would already be initalized to zero on creation by java
            cP[j] = new DenseVector(means.get(0).length());
            tmp[j] = cP[j].clone();
            
            //set up Quer Info for means
            if(dm.supportsAcceleration())
                meanQI.add(dm.getQueryInfo(means.get(j)));
            else
                meanQI.add(Collections.EMPTY_LIST);
        }

        if(threadpool==null)
        {
            for(int i = 0; i < u.length; i++)
            {
                Vec x = X.get(i);
                int j = PointAllCtrs(x, i, means, a, u, l, X, distAccel, meanQI);
                double w = W.get(i);
                q.addAndGet(j, w);
                cP[j].mutableAdd(w, x);
            }
        }
        else
        {
            final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
            for(int id = 0; id < SystemInfo.LogicalCores; id++)
            {
                final int ID = id;
                threadpool.submit(new Runnable() 
                {
                    @Override
                    public void run()
                    {
                        Vec[] deltas = localDeltas.get();
                        for (int i = ID; i < u.length; i+=SystemInfo.LogicalCores)
                        {
                            Vec x = X.get(i);
                            int j = PointAllCtrs(x, i, means, a, u, l, X, distAccel, meanQI);
                            double w = W.get(i);
                            q.addAndGet(j, w);
                            deltas[j].mutableAdd(w, x);
                        }

                        for(int i = 0; i < cP.length; i++)
                        {
                            synchronized(cP[i])
                            {
                                cP[i].mutableAdd(deltas[i]);
                            }
                            deltas[i].zeroOut();
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
                Logger.getLogger(HamerlyKMeans.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
    /**
     * 
     * @param x
     * @param i
     * @param means
     * @param a
     * @param u
     * @param l
     * @return the index of the closest cluster center 
     */
    private int PointAllCtrs(Vec x, int i, List<Vec> means, int[] a, double[] u, double[] l, final List<Vec> X, final List<Double> distAccel, final List<List<Double>> meanQI)
    {
        double secondLowest = Double.POSITIVE_INFINITY;
        int slIndex = -1;
        double lowest = Double.MAX_VALUE;
        int lIndex = -1;
        
        for(int j = 0; j < means.size(); j++)
        {
            double dist = dm.dist(i, means.get(j), meanQI.get(j), X, distAccel);

            if(dist < secondLowest)
            {
                if(dist < lowest)
                {
                    secondLowest = lowest;
                    slIndex = lIndex;
                    lowest = dist;
                    lIndex = j;
                }
                else
                {
                    secondLowest = dist;
                    slIndex = j;
                }
            }
        }
        
        a[i] = lIndex;
        u[i] = lowest;
        l[i] = secondLowest;
        return lIndex;
    }
    
    private void moveCenters(List<Vec> means, Vec[] tmpSpace, Vec[] cP, AtomicDoubleArray q, double[] p, final List<List<Double>> meanQI)
    {
        for(int j = 0; j < means.size(); j++)
        {
            double count = q.get(j);
            if(count > 0)
            {
                //compute new mean
                cP[j].copyTo(tmpSpace[j]);
                tmpSpace[j].mutableDivide(count);
            }
            else
            {
                cP[j].zeroOut();
                tmpSpace[j].zeroOut();
            }
            //compute distance betwean new and old
            p[j] = dm.dist(means.get(j), tmpSpace[j]);
            //move it to its positaiotn as new mean
            tmpSpace[j].copyTo(means.get(j));
            
            //update QI
            if(dm.supportsAcceleration())
                meanQI.set(j, dm.getQueryInfo(means.get(j)));
        }
    }
    
    private void UpdateBounds(double[] p, int[] a, double[] u, double[] l)
    {
        double secondHighest = Double.NEGATIVE_INFINITY;
        int shIndex = -1;
        double highest = -Double.MAX_VALUE;
        int hIndex = -1;
        
        //find argmax values 
        for(int j = 0; j < p.length; j++)
        {
            double dist = p[j];
            if(dist > secondHighest)
            {
                if(dist > highest)
                {
                    secondHighest = highest;
                    shIndex = hIndex;
                    highest = dist;
                    hIndex = j;
                }
                else
                {
                    secondHighest = dist;
                    shIndex = j;
                }
            }
        }
        
        final int r = hIndex;
        final int rP = shIndex;
        
        for(int i = 0; i < u.length; i++)
        {
            final int j = a[i];
            u[i] += p[j];
            if(r == j)
                l[i] -= p[rP];
            else
                l[i] -= p[r];
        }
    }

    @Override
    public HamerlyKMeans clone()
    {
        return new HamerlyKMeans(this);
    }

}
