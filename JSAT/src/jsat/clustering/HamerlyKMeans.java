package jsat.clustering;

import java.util.ArrayList;
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
import static jsat.clustering.SeedSelectionMethods.selectIntialPoints;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;

/**
 * An efficient implementation of the K-Means algorithm. This implementation uses
 * the triangle inequality to accelerate computation while maintaining the exact
 * same solution. This requires that the {@link DistanceMetric} used support 
 * {@link DistanceMetric#isSubadditive() }. It uses only O(n) extra memory. <br>
 * Only the methods that specify the exact number of clusters are supported.<br>
 * <br>
 * See: Hamerly, G. (2010). <i>Making k-means even faster</i>. SIAM 
 * International Conference on Data Mining (SDM) (pp. 130–140). Retrieved from 
 * <a href="http://72.32.205.185/proceedings/datamining/2010/dm10_012_hamerlyg.pdf">here</a>
 * 
 * @author Edward Raff
 */
public class HamerlyKMeans extends KClustererBase
{
    private DistanceMetric dm;
    private SeedSelectionMethods.SeedSelection seedSelection;
    
    private boolean storeMeans = true;
    private List<Vec> means;

    /**
     * Creates a new k-Means object 
     * @param dm the distance metric to use for clustering
     * @param seedSelection the method of initial seed selection
     */
    public HamerlyKMeans(DistanceMetric dm, SeedSelectionMethods.SeedSelection seedSelection)
    {
        this.dm = dm;
        this.seedSelection = seedSelection;
        this.means = means;
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
    
    //TODO reduce some code duplication in the methods bellow 
    
    /**
     * Performs the main clustering work
     * @param dataSet the data set to cluster
     * @param assignment the array to store assignments in
     * @param exactTotal not used at the moment. 
     */
    protected void cluster(final DataSet dataSet, final int k, final int[] assignment, boolean exactTotal, ExecutorService threadpool)
    {
        final int N = dataSet.getSampleSize();
        final int D = dataSet.getNumNumericalVars();
        
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadpool);
        
        final List<Vec> X = dataSet.getDataVectors();
        final List<Double> distAccel;
        if(threadpool == null || threadpool instanceof FakeExecutor)
            distAccel = dm.getAccelerationCache(X);
        else
            distAccel = dm.getAccelerationCache(X, threadpool);
        
        final List<List<Double>> meanQI = new ArrayList<List<Double>>(k);
        
        if(threadpool == null || threadpool instanceof  FakeExecutor)
            means = selectIntialPoints(dataSet, k, dm, distAccel, new Random(), seedSelection);
        else
            means = selectIntialPoints(dataSet, k, dm, distAccel, new Random(), seedSelection, threadpool);
        
        /**
         * vector sum of all points in cluster j <br>
         * denoted c'(j)
         */
        final Vec[] cP = new Vec[k];
        final Vec[] tmpVecs = new Vec[k];
        /**
         * number of points assigned to cluster j,<br>
         * denoted q(j)
         */
        final AtomicLongArray q = new AtomicLongArray(k);
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
        Initialize(dataSet, q, means, tmpVecs, cP, u, l, assignment, threadpool, localDeltas, X, distAccel, meanQI);
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
            updateS(s, threadpool);
            
            if(threadpool == null)
            {
                int localUpdates = 0;
                for(int i = 0; i < N; i++)
                {
                    localUpdates += mainLoopWork(dataSet, i, s, assignment, u, l, q, cP, X, distAccel, meanQI);
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
                                localUpdates += mainLoopWork(dataSet, i, s, assignment, u, l, q, deltas, X, distAccel, meanQI);
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
            double[] l, AtomicLongArray q, Vec[] deltas, final List<Vec> X, final List<Double> distAccel, final List<List<Double>> meanQI)
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
                    q.decrementAndGet(a_i);
                    q.incrementAndGet(new_a_i);
                    deltas[a_i].mutableSubtract(x);
                    deltas[new_a_i].mutableAdd(x);
                    return 1;//1 change in ownership
                }
            }
        }
        return 0;//no change
    }
    
    private void updateS(final double[] s, ExecutorService threadpool)
    {
        final int tasks = means.size();
        final CountDownLatch latch = new CountDownLatch(tasks);
        for(int j = 0; j < means.size(); j++)
        {
            if(threadpool == null)
            {
                final Vec mean_j = means.get(j);
                double tmp;
                double min = Double.POSITIVE_INFINITY;
                for(int jp = 0; jp < means.size(); jp++)
                    if(jp == j)
                        continue;
                    else if((tmp = dm.dist(mean_j, means.get(jp))) < min)
                        min = tmp;
                s[j] = min;
            }
            else
            {
                final int J = j;
                threadpool.submit(new Runnable() 
                {
                    @Override
                    public void run()
                    {
                        final Vec mean_j = means.get(J);
                        double tmp;
                        double min = Double.POSITIVE_INFINITY;
                        for (int jp = 0; jp < means.size(); jp++)
                            if (jp == J)
                                continue;
                            else if ((tmp = dm.dist(mean_j, means.get(jp))) < min)
                                min = tmp;
                        s[J] = min;
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
    
    private void Initialize(final DataSet d, final AtomicLongArray q, final List<Vec> means, final Vec[] tmp, final Vec[] cP, final double[] u, final double[] l, final int[] a, ExecutorService threadpool, final ThreadLocal<Vec[]> localDeltas, final List<Vec> X, final List<Double> distAccel, final List<List<Double>> meanQI)
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
                q.incrementAndGet(j);
                cP[j].mutableAdd(x);
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
                            q.incrementAndGet(j);
                            deltas[j].mutableAdd(x);
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
    
    private void moveCenters(List<Vec> means, Vec[] tmpSpace, Vec[] cP, AtomicLongArray q, double[] p, final List<List<Double>> meanQI)
    {
        for(int j = 0; j < means.size(); j++)
        {
            //compute new mean
            cP[j].copyTo(tmpSpace[j]);
            tmpSpace[j].mutableDivide(q.get(j));
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
    
    /**
     * Returns the vector for the i'th data point. Used to stay consistent with 
     * the algorithm's notation and description 
     * @param d dataset of points
     * @param index the index of the point to obtain
     * @return the vector value for the given index
     */
    private static Vec x(DataSet d, int index)
    {
        return d.getDataPoint(index).getNumericalValues();
    }

    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, ExecutorService threadpool, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        if(dataSet.getSampleSize() < clusters)
            throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");
        
        cluster(dataSet, clusters, designations, false, threadpool);
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
        
        cluster(dataSet, clusters, designations, false, null);
        if(!storeMeans)
            means = null;
        
        return designations;
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
