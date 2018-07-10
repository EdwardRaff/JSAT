package jsat.clustering.kmeans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
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
import jsat.utils.IndexTable;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.AtomicDoubleArray;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

/**
 * An efficient implementation of the K-Means algorithm. This implementation uses
 * the triangle inequality to accelerate computation while maintaining the exact
 * same solution. This requires that the {@link DistanceMetric} used
 * support {@link DistanceMetric#isSubadditive() }. It uses only O(n) extra
 * memory. <br>
 * <br>
 * See:
 * <ul>
 * <li>Hamerly, G. (2010). <i>Making k-means even faster</i>. SIAM International
 * Conference on Data Mining (SDM) (pp. 130–140). Retrieved from
 * <a href="http://72.32.205.185/proceedings/datamining/2010/dm10_012_hamerlyg.pdf">here</a></li>
 * <li>Ryšavý, P., & Hamerly, G. (2016). Geometric methods to accelerate k-means
 * algorithms. In Proceedings of the 2016 SIAM International Conference on Data
 * Mining (pp. 324–332). Philadelphia, PA: Society for Industrial and Applied
 * Mathematics.
 * <a href="http://doi.org/10.1137/1.9781611974348.37">http://doi.org/10.1137/1.9781611974348.37</a></li>
 * </ul>
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
    protected double cluster(final DataSet dataSet, List<Double> accelCache, final int k, final List<Vec> means, final int[] assignment, final boolean exactTotal, boolean parallel, boolean returnError, Vec dataPointWeights)
    {
        final int N = dataSet.size();
        final int D = dataSet.getNumNumericalVars();
        
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, parallel);
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
            distAccel = dm.getAccelerationCache(X, parallel);
        else
            distAccel = accelCache;
        
        final List<List<Double>> meanQI = new ArrayList<>(k);
        
        if (means.size() != k)
        {
            means.clear();
            means.addAll(selectIntialPoints(dataSet, k, dm, distAccel, rand, seedSelection, parallel));
        }

        /**
         * Used in bound updates. Contains the centroid means from the previous iteration
         */
        final Vec[] oldMeans = new Vec[means.size()];
        /**
         * Distance each mean has moved from one iteration to the next
         */
        final double[] distanceMoved = new double[means.size()];
        //Make our means dense
        for (int i = 0; i < means.size(); i++)
        {
            if (means.get(i).isSparse())
                means.set(i, new DenseVector(means.get(i)));
            oldMeans[i] = new DenseVector(means.get(i));
        }

        /**
         * vector sum of all points in cluster j <br>
         * denoted c'(j)
         */
        final Vec[] cP = new Vec[k];
        /**
         * Will get intialized in the Initialize function
         */
        final Vec[] tmpVecs = new Vec[k];
        final Vec[] tmpVecs2 = new Vec[k];
        for(int i = 0; i < tmpVecs2.length; i++)
            tmpVecs2[i] = new DenseVector(oldMeans[0].length());
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
        
        final List<Vec[]> allLocalDeltas = Collections.synchronizedList(new ArrayList<>());
        final ThreadLocal<Vec[]> localDeltas =  ThreadLocal.withInitial(()->
        {
            Vec[] toRet = new Vec[means.size()];
            for(int i = 0; i < k; i++)
                toRet[i] = new DenseVector(D);
            allLocalDeltas.add(toRet);
            return toRet;
        });
        
        //Start of algo
        Initialize(dataSet, q, means, tmpVecs, cP, u, l, assignment, parallel, localDeltas, X, distAccel, meanQI, W);
        //Use dense mean objects
        for(int i = 0; i < means.size(); i++)
            if(means.get(i).isSparse())
                means.set(i, new DenseVector(means.get(i)));
        int updates = N;

        /**
         * How many iterations over the dataset did this take? 
         */
        int iteration = 0;
        while(updates > 0)
        {
            moveCenters(means, oldMeans, tmpVecs, cP, q, p, meanQI);
            updates = 0;
            updateS(s, distanceMoved, means, oldMeans, parallel, meanQI);
            /**
             * we maintain m(ci), which is the radius of a hypersphere centered
             * at centroid ci that contains all points assigned to ci. (Note
             * that m(ci) is easily obtained as the maximum upper-bound of all
             * points in the cluster.)
             */
            double[] m = new double[means.size()];
            Arrays.fill(m, 0.0);
            for(int i = 0; i < N; i++)
                m[assignment[i]] = Math.max(m[assignment[i]], u[i]);
            double[] updateB = new double[m.length];
            //Algorithm 3, new bounds update scheme. See "Geometric methods to accelerate k-means algorithms"
            EnhancedUpdateBounds(means, distanceMoved, m, s, oldMeans, tmpVecs, tmpVecs2, updateB, p, assignment, u, l);
            
            //perform all updates
            updates = ParallelUtils.run(parallel, N, (i) ->
            {
                Vec[] deltas = localDeltas.get();
                return mainLoopWork(dataSet, i, s, assignment, u, l, q, deltas, X, distAccel, means, meanQI, W);
            }, (a, b) -> a + b);
            
            //acumulate all deltas
            ParallelUtils.range(cP.length, parallel).forEach(i -> 
            {
                for (Vec[] deltas : allLocalDeltas)
                {
                    cP[i].mutableAdd(deltas[i]);
                    deltas[i].zeroOut();
                }
            });
            iteration++;
        }
        
        if (returnError)
        {   
            if (saveCentroidDistance)
                nearestCentroidDist = new double[N];
            else
                nearestCentroidDist = null;

            double totalDistance = ParallelUtils.run(parallel, N, (start, end) ->
            {
                double localDistTotal = 0;
                for(int i = start; i < end; i++)
                {
                    double dist;
                    if(exactTotal)
                        dist = dm.dist(i, means.get(assignment[i]), meanQI.get(assignment[i]), X, distAccel);
                    else
                        dist = u[i];
                    localDistTotal += Math.pow(dist, 2);
                    if (saveCentroidDistance)
                        nearestCentroidDist[i] = dist;
                }
                return localDistTotal;
            },
            (a,b)->a+b);

            return totalDistance;
        }
        else
            return 0;//who cares
    }

    private void EnhancedUpdateBounds(final List<Vec> means1, final double[] distanceMoved, double[] m, final double[] s, final Vec[] oldMeans, final Vec[] tmpVecs, final Vec[] tmpVecs2, double[] updateB, final double[] p, final int[] assignment, final double[] u, final double[] l)
    {
        //NOTE: special here c'j eans the new cluster location.  Only for algorithm 3 update. Rest of code uses different notation
        //Paper uses current and next, but we are coding current and previous
        //so in the paper's termonology, c'j would be the current mean, and cj the previous
        for (int i = 0; i < means1.size(); i++)
        {
            //3: update←−∞
            double update = Double.NEGATIVE_INFINITY;
            //4: for each cj in centroids that fulfill (3.10) in decreasing order of ||c'j −cj|| do.
            IndexTable c_order = new IndexTable(distanceMoved);
            c_order.reverse();//we want decreasing order
            for (int order = 0; order < c_order.length(); order++)
            {
                int j = c_order.index(order);
                if(j == i)
                    continue;
                //check (3.10)
                if(2*m[j] + s[j] < distanceMoved[j] )//paper uses s(c_i) for half the distance, but our code uses it for the full distance
                    continue;//you didn't satisfy (3.10)
                //5: if ||c'j −cj|| ≤ update then break
                if(distanceMoved[j] <= update)
                    break;
                //6: update ← max{update calculated by Algorithm 2 using ci and cj, update}
                double algo2_1_out;
                //begin Algorithm 2 Algorithm for update of l(x, cj) in the multidimensional case, where c(x) = ci
                //3: t← eq (3.6)
                oldMeans[i].copyTo(tmpVecs[i]);
                means1.get(j).copyTo(tmpVecs2[i]);
                tmpVecs[i].mutableSubtract(oldMeans[j]);//tmpVec[i] = (ci - cj)
                tmpVecs2[i].mutableSubtract(oldMeans[j]);//tmpVec2[i] = (c'j - cj)
                double t = tmpVecs[i].dot(tmpVecs2[i])/(distanceMoved[j]*distanceMoved[j]);
                //4: dist←||cj + t · (c'j −cj)−ci||
                //can be re-arragned as  ||cj −ci + t · (c'j −cj)||
                // = || (cj −ci) + t · (c'j −cj)||
                // = || -(ci - cj) + t · (c'j −cj)||
//                        double dist = oldMeans[j].add(means.get(j).subtract(oldMeans[j]).multiply(t)).subtract(oldMeans[i]).pNorm(2);
                tmpVecs[i].mutableMultiply(-1);
                tmpVecs[i].mutableAdd(t, tmpVecs2[i]);
                double dist = tmpVecs2[i].pNorm(2);
                //5: cix ← (3.7)
                double c_ix = dist*2/distanceMoved[j];
                //6: ciy ←1−2t
                double c_iy = 1 - 2 * t;
                //7: r ←  (3.9)
                double r = m[i]*2/distanceMoved[j];
                //8: return update calculated by Algorithm 1 (using r and ci = (cix, ciy)) multiplied by ||c'j−cj||/2
                //Algorithm 1 Algorithm for update of l(x, cj) in the simplified two dimensional case, where c(x) = ci.
                //3: if cix ≤ r then return max{0,min{2, 2(r − ciy)}}
                if(c_ix <= r)
                    algo2_1_out = Math.max(0, Math.min(2, 2*(r-c_iy)));
                else
                {
                    //4: if ciy > r then
                    if(c_iy > r)
                        c_iy--;//5:ciy ←ciy −1
                    //7: return eq (3.2)
                    double proj_norm_sqrd = Math.sqrt(c_ix * c_ix + c_iy * c_iy);
                    proj_norm_sqrd *= proj_norm_sqrd;
                    algo2_1_out = 2*(c_ix*r-c_iy*Math.sqrt(proj_norm_sqrd-r*r))/proj_norm_sqrd;
                }
                //end Algorithm 1
                algo2_1_out *= distanceMoved[j]/2;
                //end Algorithm 2
                update = Math.max(algo2_1_out, update);
            }
            updateB[i] = update;
        }
        //"The appropriate place to calculate the maximum upper bound is before any centroids move", ok we can do upper bound now
        UpdateBounds(p, assignment, u, l, updateB);
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
    
    /**
     * 
     * @param s updated by this method call
     * @param distanceMoved distance each cluster has moved from the previous locations. Updated by this method call. 
     * @param means the new cluster means
     * @param oldMeans the old cluster means
     * @param parallel
     * @param meanQIs 
     */
    private void updateS(final double[] s, final double[] distanceMoved, final List<Vec> means, final Vec[] oldMeans, final boolean parallel, final List<List<Double>> meanQIs)
    {
        Arrays.fill(s, Double.MAX_VALUE);
        //TODO temp object for puting all the query info into a cache, should probably be cleaned up - or change original code to have one massive list and then use sub lits to get the QIs individualy 
        final DoubleList meanCache = meanQIs.get(0).isEmpty() ? null : new DoubleList(meanQIs.size());
        if (meanCache != null)
            for (List<Double> qi : meanQIs)
                meanCache.addAll(qi);
        
        final ThreadLocal<double[]> localS = ThreadLocal.withInitial(()->new double[s.length]);
        
        ParallelUtils.run(parallel, means.size(), (j)->
        {
            double[] sTmp = localS.get();
            Arrays.fill(sTmp, Double.POSITIVE_INFINITY);
            distanceMoved[j] = dm.dist(oldMeans[j], means.get(j));
            double tmp;
            for (int jp = j + 1; jp < means.size(); jp++)
            {
                tmp = dm.dist(j, jp, means, meanCache);
                sTmp[j] = Math.min(sTmp[j], tmp);
                sTmp[jp] = Math.min(sTmp[jp], tmp);
            }

            synchronized(s)
            {
                for(int i = 0; i < s.length; i++)
                    s[i] = Math.min(s[i], sTmp[i]);
            }

        });
    }
    
    private void Initialize(final DataSet d, final AtomicDoubleArray q, final List<Vec> means, final Vec[] tmp, final Vec[] cP, final double[] u, final double[] l, final int[] a, boolean parallel, final ThreadLocal<Vec[]> localDeltas, final List<Vec> X, final List<Double> distAccel, final List<List<Double>> meanQI, final Vec W)
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
        
        ParallelUtils.run(parallel, u.length, (start, end) ->
        {
            Vec[] deltas = localDeltas.get();
            for (int i = start; i < end; i++)
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
        });
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
    
    private void moveCenters(List<Vec> means, Vec[] oldMeans, Vec[] tmpSpace, Vec[] cP, AtomicDoubleArray q, double[] p, final List<List<Double>> meanQI)
    {
        for(int j = 0; j < means.size(); j++)
        {
            double count = q.get(j);
            //save old mean
            means.get(j).copyTo(oldMeans[j]);
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
    
    /**
     * 
     * @param p distance that c(j) last moved, denoted p(j)
     * @param a
     * @param u
     * @param l 
     */
    private void UpdateBounds(double[] p, int[] a, double[] u, double[] l, double[] updateB)
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
                l[i] -= Math.min(p[rP], updateB[j]);
            else
                l[i] -= Math.min(p[r], updateB[j]);
        }
    }

    @Override
    public HamerlyKMeans clone()
    {
        return new HamerlyKMeans(this);
    }

}
