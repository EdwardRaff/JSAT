package jsat.linear.vectorcollection.lsh;

import java.util.List;

import jsat.distributions.Normal;
import jsat.linear.Vec;
import static java.lang.Math.*;

import java.util.*;

import jsat.linear.DenseVector;
import jsat.linear.VecPaired;
import jsat.linear.VecPairedComparable;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.ManhattanDistance;
import jsat.utils.IntList;
import jsat.utils.IntSet;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * This is an implementation of Locality Sensitive Hashing for the 
 * {@link ManhattanDistance L<sub>1</sub>} and 
 * {@link EuclideanDistance L<sub>2</sub> } distance metrics. This is 
 * essentially a vector collection that can only perform a radius search for a 
 * pre-defined radius. In addition, the results are only approximate - not all 
 * of the correct points may be returned, and it is possible no points will be 
 * returned when the truth is that some data points do exist. 
 * <br><br>
 * Searching is done using the {@link #searchR(jsat.linear.Vec, boolean) } 
 * methods. While the set of points returned is approximate, the distance values
 * are exact. This is because no approximate distance is available, so the 
 * distances must be computed to remove violators. 
 * <br><br>
 * LSH may be useful if any of the following apply to your problem<br>
 * <ul>
 * <li>Only need to do a radius searches of a small number of fixed size 
 * increments</li>
 * <li>You need only the first few nearest neighbors, and can compute a 
 * threshold for the NN</li>
 * <li>Approximate neighbor results do not heavily impact the results of your 
 * algorithm</li>
 * <li>You want to find near-duplicates in a data set</li>
 * </ul>
 * <br><br>
 * This implementation is based heavily on the following, but is not an 
 * exact re-implementation.
 * <br><br>
 * See:<br>
 * <ul>
 * <li>Datar, M., Immorlica, N., Indyk, P.,&amp;Mirrokni, V. S. (2004). <i>
 * Locality-sensitive hashing scheme based on p-stable distributions</i>. 
 * Proceedings of the twentieth annual symposium on Computational geometry - 
 * SCG  ’04 (pp. 253–262). New York, New York, USA: ACM Press. 
 * doi:10.1145/997817.997857</li>
 * <li> Andoni, Alex (2005). 
 * <a href="http://www.mit.edu/~andoni/LSH/manual.pdf">E2LSH Manual 0.1</a></li>
 * </ul>
 * 
 * @author Edward Raff
 */
public class E2LSH<V extends Vec>
{
    private List<V> vecs;
    private DistanceMetric dm;
    private double radius;
    private double eps;
    private double p1;
    private double p2;
    private int w;
    private double c;
    /**
     * 1-delta is the probability of correctly selecting the neighbors within 
     * radius R
     */
    private double delta = Double.NaN;
    private int L;
    
    private int k;
    
    private List<Double> distCache;
    
    
    private Vec[][] h;
    private double[][] b;

    private List<Map<Integer, List<Integer>>> tables;

    /**
     * Creates a new LSH scheme for a given distance metric
     * @param vecs the set of vector to place into the LSH
     * @param radius the searchR radius for vectors
     * @param eps the approximation error, where vectors as fast as R(1+eps) are
     * likely to be returned. Must be positive. 
     * @param w the projection radius. If given a value &lt;= 0, a default value
     * of 4 will be used. 
     * @param k the number of hash functions to conjoin into the final hash per 
     * vector. If a value &lt;= 0 is given, a default value will be computed. 
     * @param delta (1-delta) will be the desired minimum probability of 
     * correctly selecting the correct nearest neighbor if there is only 1-NN 
     * within a distance of {@code radius}. It will be used to determine some 
     * number {@link #getL() } hash tables to reach the desired probability. 
     * 0.10 is a good value. 
     * @param dm the distance metric to use, must be {@link EuclideanDistance}
     * or {@link ManhattanDistance}. 
     * @param distCache the distance acceleration cache to use, if {@code null},
     * and it is supported, one will not be built. This is provided to a void 
     * redundant calculation when initializing multiple LSH tables using the 
     * same data set. 
     */
    public E2LSH(List<V> vecs, double radius, double eps, int w, int k, double delta, DistanceMetric dm, List<Double> distCache)
    {
        this.vecs = vecs;
        setRadius(radius);
        this.delta = delta;
        setEps(eps);
        if(w <= 0)
            this.w = 4;
        else
            this.w = w;
        setDistanceMetric(dm);
        this.distCache = distCache;

        if(k <= 0)
            this.k = (int) ceil(log(vecs.size())/log(1/p2));
        else
            this.k = k;
        
        if(delta <= 0 || delta >= 1)
            throw new IllegalArgumentException("dleta must be in range (0,1)");
        L = (int)ceil(log(1/delta)/-log(1-pow(p1, this.k)));
        
        
//        L = (int) ceil(pow(vecs.size(), log(1/p1)/log(1/p2)));
        
        Random rand = RandomUtil.getRandom();
        createTablesAndHashes(rand);
    }
    
    /**
     * Creates a new LSH scheme for a given distance metric
     * @param vecs the set of vector to place into the LSH
     * @param radius the searchR radius for vectors
     * @param eps the approximation error, where vectors as fast as R(1+eps) are
     * likely to be returned. Must be positive. 
     * @param w the projection radius. If given a value &lt;= 0, a default value
     * of 4 will be used. 
     * @param k the number of hash functions to conjoin into the final hash per 
     * vector. If a value &lt;= 0 is given, a default value will be computed. 
     * @param delta (1-delta) will be the desired minimum probability of 
     * correctly selecting the correct nearest neighbor if there is only 1-NN 
     * within a distance of {@code radius}. It will be used to determine some 
     * number {@link #getL() } hash tables to reach the desired probability. 
     * 0.10 is a good value. 
     * @param dm the distance metric to use, must be {@link EuclideanDistance}
     * or {@link ManhattanDistance}. 
     */
    public E2LSH(List<V> vecs, double radius, double eps, int w, int k, double delta, DistanceMetric dm)
    {
        this(vecs, radius, eps, w, k, delta, dm, dm.getAccelerationCache(vecs));
    }
    
    /**
     * Performs a search for points within the set {@link #getRadius() radius} 
     * of the query point. 
     * @param q the query point to search near
     * @return a list of vectors paired with their true distance from the query 
     * point that are within the desired radius of the query point
     */
    public List<? extends VecPaired<Vec, Double>> searchR(Vec q)
    {
        return searchR(q, false);
    }
        
    /**
     * Performs a search for points within the set {@link #getRadius() radius} 
     * of the query point. 
     * @param q the query point to search near
     * @param approx whether or not to return results in the approximate query 
     * range
     * @return a list of vectors paired with their true distance from the query 
     * point that are within the desired radius of the query point
     */
    public List<? extends VecPaired<Vec, Double>> searchR(Vec q, boolean approx)
    {
        List<VecPairedComparable<Vec, Double>> toRet = new ArrayList<VecPairedComparable<Vec, Double>>();
        
        Set<Integer> candidates = new IntSet();
        for (int l = 0; l < L; l++)
        {
            int hash = hash(l, q);
            List<Integer> list = tables.get(l).get(hash);
            for(int id : list)
                candidates.add(id);
        }
        
        final List<Double> q_qi = dm.getQueryInfo(q);
        
        final double R = approx ? radius*getC() : radius;
        for(int id : candidates)
        {
            double trueDist = dm.dist(id, q, q_qi, vecs, distCache);
            if(trueDist <= R)
                toRet.add(new VecPairedComparable<Vec, Double>(vecs.get(id), trueDist));
        }
        Collections.sort(toRet);
        return toRet;
    }
    
    private int hash(int l, Vec v)
    {
        final int[] vals = new int[k];
        
        for(int i = 0; i < k; i++)
            vals[i] = (int) floor(  ( (v.dot(h[l][i])/radius)+b[l][i])/w  );

        return Arrays.hashCode(vals);
    }

    private void setEps(double eps)
    {
        this.eps = eps;
        this.c = eps+1;
    }

    /**
     * Returns the multiplier used on the radius that controls the degree
     * of approximation. 
     * @return the radius approximation multiplier &gt; 1
     */
    public double getC()
    {
        return c;
    }
    
    /**
     * Returns the desired approximate radius for which to return results
     * @return the radius in use
     */
    public double getRadius()
    {
        return radius;
    }

    /**
     * Returns how many separate hash tables have been created for this distance
     * metric. 
     * @return the number of hash tables in use
     */
    public int getL()
    {
        return L;
    }
    
    /**
     * Returns the exact value value that should be used with the euclidean 
     * distance for the P2 probability . 
     * @param w the projection distance
     * @param c the approximation constant &gt; > 1
     * @return the exact P2 value to use
     */
    private static double getP2L2(double w, double c)
    {
        return 1 - 2 * Normal.cdf(-w/c, 0, 1)-2/(sqrt(2*PI)*w/c)*(1-exp(-w*w/(2*c*c)));
    }
    
    /**
     * Returns the exact value value that should be used with the manhattan 
     * distance for the P2 probability . 
     * @param w the projection distance
     * @param c the approximation constant &gt; > 1
     * @return the exact P2 value to use
     */
    private static double getP2L1(double w, double c)
    {
        return 2*atan(w/c)/PI-log(1+pow(w/c, 2))/(PI*w/c);
    }

    /**
     * Creates and initializes the tables of hash functions for {@link #h} and 
     * {@link #b}
     * @param rand source of randomness
     */
    private void createTablesAndHashes(Random rand)
    {
        int D = vecs.get(0).length();
        h = new Vec[L][k];
        b = new double[L][k];
        
        for(int l = 0; l < L; l++)
            for(int i = 0; i < k; i++)
            {
                DenseVector dv = new DenseVector(D);
                for(int j = 0; j < D; j++)
                    dv.set(j, rand.nextGaussian());
                h[l][i] = dv;
                b[l][i] = rand.nextDouble()*w;
            }
        
        tables = new ArrayList<Map<Integer, List<Integer>>>(L);
        for(int l = 0; l < L; l++)
        {
            tables.add(new HashMap<Integer, List<Integer>>());
            for(int id = 0; id < vecs.size(); id++)
            {
                int hash = hash(l, vecs.get(id));
                List<Integer> ints = tables.get(l).get(hash);
                if(ints == null)
                {
                    ints = new IntList(3);
                    tables.get(l).put(hash, ints);
                }
                ints.add(id);
            }
        }
    }

    /**
     * Sets the distance metric and {@link #p1} and {@link #p2}. Must be called 
     * after {@link #setEps(double) } and {@link #w} are set. 
     * @param dm the distance metric to use
     */
    private void setDistanceMetric(DistanceMetric dm)
    {
        if(dm instanceof EuclideanDistance || dm instanceof ManhattanDistance)
        {
            this.dm = dm;
            if(dm instanceof EuclideanDistance)
            {
                p1 = getP2L2(w, 1);
                p2 = getP2L2(w, c);
            }
            else
            {
                p1 = getP2L1(w, 1);
                p2 = getP2L1(w, c);
            }
        }
        else
            throw new IllegalArgumentException("only Euclidean and Manhatan (L1 and L2 norm) distances are supported");
    }

    private void setRadius(double radius)
    {
        if(Double.isInfinite(radius) || Double.isNaN(radius) || radius <= 0)
            throw new IllegalArgumentException("Radius must be a positive constant, not " + radius);
        this.radius = radius;
    }
    
}
