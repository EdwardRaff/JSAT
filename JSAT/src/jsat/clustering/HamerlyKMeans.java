package jsat.clustering;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import static jsat.clustering.SeedSelectionMethods.selectIntialPoints;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;

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
    
    /**
     * Performs the main clustering work
     * @param dataSet the data set to cluster
     * @param assignment the array to store assignments in
     * @param exactTotal not used at the moment. 
     */
    protected void cluster(DataSet dataSet, final int[] assignment, boolean exactTotal)
    {
        final int k = means.size();
        final int N = dataSet.getSampleSize();
        
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
        final int[] q = new int[k];
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
        
        //Start of algo
        Initialize(dataSet, q, means, tmpVecs, cP, u, l, assignment);
        int updates = N;
        while(updates > 0)
        {
            moveCenters(means, tmpVecs, cP, q, p);
            UpdateBounds(p, assignment, u, l);
            updates = 0;
            for(int j = 0; j < means.size(); j++)
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
            
            for(int i = 0; i < N; i++)
            {
                final int a_i = assignment[i];
                double m = Math.max(s[a_i]/2, l[i]);
                if(u[i] > m)//first bound test
                {
                    Vec x = x(dataSet, i);
                    u[i] = dm.dist(x, means.get(a_i));//tighten upper bound
                    if(u[i] > m)  //second bound test
                    {
                        final int new_a_i = PointAllCtrs(x, i, means, assignment, u, l);
                        if(a_i != new_a_i)
                        {
                            updates++;
                            q[a_i]--;
                            q[new_a_i]++;
                            cP[a_i].mutableSubtract(x);
                            cP[new_a_i].mutableAdd(x);
                        }
                    }
                }
            }
        }
    }
    
    private void Initialize(DataSet d, int[] q, List<Vec> means, Vec[] tmp, Vec[] cP, double[] u, double[] l, int[] a)
    {
        for(int j = 0; j < means.size(); j++)
        {
            //q would already be initalized to zero on creation by java
            cP[j] = new DenseVector(means.get(0).length());
            tmp[j] = cP[j].clone();
        }
        
        for(int i = 0; i < u.length; i++)
        {
            Vec x = x(d, i);
            int j = PointAllCtrs(x, i, means, a, u, l);
            q[j]++;
            cP[j].mutableAdd(x);
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
    private int PointAllCtrs(Vec x, int i, List<Vec> means, int[] a, double[] u, double[] l )
    {
        double secondLowest = Double.POSITIVE_INFINITY;
        int slIndex = -1;
        double lowest = Double.MAX_VALUE;
        int lIndex = -1;
        
        for(int j = 0; j < means.size(); j++)
        {
            double dist = dm.dist(x, means.get(j));
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
    
    private void moveCenters(List<Vec> means, Vec[] tmpSpace, Vec[] cP, int[] q, double[] p)
    {
        for(int j = 0; j < means.size(); j++)
        {
            //compute new mean
            cP[j].copyTo(tmpSpace[j]);
            tmpSpace[j].mutableDivide(q[j]);
            //compute distance betwean new and old
            p[j] = dm.dist(means.get(j), tmpSpace[j]);
            //move it to its positaiotn as new mean
            tmpSpace[j].copyTo(means.get(j));
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
        
        means = selectIntialPoints(dataSet, clusters, dm, new Random(), seedSelection, threadpool);
        cluster(dataSet, designations, false);
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
        
        means = selectIntialPoints(dataSet, clusters, dm, new Random(), seedSelection);
        cluster(dataSet, designations, false);
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
