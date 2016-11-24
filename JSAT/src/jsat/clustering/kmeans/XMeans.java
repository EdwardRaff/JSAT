
package jsat.clustering.kmeans;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import jsat.clustering.SeedSelectionMethods;
import jsat.linear.*;
import static java.lang.Math.*;

/**
 * This class provides a method of performing {@link KMeans} clustering when the
 * value of {@code K} is not known. It works by recursively splitting means 
 * up to some specified maximum.
 * value. <br>
 * <br>
 * When the value of {@code K} is specified, the implementation will simply call
 * the regular KMeans object it was constructed with. <br>
 * <br>
 * Note, that specifying a minimum value of {@code K=1} has a tendency to not be
 * split by the algorithm, returning the naive result of 1 cluster. It is better 
 * to use at least {@code K=2} as the default minimum, which is what the 
 * implementation will start from when no range of {@code K} is given. <br>
 * <br>
 * See: Pelleg, D.,&amp;Moore, A. (2000). <i>X-means: Extending K-means with 
 * Efficient Estimation of the Number of Clusters</i>. In ICML (pp. 727–734).
 * San Francisco, CA, USA: Morgan Kaufmann Publishers Inc. Retrieved from 
 * <a href="http://pdf.aminer.org/000/335/443/x_means_extending_k_means_with_efficient_estimation_of_the.pdf">
 * here</a>
 *
 * @author Edward Raff
 */
public class XMeans extends KMeans 
{
    private static final long serialVersionUID = -2577160317892141870L;
    private boolean stopAfterFail = false;
    private boolean iterativeRefine = true;
    
    private int minClusterSize = 25;
    private KMeans kmeans;

    public XMeans()
    {
        this(new HamerlyKMeans());
    }
    
    public XMeans(KMeans kmeans)
    {
        super(kmeans.dm, kmeans.seedSelection, kmeans.rand);
        this.kmeans = kmeans;
        this.kmeans.saveCentroidDistance = true;
        this.kmeans.setStoreMeans(true);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public XMeans(XMeans toCopy)
    {
        super(toCopy);
        this.kmeans = toCopy.kmeans.clone();
        this.stopAfterFail = toCopy.stopAfterFail;
        this.iterativeRefine = toCopy.iterativeRefine;
        this.minClusterSize = toCopy.minClusterSize;
    }
    
    /**
     * Each new cluster will be tested for improvement according to the BIC 
     * metric. If this is set to {@code true} then an optimization is done that 
     * once a center fails be improved by splitting, it will never be tested 
     * again. This is a safe assumption when
     * {@link #setIterativeRefine(boolean) } is set to {@code false}, but 
     * otherwise may not quite be true. <br>
     * <br>
     * When {@code trustH0} is {@code true} , X-Means will 
     * make at most O(k) runs of k-means for the final value of k chosen. When 
     * {@code false} (the default option), at most O(k<sup>2</sup>) runs of
     * k-means will occur. 
     * 
     * @param stopAfterFail {@code true} if a centroid shouldn't be re-tested once it 
     * fails to split. 
     */
    public void setStopAfterFail(boolean stopAfterFail)
    {
        this.stopAfterFail = stopAfterFail;
    }

    /**
     * 
     * @return {@code true} if clusters that fail to split wont be re-tested. 
     * {@code false} if they will. 
     */
    public boolean isStopAfterFail()
    {
        return stopAfterFail;
    }
    
    /**
     * Sets the minimum size for splitting a cluster. 
     * @param minClusterSize the minimum number of data points that must be present in a 
     * cluster to consider splitting it
     */
    public void setMinClusterSize(int minClusterSize)
    {
        if(minClusterSize < 2)
            throw new IllegalArgumentException("min cluster size that could be split is 2, not " + minClusterSize);
        this.minClusterSize = minClusterSize;
    }

    /**
     * 
     * @return the minimum number of data points that must be present in a 
     * cluster to consider splitting it
     */
    public int getMinClusterSize()
    {
        return minClusterSize;
    }

    /**
     * Sets whether or not the set of all cluster centers should be refined at 
     * every iteration. By default this is {@code true} and part of how the 
     * X-Means algorithm is described. Setting this to {@code false} can result
     * in large speedups at the potential cost of quality. 
     * @param refineCenters {@code true} to refine the cluster centers at every 
     * step, {@code false} to skip this step of the algorithm. 
     */
    public void setIterativeRefine(boolean refineCenters)
    {
        this.iterativeRefine = refineCenters;
    }

    /**
     * 
     * @return {@code true} if the cluster centers are refined at every 
     * step, {@code false} if skipping this step of the algorithm. 
     */
    public boolean getIterativeRefine()
    {
        return iterativeRefine;
    }
    

    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        return cluster(dataSet, 2, Math.max(dataSet.getSampleSize()/20, 10), designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, 2, Math.max(dataSet.getSampleSize()/20, 10), threadpool, designations);
    }
    
    /**
     * "p_j is simply the sum of K- 1 class probabilities, M * K centroid 
     * coordinates, and one variance estimate."
     * 
     * @param K the number of clusters
     * @param D the number of dimensions
     * @return the number of free parameters
     */
    private static int freeParameters(int K, int D)
    {
        return (K - 1) + (D * K) + 1;
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
        final int N = dataSet.getSampleSize();
        final int D = dataSet.getNumNumericalVars();//"M" in orig paper
        
        if(designations == null || designations.length < dataSet.getSampleSize())
            designations = new int[N];
        
        List<Vec> data = dataSet.getDataVectors();
        final List<Double> accelCache = dm.getAccelerationCache(data, threadpool);
        
        /**
         * The sum of ||x - \mu_i||^2 for each cluster currently kept
         */
        double[] localVar = new double[highK];
        int[] localOwned = new int[highK];
        //initiate
        if(lowK >= 2)
        {
            means = new ArrayList<Vec>();
            kmeans.cluster(dataSet, accelCache, lowK, means, designations, true, threadpool, true, null);
            for(int i = 0; i < data.size(); i++)
            {
                localVar[designations[i]] += Math.pow(kmeans.nearestCentroidDist[i], 2);
                localOwned[designations[i]]++;
            }
        }
        else//1 mean of all the data
        {
            if(designations == null || designations.length < N)
                designations = new int[N];
            else
                Arrays.fill(designations, 0);
            means = new ArrayList<Vec>(Arrays.asList(MatrixStatistics.meanVector(dataSet)));
            localOwned[0] = N;
            List<Double> qi = dm.getQueryInfo(means.get(0));
            for(int i = 0; i < data.size(); i++)
                localVar[0] += Math.pow(dm.dist(i, means.get(0), qi, data, accelCache), 2);
        }
        
        int[] subS = new int[designations.length];
        int[] subC = new int[designations.length];
        
        //tract if we should stop testing a mean or not
        List<Boolean> dontRedo = new ArrayList<Boolean>(Collections.nCopies(means.size(), false));
        
        int origMeans;
        do
        {
            origMeans = means.size();
            
            for(int c = 0; c < origMeans; c++)
            {
                if(dontRedo.get(c))
                    continue;
                /*
                 * Next, in each parent region we run a local K-means (with 
                 * K = 2) for each pair of children. It is local in that the 
                 * children are fighting each other for the points in the 
                 * parent's region: no others
                 */
                
                List<DataPoint> X = getDatapointsFromCluster(c, designations, dataSet, subS);
                final int n = X.size();//NOTE, not the same as N. PAY ATENTION
                //TODO add the optimization in the paper where we check for movment, and dont test means that haven't mvoed much
                if(X.size() < minClusterSize || means.size() == highK)
                    continue;//this loop with force it to exit when we hit max K
                
                subC = kmeans.cluster(new SimpleDataSet(X), 2, threadpool, subC);
                //call explicitly to force that distance to nearest center is saved
                List<Vec> subMean = new ArrayList<Vec>(2);
                kmeans.cluster(new SimpleDataSet(X), null, 2, subMean, subC, true, threadpool, true, null);
                double[] nearDist = kmeans.nearestCentroidDist;
                Vec c1 = subMean.get(0);
                Vec c2 = subMean.get(1);
                
                /*
                 * "it determines which one to explore by improving the BIC 
                 * locally in each region."
                 * so we only compute BIC from local information
                 */
                
                double newSigma = 0;
                int size_c1 = 0;
                for(int i = 0; i < X.size(); i++)
                {
                    newSigma += Math.pow(nearDist[i], 2);
                    if(subC[i] == 0)
                        size_c1++;
                }
                newSigma /= D*(n-2);
                int size_c2 = n-size_c1;
    
                //have needed values, now compute BIC for LOCAL models
                
                double localNewBic = size_c1*log(size_c1) + size_c2*log(size_c2)
                        - n*log(n)
                        - n*D/2.0*log(2*PI*newSigma)
                        - D/2.0*(n-2)//that gets us the log like, last line to penalize for bic
                        -freeParameters(2, D)/2.0*log(n);
                
                double localOldBic = 
                        - n*D/2.0*log(2*PI*localVar[c]/(D*(n-1)))
                        - D/2.0*(n-1)//that gets us the log like, last line to penalize for bic
                        -freeParameters(1, D)/2.0*log(n);

                if(localOldBic > localNewBic)
                {
                    if(stopAfterFail)//if we are going to trust that H0 is true forever, mark it
                        dontRedo.set(c, true);
                    continue;//passed the test, do not split
                }
                //else, accept the split

                //first, update assignment array. Cluster '0' stays as is, re-set cluster '1'
                for(int i = 0; i < X.size(); i++)
                    if(subC[i] == 1)
                        designations[subS[i]] = means.size();
                //replace current mean and add new one
                means.set(c, c1.clone());//cur index in dontRedo stays false
                means.add(c2.clone());//add a 'false' for new center
                dontRedo.add(false);
            }
            //"Between each round of splitting, we run k-means on the entire dataset and all the centers to refine the current solution"
            if(iterativeRefine && means.size() > 1)
            {
                kmeans.cluster(dataSet, accelCache, means.size(), means, designations, true, threadpool, true, null);
                Arrays.fill(localVar, 0.0);
                Arrays.fill(localOwned, 0);
                for(int i = 0; i < data.size(); i++)
                {
                    localVar[designations[i]] += Math.pow(kmeans.nearestCentroidDist[i], 2);
                    localOwned[designations[i]]++;
                }
            }
        }
        while (origMeans < means.size());
        
        if(!iterativeRefine)//if we havn't been refining we need to do so now!
            kmeans.cluster(dataSet, accelCache, means.size(), means, designations, false, threadpool, false, null);
        return designations;
    }
    
    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        return cluster(dataSet, lowK, highK, null, designations);
    }

    @Override
    public int getIterationLimit()
    {
        return kmeans.getIterationLimit();
    }

    @Override
    public void setIterationLimit(int iterLimit)
    {
        kmeans.setIterationLimit(iterLimit);
    }

    @Override
    public void setSeedSelection(SeedSelectionMethods.SeedSelection seedSelection)
    {
        if(kmeans != null)//needed when initing
            kmeans.setSeedSelection(seedSelection);
    }

    @Override
    public SeedSelectionMethods.SeedSelection getSeedSelection()
    {
        return kmeans.getSeedSelection();
    }
    

    @Override
    protected double cluster(DataSet dataSet, List<Double> accelCache, int k, List<Vec> means, int[] assignment, boolean exactTotal, ExecutorService threadpool, boolean returnError, Vec dataPointWeights)
    {
        return kmeans.cluster(dataSet, accelCache, k, means, assignment, exactTotal, threadpool, returnError, null);
    }

    @Override
    public XMeans clone()
    {
        return new XMeans(this);
    }
}
