
package jsat.clustering.kmeans;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.ClusterFailureException;
import jsat.clustering.KClustererBase;
import jsat.clustering.PAM;
import jsat.clustering.SeedSelectionMethods;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.math.OnLineStatistics;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.*;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.random.RandomUtil;

/**
 * Base class for the numerous implementations of k-means that exist. This base
 * class provides an slow heuristic approach to the selection of k. 
 * 
 * @author Edward Raff
 */
public abstract class KMeans extends KClustererBase implements Parameterized
{

	private static final long serialVersionUID = 8730927112084289722L;

	/**
     * This is the default seed selection method used in ElkanKMeans. When used with 
     * the {@link EuclideanDistance}, it selects seeds that are log optimal with
     * a high probability. 
     */
    public static final SeedSelectionMethods.SeedSelection DEFAULT_SEED_SELECTION = SeedSelectionMethods.SeedSelection.KPP;
    
    @ParameterHolder
    protected DistanceMetric dm;
    protected SeedSelectionMethods.SeedSelection seedSelection;
    protected Random rand;
    
    /**
     * Indicates whether or not the means from the clustering should be saved
     */
    protected boolean storeMeans = true;
    /**
     * Indicates whether or not the distance between a datapoint and its nearest
     * centroid should be saved after clustering. This only applies when the 
     * error of the model is requested 
     */
    protected boolean saveCentroidDistance = true;
    
    /**
     * Distance from a datapoint to its nearest centroid. May be an approximate 
     * distance 
     */
    protected double[] nearestCentroidDist;
    
    /**
     * The list of means
     */
    protected List<Vec> means;
    
    /**
     * Control the maximum number of iterations to perform. 
     */
    protected int MaxIterLimit = Integer.MAX_VALUE;

    public KMeans(DistanceMetric dm, SeedSelectionMethods.SeedSelection seedSelection, Random rand)
    {
        this.dm = dm;
        setSeedSelection(seedSelection);
        this.rand = rand;
    }

    /**
     * Copy constructor
     * @param toCopy 
     */
    public KMeans(KMeans toCopy)
    {
        this.dm = toCopy.dm.clone();
        this.seedSelection = toCopy.seedSelection;
        this.rand = RandomUtil.getRandom();
        if (toCopy.nearestCentroidDist != null)
            this.nearestCentroidDist = Arrays.copyOf(toCopy.nearestCentroidDist, toCopy.nearestCentroidDist.length);
        if (toCopy.means != null)
        {
            this.means = new ArrayList<Vec>(toCopy.means.size());
            for (Vec v : toCopy.means)
                this.means.add(v.clone());
        }
    }
    
    /**
     * Sets the maximum number of iterations allowed
     * @param iterLimit the maximum number of iterations of the ElkanKMeans algorithm 
     */
    public void setIterationLimit(int iterLimit)
    {
        if(iterLimit < 1)
            throw new IllegalArgumentException("Iterations must be a positive value, not " + iterLimit);
        this.MaxIterLimit = iterLimit;
    }

    /**
     * Returns the maximum number of iterations of the ElkanKMeans algorithm that will be performed. 
     * @return the maximum number of iterations of the ElkanKMeans algorithm that will be performed. 
     */
    public int getIterationLimit()
    {
        return MaxIterLimit;
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
     * Sets the method of seed selection to use for this algorithm. {@link SeedSelection#KPP} is recommended for this algorithm in particular. 
     * @param seedSelection the method of seed selection to use
     */
    public void setSeedSelection(SeedSelectionMethods.SeedSelection seedSelection)
    {
        this.seedSelection = seedSelection;
    }

    /**
     * 
     * @return the method of seed selection used
     */
    public SeedSelectionMethods.SeedSelection getSeedSelection()
    {
        return seedSelection;
    }

    /**
     * Returns the distance metric in use
     * @return the distance metric in use
     */
    public DistanceMetric getDistanceMetric()
    {
        return dm;
    }
    
    /**
     * This is a helper method where the actual cluster is performed. This is because there
     * are multiple strategies for modifying kmeans, but all of them require this step. 
     * <br>
     * The distance metric used is trained if needed
     * 
     * @param dataSet The set of data points to perform clustering on
     * @param accelCache acceleration cache to use, or {@code null}. If
     * {@code null}, the kmeans code will attempt to create one
     * @param k the number of clusters
     * @param means the initial points to use as the means. Its length is the
     * number of means that will be searched for. These means will be altered,
     * and should contain deep copies of the points they were drawn from. May be
     * empty, in which case the list will be filled with some selected means
     * @param assignment an empty temp space to store the clustering
     * classifications. Should be the same length as the number of data points
     * @param exactTotal determines how the objective function (return value)
     * will be computed. If true, extra work will be done to compute the exact
     * distance from each data point to its cluster. If false, an upper bound
     * approximation will be used. This also impacts the value stored in 
     * {@link #nearestCentroidDist}
     * @param threadpool the source of threads for parallel computation. If
     * <tt>null</tt>, single threaded execution will occur
     * @param returnError {@code true} is the sum of squared distances should be
     * returned. {@code false} means any value can be returned. 
     * {@link #saveCentroidDistance} only applies if this is {@code true}
     * @param dataPointWeights the weight value to use for each data point. If
     * <tt>null</tt>, assume each point has equal weight.
     * @return the double
     */
    abstract protected double cluster(final DataSet dataSet, List<Double> accelCache, final int k, final List<Vec> means, final int[] assignment, boolean exactTotal, ExecutorService threadpool, boolean returnError, Vec dataPointWeights);
    
    static protected List<List<DataPoint>> getListOfLists(int k)
    {
        List<List<DataPoint>> ks = new ArrayList<List<DataPoint>>(k);
        for(int i = 0; i < k; i++)
            ks.add(new ArrayList<DataPoint>());
        return ks;
    }

    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()/2), designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()/2), threadpool, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, ExecutorService threadpool, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        if(dataSet.getSampleSize() < clusters)
            throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");
        
        means = new ArrayList<Vec>(clusters);
        cluster(dataSet, null, clusters, means, designations, false, threadpool, false, null);
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
        means = new ArrayList<Vec>(clusters);
        cluster(dataSet, null, clusters, means, designations, false, null, false, null);
        if(!storeMeans)
            means = null;
        
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
            rand = RandomUtil.getRandom();
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
            result = cluster(dataSet, null, k, new ArrayList<Vec>(), clusterIDs, true, null, true, null);
            putSelf.add(this);
        }
        
    }
    
    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
        if(dataSet.getSampleSize() < highK)
            throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");
        
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
        
        return findK(lowK, highK, totDistances, dataSet, designations);
    }
    
    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        /**
         * Stores the cluster ids associated with each data point
         */
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        if(dataSet.getSampleSize() < highK)
            throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");
        
        double[] totDistances = new double[highK-lowK+1];

        for(int i = lowK; i <= highK; i++)
            totDistances[i-lowK] = cluster(dataSet, null, i, new ArrayList<Vec>(), designations, true, null, true, null);

        return findK(lowK, highK, totDistances, dataSet, designations);
    }
    
    private int[] findK(int lowK, int highK, double[] totDistances, DataSet dataSet, int[] designations)
    {
        //Now we process the distance changes
        /**
         * Keep track of the changes
         */
        OnLineStatistics stats = new OnLineStatistics();
        
        double maxChange = Double.MIN_VALUE;
        int maxChangeK = lowK;

        for(int i = lowK; i <= highK; i++)
        {
            double totDist = totDistances[i-lowK];
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
            double tmp;
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

    @Override
    abstract public KMeans clone();

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }
    
    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
}
