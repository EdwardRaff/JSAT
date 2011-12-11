
package jsat.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.math.OnLineStatistics;
import jsat.utils.SystemInfo;
import static jsat.clustering.SeedSelectionMethods.*;

/**
 *
 * @author Edward Raff
 */
public class KMeans implements KClusterer
{
    private DistanceMetric dm;
    private Random rand;
    private SeedSelection seedSelection;
    /**
     * Variable used to control used to combat local optimia. 
     * KMeans will be run once for each value of this variable,
     * the best solution with the smallest sum of squared 
     * error chosen. 
     */
    protected int repeats = 1;
    
    /**
     * Controll the maximum number of iterations to perform. 
     */
    protected int iterLimit = 100;

    public KMeans(DistanceMetric dm, Random rand, SeedSelection seedSelection)
    {
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
     * @param iterLimit 
     */
    public void setIterationLimit(int iterLimit)
    {
        this.iterLimit = iterLimit;
    }

    public int getIterationLimit()
    {
        return iterLimit;
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
    
    
    public List<List<DataPoint>> cluster(DataSet dataSet)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()/2));
    }
    
    public List<List<DataPoint>> cluster(DataSet dataSet, ExecutorService threadpool)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()/2), threadpool);
    }
    
    /**
     * This is a helper method where the actual cluster is performed. This is because there
     * are multiple strategies for modifying kmeans, but all of them require this step. 
     * 
     * ks must be at least the same size as initialMeans. It can be larger, and those spaces will be ignored. 
     * 
     * tmp should be provided with enough space for every value. The values will be copied to it in-between iterations.  
     * 
     * 
     * @param dataSet The set of data points to perform clustering on
     * @param initialMeans the initial points to use as the means. Its
     * length is the number of means that will be searched for.
     * @param ks a list of empty lists to store the clusters in, each list corresponding to a different cluster. 
     * @param catTrack an empty temp space to store the clustering classifications
     * 
     * @return the sum of squares distances from each data point to its closest cluster
     */
    protected double cluster(final DataSet dataSet, final List<Vec> initialMeans, final int[] catTrack)
    {   
        double totalDistance = 0;
        int changes = -1;
        
        double bestTotal = Double.MAX_VALUE;
        int[] bestCats = Arrays.copyOf(catTrack, catTrack.length);
        
        int[] clusterMemberCounts = new int[initialMeans.size()];
        
        for (int r = 0; r < repeats; r++)
        {
            changes = -1;
            Arrays.fill(catTrack, -1);//-1, invalid category!
            int iter = 0;
            
            do
            {
                if (changes != -1)//Update the means
                {
                    Arrays.fill(clusterMemberCounts, 0);

                    for (Vec vec : initialMeans)
                        vec.zeroOut();
                    for (int i = 0; i < dataSet.getSampleSize(); i++)
                    {
                        Vec dpVec = dataSet.getDataPoint(i).getNumericalValues();
                        initialMeans.get(catTrack[i]).mutableAdd(dpVec);//add the the mean
                        clusterMemberCounts[catTrack[i]]++;//count the assignments
                    }

                    //Divide by the coutns
                    for (int i = 0; i < initialMeans.size(); i++)
                        initialMeans.get(i).mutableDivide(clusterMemberCounts[i]);
                }

                //Set up to start grouping
                changes = 0;
                totalDistance = 0.0;

                for (int i = 0; i < dataSet.getSampleSize(); i++)
                {
                    Vec dpVec = dataSet.getDataPoint(i).getNumericalValues();
                    int cat = 0;
                    double minDist = dm.dist(initialMeans.get(0), dpVec);

                    for (int k = 1; k < initialMeans.size(); k++)
                    {
                        double dist = dm.dist(initialMeans.get(k), dpVec);
                        if (dist < minDist)
                        {
                            minDist = dist;
                            cat = k;
                        }
                    }

                    //Update which cluster it is in
                    if (catTrack[i] != cat)
                    {
                        changes++;
                        catTrack[i] = cat;
                    }
                    totalDistance += minDist * minDist;
                    
                }

            }
            while (changes != 0 && iter++ < iterLimit);
            
            if(totalDistance < bestTotal)
            {
                bestTotal = totalDistance;
                System.arraycopy(catTrack, 0, bestCats, 0, catTrack.length);
            }
        }
        
        System.arraycopy(bestCats, 0, catTrack , 0, catTrack.length);

        return totalDistance;
    }

    public List<List<DataPoint>> cluster(DataSet dataSet, int clusters, ExecutorService threadpool)
    {
        return cluster(dataSet, clusters);
    }

    public List<List<DataPoint>> cluster(DataSet dataSet, int clusters)
    {
        List<List<DataPoint>> ks = getListOfLists(clusters);
        
        /**
         * Stores the cluster ids associated with each data point
         */
        int[] clusterIDs = new int[dataSet.getSampleSize()];
        
        cluster(dataSet, selectIntialPoints(dataSet, clusters, dm, rand, seedSelection), clusterIDs);
        
        for(int i = 0; i < clusterIDs.length; i++)
            ks.get(clusterIDs[i]).add(dataSet.getDataPoint(i));
        
        return ks;
    }

    static protected List<List<DataPoint>> getListOfLists(int k)
    {
        List<List<DataPoint>> ks = new ArrayList<List<DataPoint>>(k);
        for(int i = 0; i < k; i++)
            ks.add(new ArrayList<DataPoint>());
        return ks;
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
            result = cluster(dataSet, selectIntialPoints(dataSet, k, dm, rand, seedSelection), clusterIDs);
            putSelf.add(this);
        }
        
    }

    public List<List<DataPoint>> cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool)
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
        
        System.out.println(Arrays.toString(totDistances));         
        
        return cluster(dataSet, maxChangeK);
    }

    public List<List<DataPoint>> cluster(DataSet dataSet, int lowK, int highK)
    {
        /**
         * Stores the cluster ids associated with each data point
         */
        int[] clusterIDs = new int[dataSet.getSampleSize()];

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
            double totDist = cluster(dataSet, selectIntialPoints(dataSet, i, dm, rand, seedSelection), clusterIDs);
            totDistances[i-lowK] = totDist;
            System.out.println("Finished " + i + "/ " + highK);
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
            for(int i = 1; i < totDistances.length; i++)
            {
                if(Math.abs(totDistances[i]-totDistances[i-1]) < changeMean )
                {
                    maxChangeK = i+lowK;
                    break;
                }
            }
        }
        
        
        return cluster(dataSet, maxChangeK);
    }
}
