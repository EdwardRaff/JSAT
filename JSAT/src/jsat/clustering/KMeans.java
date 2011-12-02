
package jsat.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.math.OnLineStatistics;
import jsat.utils.PairedReturn;
import jsat.utils.SystemInfo;
import org.omg.CORBA.INTERNAL;

/**
 *
 * @author Edward Raff
 */
public class KMeans implements Clusterer
{
    static public enum SeedSelection 
    {
        /**
         * The seed values will be randomly selected from the data set
         */
        RANDOM, 
        
        /**
         * The k-means++ seeding algo: <br>
         * The seed values will be probabilistically selected from the 
         * data set. <br>
         * The solution is O(log(k)) competitive with the 
         * optimal k clustering when using {@link EuclideanDistance}. 
         * <br><br>
         * See k-means++: The Advantages of Careful Seeding
         */
        KPP
    };
    
    protected DistanceMetric dm;
    protected Random rand;
    private SeedSelection seedSelection;

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
        Arrays.fill(catTrack, -1);//-1, invalid category!
        
        
        int[] clusterMemberCounts = new int[initialMeans.size()];
        
        do
        {
            if(changes != -1)//Update the means
            {
                Arrays.fill(clusterMemberCounts, 0);
                
                for(Vec vec : initialMeans)
                    vec.zeroOut();
                for(int i = 0; i < dataSet.getSampleSize(); i++)
                {
                    Vec dpVec = dataSet.getDataPoint(i).getNumericalValues();
                    initialMeans.get(catTrack[i]).mutableAdd(dpVec);//add the the mean
                    clusterMemberCounts[catTrack[i]]++;//count the assignments
                }
                
                //Divide by the coutns
                for(int i = 0; i < initialMeans.size(); i++)
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
                totalDistance += minDist*minDist;
            }

        }
        while(changes != 0);
        
        
        
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

    private List<List<DataPoint>> getListOfLists(int k)
    {
        List<List<DataPoint>> ks = new ArrayList<List<DataPoint>>(k);
        for(int i = 0; i < k; i++)
            ks.add(new ArrayList<DataPoint>());
        return ks;
    }
    
    //We use the object itself to return the k 
    private class ClusterKCallable implements Callable<PairedReturn<ClusterKCallable, Double>>
    {
        private DataSet dataSet;
        private int k;
        int[] clusterIDs;
        private Random rand;

        public ClusterKCallable(DataSet dataSet, int k)
        {
            this.dataSet = dataSet;
            this.k = k;
            clusterIDs = new int[dataSet.getSampleSize()];
            rand = new Random();
        }

        public ClusterKCallable(DataSet dataSet)
        {
            this(dataSet, 2);
        }

        public void setK(int k)
        {
            this.k = k;
        }

        public int getK()
        {
            return k;
        }
        
        public PairedReturn<ClusterKCallable, Double> call() throws Exception
        {
            double dist = cluster(dataSet, selectIntialPoints(dataSet, k, dm, rand, seedSelection), clusterIDs);
            
            return new PairedReturn<ClusterKCallable, Double>(this, dist);
        }
        
    }

    public List<List<DataPoint>> cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool)
    {
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
        
        Queue<ClusterKCallable> callables = new LinkedList<ClusterKCallable>();
        Queue<Future<PairedReturn<ClusterKCallable, Double>>> futures = new LinkedList<Future<PairedReturn<ClusterKCallable, Double>>>();
        
        for(int i = 0; i < SystemInfo.LogicalCores; i++)
            callables.add(new ClusterKCallable(dataSet));
        
        int inUse = 0;
        int j;
        for(j = lowK; j <= highK && inUse < SystemInfo.LogicalCores; j++)
        {
            callables.peek().setK(j);
            futures.add(threadpool.submit(callables.remove()));
            inUse++;
        }
        try
        {
            while (!futures.isEmpty())
            {
                PairedReturn<ClusterKCallable, Double> pr = futures.remove().get();

                ClusterKCallable ckc = pr.getFirstItem();
                double totDist = pr.getSecondItem();
                int i = ckc.getK();
                
                totDistances[i - lowK] = totDist;

                if (i > lowK)
                {
                    double change = Math.abs(totDist - totDistances[i - lowK - 1]);
                    stats.add(change);
                    if (change > maxChange)
                    {
                        maxChange = change;
                        maxChangeK = i;
                    }
                }
                
                if(j <= highK)
                {
                    ckc.setK(j);
                    j++;
                    futures.add(threadpool.submit(ckc));
                }
            }
        }
        catch (InterruptedException ex)
        {
            ex.printStackTrace();
        }
        catch (ExecutionException ex)
        {
            ex.printStackTrace();
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
    
    static protected List<Vec> selectIntialPoints(DataSet d, int k, DistanceMetric dm, Random rand, SeedSelection selectionMethod)
    {
        ArrayList<Vec> means = new ArrayList<Vec>(k);
        
        if(selectionMethod ==  SeedSelection.RANDOM)
        {
            Set<Integer> indecies = new HashSet<Integer>(k);

            while(indecies.size() != k)//Keep sampling, we cant use the same point twice. 
                indecies.add(rand.nextInt(d.getSampleSize()));//TODO create method to do uniform sampleling for a select range

            for(Integer i : indecies)
                means.add(d.getDataPoint(i).getNumericalValues().copy());
        }
        else if(selectionMethod == SeedSelection.KPP)
        {
            /*
             * http://www.stanford.edu/~darthur/kMeansPlusPlus.pdf : k-means++: The Advantages of Careful Seeding
             * 
             */
            //Initial random point
            means.add(d.getDataPoint(rand.nextInt(d.getSampleSize())).getNumericalValues().copy());
            
            double[] closestDist = new double[d.getSampleSize()];
            double sqrdDistSum = 0.0;
            double newDist;
            while(means.size() < k)
            {
                newDist = Double.MAX_VALUE;
                
                //Compute the distance from each data point to the closest mean
                Vec newMean = means.get(means.size()-1);//Only the most recently added mean needs to get distances computed. 
                for(int i = 0; i < d.getSampleSize(); i++)
                {
                    newDist = dm.dist(newMean, d.getDataPoint(i).getNumericalValues());
                    
                    if(newDist < closestDist[i] || means.size() == 1)
                    {
                        newDist*=newDist;
                        sqrdDistSum -= closestDist[i];//on inital, -= 0  changes nothing. on others, removed the old value
                        sqrdDistSum += newDist;
                        closestDist[i] = newDist;
                    }
                }
                
                //Choose new x as weighted probablity by the squared distances
                double rndX = rand.nextDouble()*sqrdDistSum;
                double searchSum = 0;
                int i = -1;
                while(searchSum < rndX && i < d.getSampleSize()-1)/// rndX = 5, 
                    searchSum += closestDist[++i];
                
                means.add(d.getDataPoint(i).getNumericalValues().copy());
            }
            
            
        }
        
        return means;
    }
    
}
