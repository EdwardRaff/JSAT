package jsat.clustering;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.SynchronousQueue;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.KDTree;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionFactory;
import jsat.math.OnLineStatistics;
import jsat.utils.SystemInfo;

/**
 * A density-based algorithm for discovering clusters in large spatial databases 
 * with noise (1996) by Martin Ester , Hans-peter Kriegel , Jörg S , Xiaowei Xu
 * 
 * @author Edward Raff
 */
public class DBSCAN implements Clusterer
{
    /**
     * Used by {@link #cluster(jsat.DataSet, double, int, jsat.linear.vectorcollection.VectorCollection) } 
     * to mark that a data point as not yet been visited. <br>
     * Clusters that have been visited have a value >= 0, that indicates their cluster. Or have the value {@link #NOISE}
     */
    private static final int UNCLASSIFIED = -1;
    /**
     * Used by {@link #cluster(jsat.DataSet, double, int, jsat.linear.vectorcollection.VectorCollection) } 
     * to mark that a data point has been visited, but was considered noise. 
     */
    private static final int NOISE = -2;
    
    /**
     * Factory used to create a vector space of the inputs. 
     * The paired Integer is the vector's index in the original dataset
     */
    private VectorCollectionFactory<VecPaired<Integer, Vec> > vecFactory;
    private DistanceMetric dm;

    public DBSCAN(DistanceMetric dm, VectorCollectionFactory<VecPaired<Integer, Vec>> vecFactory)
    {
        this.dm = dm;
        this.vecFactory = vecFactory;
    }

    public DBSCAN()
    {
        this(new EuclideanDistance() ,new KDTree.KDTreeFactory<VecPaired<Integer, Vec>>());
    }
    
    public List<List<DataPoint>> cluster(DataSet dataSet, int minPts)
    {
        OnLineStatistics stats = new OnLineStatistics();
        VectorCollection<VecPaired<Integer, Vec>> vc = vecFactory.getVectorCollection(getVecIndexPairs(dataSet), dm);
        
        List<DataPoint> dps = dataSet.getDataPoints();
        for(DataPoint dp :  dps)
            stats.add(vc.search(dp.getNumericalValues(), minPts+1).get(minPts).getPair());
        
        
        
        double eps = stats.getMean() + stats.getStandardDeviation()*0.5;
        
        return cluster(dataSet, eps, minPts, vc);
    }

    public List<List<DataPoint>> cluster(DataSet dataSet)
    {
        return cluster(dataSet, 3);
    }

    public List<List<DataPoint>> cluster(DataSet dataSet, ExecutorService threadpool)
    {
        return cluster(dataSet, 3, threadpool);
    }
    
    private class StatsWorker implements Callable<OnLineStatistics>
    {
        final BlockingQueue<DataPoint> queue;
        final VectorCollection<VecPaired<Integer, Vec>> vc;
        final int minPts;

        public StatsWorker(BlockingQueue<DataPoint> queue, VectorCollection<VecPaired<Integer, Vec>> vc, int minPts)
        {
            this.queue = queue;
            this.vc = vc;
            this.minPts = minPts;
        }

        public OnLineStatistics call() throws Exception
        {
            OnLineStatistics stats = new OnLineStatistics();
            while(true)
            {
                DataPoint dp = queue.take();
                
                if(dp.numNumericalValues() == 0 && dp.numCategoricalValues() == 0)
                    break;//Posion value, nonsense data point used to signal end
                stats.add(vc.search(dp.getNumericalValues(), minPts+1).get(minPts).getPair());
            }
            return stats;
        }
        
    }
    
    public List<List<DataPoint>> cluster(DataSet dataSet, int minPts, ExecutorService threadpool)
    {
        OnLineStatistics stats = null;
        VectorCollection<VecPaired<Integer, Vec>> vc = vecFactory.getVectorCollection(getVecIndexPairs(dataSet), dm);
        
        BlockingQueue<DataPoint> queue = new ArrayBlockingQueue<DataPoint>(SystemInfo.L2CacheSize*2);
        List<Future<OnLineStatistics>> futures = new ArrayList<Future<OnLineStatistics>>(SystemInfo.LogicalCores);
        
        //Setup
        for(int i = 0; i < SystemInfo.LogicalCores; i++)
            futures.add(threadpool.submit(new StatsWorker(queue, vc, minPts)));
        //Feed data
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            queue.add(dataSet.getDataPoint(i));
        //Posion stop
        for(int i = 0; i < SystemInfo.LogicalCores; i++)
            queue.add(new DataPoint(new DenseVector(0), new int[0], new CategoricalData[0]));
        
        for( Future<OnLineStatistics> future : futures)
        {
            try
            {
                if(stats == null)
                    stats = future.get();
                else 
                    stats = OnLineStatistics.add(stats, future.get());
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(DBSCAN.class.getName()).log(Level.SEVERE, null, ex);
            }
            catch (ExecutionException ex)
            {
                Logger.getLogger(DBSCAN.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        double eps = stats.getMean() + stats.getStandardDeviation()*0.5;
        
        return cluster(dataSet, eps, minPts, vc, threadpool);
    }

    private List<VecPaired<Integer, Vec>> getVecIndexPairs(DataSet dataSet)
    {
        List<VecPaired<Integer, Vec>> vecs = new ArrayList<VecPaired<Integer, Vec>>(dataSet.getSampleSize());
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            vecs.add(new VecPaired<Integer, Vec>(dataSet.getDataPoint(i).getNumericalValues(), i));
        return vecs;
    }
    
    public List<List<DataPoint>> cluster(DataSet dataSet, double eps, int minPts)
    {
        return cluster(dataSet, eps, minPts, vecFactory.getVectorCollection(getVecIndexPairs(dataSet), dm));
    }
    
    public List<List<DataPoint>> cluster(DataSet dataSet, double eps, int minPts, ExecutorService threadpool)
    {
        return cluster(dataSet, eps, minPts, vecFactory.getVectorCollection(getVecIndexPairs(dataSet), dm), threadpool);
    }
    
    private List<List<DataPoint>> cluster(DataSet dataSet, double eps, int minPts, VectorCollection<VecPaired<Integer, Vec>> vc)
    {
        int[] pointCats = new int[dataSet.getSampleSize()];
        Arrays.fill(pointCats, UNCLASSIFIED);
        
        int curClusterID = 0;
        for(int i = 0; i < pointCats.length; i++)
        {
            if(pointCats[i] == UNCLASSIFIED)
            {
                //All assignments are done by expandCluster
                if(exapndCluster(pointCats, dataSet, i, curClusterID, eps, minPts, vc))
                    curClusterID++;
            }
        }
        
        List<List<DataPoint>> ks = new ArrayList<List<DataPoint>>(curClusterID);
        for(int i = 0; i < curClusterID; i++)
            ks.add(new ArrayList<DataPoint>());
        
        for(int i = 0; i < pointCats.length; i++)
            if(pointCats[i] > UNCLASSIFIED)
                ks.get(pointCats[i]).add(dataSet.getDataPoint(i));
        
        return ks;
    }
    
    private List<List<DataPoint>> cluster(DataSet dataSet, double eps, int minPts, VectorCollection<VecPaired<Integer, Vec>> vc, ExecutorService threadpool)
    {
        int[] pointCats = new int[dataSet.getSampleSize()];
        Arrays.fill(pointCats, UNCLASSIFIED);
        
        int curClusterID = 0;
        for(int i = 0; i < pointCats.length; i++)
        {
            if(pointCats[i] == UNCLASSIFIED)
            {
                //All assignments are done by expandCluster
                if(exapndCluster(pointCats, dataSet, i, curClusterID, eps, minPts, vc, threadpool))
                    curClusterID++;
            }
        }
        
        List<List<DataPoint>> ks = new ArrayList<List<DataPoint>>(curClusterID);
        for(int i = 0; i < curClusterID; i++)
            ks.add(new ArrayList<DataPoint>());
        
        for(int i = 0; i < pointCats.length; i++)
            if(pointCats[i] > UNCLASSIFIED)
                ks.get(pointCats[i]).add(dataSet.getDataPoint(i));
        
        return ks;
    }
    
    private boolean exapndCluster(int[] pointCats, DataSet dataSet, int point, int clId, double eps, int minPts, VectorCollection<VecPaired<Integer, Vec>> vc)
    {
        Vec queryPoint = dataSet.getDataPoint(point).getNumericalValues();
        List<VecPaired<Double,VecPaired<Integer,Vec>>> seeds = vc.search(queryPoint, eps);
        
        if(seeds.size() < minPts)// no core point
        {
            pointCats[point] = NOISE;
            return false;
        }
        //Else, all points in seeds are density-reachable from Point
        
        List<VecPaired<Double,VecPaired<Integer,Vec>>> results;
        
        pointCats[point] = clId;
        Queue<VecPaired<Double,VecPaired<Integer,Vec>>> workQue = new ArrayDeque<VecPaired<Double, VecPaired<Integer, Vec>>>(seeds);
        while(!workQue.isEmpty())
        {
            VecPaired<Double,VecPaired<Integer,Vec>> currentP = workQue.poll();
            results = vc.search(currentP, eps);
            
            if(results.size() >= minPts)
                for(VecPaired<Double,VecPaired<Integer,Vec>> resultP :  results)
                {
                    int resultPIndx = resultP.getVector().getPair();
                    if(pointCats[resultPIndx] < 0)// is UNCLASSIFIED or NOISE
                    {
                        if(pointCats[resultPIndx] == UNCLASSIFIED)
                            workQue.add(resultP);
                        pointCats[resultPIndx] = clId;
                    }
                }   
        }
        
        return true;
    }
    
    private class ClusterWorker implements Runnable
    {
        private VectorCollection<VecPaired<Integer, Vec>> vc;
        private volatile List<VecPaired<Double,VecPaired<Integer,Vec>>> results;
        private volatile Vec searchPoint;
        private final double range;
        private final BlockingQueue<ClusterWorker> workQue;

        /**
         * 
         * @param vc the accelerate structure to search for data points. Must support concurent method calls
         * @param range the range to search <tt>tt</tt> with
         * @param workQue The que to place this worker object into on completion
         */
        public ClusterWorker(VectorCollection<VecPaired<Integer, Vec>> vc, double range, BlockingQueue<ClusterWorker> workQue)
        {
            this.vc = vc;
            this.results = results;
            this.searchPoint = searchPoint;
            this.range = range;
            this.workQue = workQue;
        }

        public List<VecPaired<Double, VecPaired<Integer, Vec>>> getResults()
        {
            return results;
        }

        public Vec getSearchPoint()
        {
            return searchPoint;
        }

        public void setSearchVec(Vec searchPoint)
        {
            this.searchPoint = searchPoint;
        }
        
        public void run()
        {
            results = vc.search(searchPoint, range);
            workQue.add(this);
        }
        
    }
    
    private boolean exapndCluster(int[] pointCats, DataSet dataSet, int point, int clId, double eps, int minPts, VectorCollection<VecPaired<Integer, Vec>> vc, ExecutorService threadpool)
    {
        Vec queryPoint = dataSet.getDataPoint(point).getNumericalValues();
        List<VecPaired<Double,VecPaired<Integer,Vec>>> seeds = vc.search(queryPoint, eps);
        
        if(seeds.size() < minPts)// no core point
        {
            pointCats[point] = NOISE;
            return false;
        }
        //Else, all points in seeds are density-reachable from Point
        
        List<VecPaired<Double,VecPaired<Integer,Vec>>> results;
        
        pointCats[point] = clId;
        Queue<VecPaired<Double,VecPaired<Integer,Vec>>> workQueue = new ArrayDeque<VecPaired<Double, VecPaired<Integer, Vec>>>(seeds);
        
        BlockingQueue<ClusterWorker> searchQue = new ArrayBlockingQueue<ClusterWorker>(SystemInfo.LogicalCores);
        Queue<ClusterWorker> freeWorkers = new ArrayDeque<ClusterWorker>(SystemInfo.LogicalCores);
        for(int i = 0; i < SystemInfo.LogicalCores; i++)
            freeWorkers.add(new ClusterWorker(vc, eps, searchQue));
        int received = 0, given = 0;
        
        do
        {
            while (!workQueue.isEmpty())
            {
                VecPaired<Double, VecPaired<Integer, Vec>> currentP = workQueue.poll();
                while(!freeWorkers.isEmpty())
                {
                    given++;
                    ClusterWorker worker = freeWorkers.poll();
                    worker.setSearchVec(currentP);
                    threadpool.submit(worker);
                }
            }


            try
            {
                ClusterWorker worker = searchQue.take();
                received++;
                results = worker.getResults();
                freeWorkers.add(worker);

                if (results.size() >= minPts)
                    for (VecPaired<Double, VecPaired<Integer, Vec>> resultP : results)
                    {
                        int resultPIndx = resultP.getVector().getPair();
                        if (pointCats[resultPIndx] < 0)// is UNCLASSIFIED or NOISE
                        {
                            if (pointCats[resultPIndx] == UNCLASSIFIED)
                                workQueue.add(resultP);
                            pointCats[resultPIndx] = clId;
                        }
                    }
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(DBSCAN.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        while(received < given);
        
        return true;
    }

}
