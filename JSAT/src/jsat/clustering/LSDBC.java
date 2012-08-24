package jsat.clustering;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.*;
import jsat.linear.vectorcollection.*;
import jsat.parameters.*;
import jsat.text.GreekLetters;
import jsat.utils.*;

/**
 * A parallel implementation of <i>Locally Scaled Density Based Clustering</i>. 
 * <br><br>
 * See paper:<br>
 * <a href="http://www.springerlink.com/index/0116171485446868.pdf">Biçici, E., 
 * & Yuret, D. (2007). Locally scaled density based clustering. 
 * In B. Beliczynski, A. Dzielinski, M. Iwanowski, & B. Ribeiro (Eds.), 
 * Adaptive and Natural Computing Algorithms (pp. 739–748). 
 * Warsaw, Poland: Springer-Verlag. </a>
 * @author Edward Raff
 */
public class LSDBC extends ClustererBase implements Parameterized
{
    /**
     * {@value #DEFAULT_NEIGHBORS} is the default number of neighbors used when performing clustering
     * @see #setNeighbors(int) 
     */
    public static final int DEFAULT_NEIGHBORS = 15;
    /**
     * {@value #DEFAULT_ALPHA} is the default scale value used when performing clustering. 
     * @see #setAlpha(double) 
     */
    public static final double DEFAULT_ALPHA = 4;
    private static final int UNCLASSIFIED = -1;
    private DistanceMetric dm;
    
    private VectorCollectionFactory<VecPaired<Integer, Vec>> vectorCollectionFactory = new DefaultVectorCollectionFactory<VecPaired<Integer, Vec>>();
    
    /**
     * The weight parameter for forming new clusters
     */
    private double alpha;
    /**
     * The number of neighbors to use
     */
    private int k;
    
    private List<Parameter> parameters = Collections.unmodifiableList(new ArrayList<Parameter>(3)
    {{
            add(new DoubleParameter() {

                @Override
                public double getValue()
                {
                    return getAlpha();
                }

                @Override
                public boolean setValue(double val)
                {
                    try
                    {
                        setAlpha(val);
                        return true;
                    }
                    catch(ArithmeticException ex)
                    {
                        return false;
                    }
                }

                @Override
                public String getASCIIName()
                {
                    return "Alpha";
                }

                @Override
                public String getName()
                {
                    return GreekLetters.alpha;
                } 
            });
            
            add(new IntParameter() {

                @Override
                public int getValue()
                {
                    return getNeighbors();
                }

                @Override
                public boolean setValue(int val)
                {
                    if(val <= 0)
                        return false;
                    setNeighbors(val);
                    return true;
                }

                @Override
                public String getASCIIName()
                {
                    return "Neighbors";
                }
            });
            
            add(new MetricParameter() {

                @Override
                public boolean setMetric(DistanceMetric val)
                {
                    setDistanceMetric(val);
                    return true;
                }

                @Override
                public DistanceMetric getMetric()
                {
                    return getDistanceMetric();
                }
            });
    }});
    
    private Map<String, Parameter> parameterMap = Parameter.toParameterMap(parameters);

    /**
     * Creates a new LSDBC clustering object using the given distance metric
     * @param dm the distance metric to use 
     * @param alpha the scale factor to use when forming clusters
     * @param neighbors the number of neighbors to consider when determining clusters
     */
    public LSDBC(DistanceMetric dm, double alpha, int neighbors)
    {
        setDistanceMetric(dm);
        setAlpha(alpha);
        setNeighbors(neighbors);
    }

    /**
     * Creates a new LSDBC clustering object using the given distance metric
     * @param dm the distance metric to use 
     * @param alpha the scale factor to use when forming clusters
     */
    public LSDBC(DistanceMetric dm, double alpha)
    {
        this(dm, alpha, DEFAULT_NEIGHBORS);
    }
    
    /**
     * Creates a new LSDBC clustering object using the given distance metric
     * @param dm the distance metric to use 
     */
    public LSDBC(DistanceMetric dm)
    {
        this(dm, DEFAULT_ALPHA);
    }

    /**
     * Creates a new LSDBC clustering object using the {@link EuclideanDistance}
     * and default parameter values. 
     */
    public LSDBC()
    {
        this(new EuclideanDistance());
    }

    /**
     * Sets the vector collection factory used for acceleration of neighbor searches. 
     * @param vectorCollectionFactory the vector collection factory to use
     */
    public void setVectorCollectionFactory(VectorCollectionFactory<VecPaired<Integer, Vec>> vectorCollectionFactory)
    {
        this.vectorCollectionFactory = vectorCollectionFactory;
    }
    
    
    /**
     * Sets the distance metric used when performing clustering. 
     * @param dm the distance metric to use. 
     */
    public void setDistanceMetric(DistanceMetric dm)
    {
        if(dm != null)
            this.dm = dm;
    }
    
    /**
     * Returns the distance metric used when performing clustering. 
     * @return the distance metric used
     */
    private DistanceMetric getDistanceMetric()
    {
        return dm;
    }

    /**
     * Sets the number of neighbors that will be considered when clustering 
     * data points
     * @param neighbors the number of neighbors the algorithm will use
     */
    public void setNeighbors(int neighbors)
    {
        if(neighbors <= 0)
            throw new ArithmeticException("Can not use a non positive number of neighbors");
        this.k = neighbors;
    }

    /**
     * Returns the number of neighbors that will be considered when clustering 
     * data points
     * @return the number of neighbors the algorithm will use
     */
    public int getNeighbors()
    {
        return k;
    }
    
    /**
     * Sets the scale value that will control how many points are added to a 
     * cluster. Smaller values will create more, smaller clusters - and more 
     * points will be labeled as noise. Larger values causes larger and fewer 
     * clusters.
     * 
     * @param alpha the scale value to use
     */
    public void setAlpha(double alpha)
    {
        if(alpha <= 0 || Double.isNaN(alpha) || Double.isInfinite(alpha))
            throw new ArithmeticException("Can not use the non positive scale value " + alpha );
        this.alpha = alpha;
    }

    /**
     * Returns the scale value that will control how many points are added to a 
     * cluster. Smaller values will create more, smaller clusters - and more 
     * points will be labeled as noise. Larger values causes larger and fewer 
     * clusters.
     * 
     * @return the scale value to use
     */
    public double getAlpha()
    {
        return alpha;
    }
    
    
    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        return cluster(dataSet, null, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
     
        //Compute all k-NN 
        final VectorCollection<VecPaired<Integer, Vec>> vc;
        List<List<VecPaired<Double, VecPaired<Integer, Vec>>>> knnVecList =
                new ArrayList<List<VecPaired<Double, VecPaired<Integer, Vec>>>>(dataSet.getSampleSize());

        try
        {
            //Set up
            List<VecPaired<Integer, Vec>> vecs = new ArrayList<VecPaired<Integer, Vec>>(dataSet.getSampleSize());

            for (int i = 0; i < dataSet.getSampleSize(); i++)
                vecs.add(new VecPaired<Integer, Vec>(dataSet.getDataPoint(i).getNumericalValues(), i));

            if (threadpool == null || threadpool instanceof FakeExecutor)
            {
                if(threadpool == null)
                    threadpool = new FakeExecutor();
                vc = vectorCollectionFactory.getVectorCollection(vecs, dm);
                TrainableDistanceMetric.trainIfNeeded(dm, dataSet);
            }
            else
            {
                vc = vectorCollectionFactory.getVectorCollection(vecs, dm, threadpool);
                TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadpool);
            }


            //Pre compute all kNN pairs
            List<Future<List<List<VecPaired<Double, VecPaired<Integer, Vec>>>>>> collectedResults =
                    new ArrayList<Future<List<List<VecPaired<Double, VecPaired<Integer, Vec>>>>>>(SystemInfo.LogicalCores);
            for (final List<VecPaired<Integer, Vec>> subList : ListUtils.splitList(vecs, SystemInfo.LogicalCores))
                collectedResults.add(threadpool.submit(new Callable<List<List<VecPaired<Double, VecPaired<Integer, Vec>>>>>()
                {

                    @Override
                    public List<List<VecPaired<Double, VecPaired<Integer, Vec>>>> call() throws Exception
                    {
                        List<List<VecPaired<Double, VecPaired<Integer, Vec>>>> subResult =
                                new ArrayList<List<VecPaired<Double, VecPaired<Integer, Vec>>>>(subList.size());

                        for (VecPaired<Integer, Vec> vec : subList)
                            subResult.add(vc.search(vec.getVector(), k + 1));

                        return subResult;

                    }
                }));

            //Collect parts, futers are in order, and results in sub lists are in order, so we get the original order back from iteration
            for (Future<List<List<VecPaired<Double, VecPaired<Integer, Vec>>>>> future : collectedResults)
                knnVecList.addAll(future.get());
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(LSDBC.class.getName()).log(Level.SEVERE, null, ex);
        }
        catch (ExecutionException ex)
        {
            Logger.getLogger(LSDBC.class.getName()).log(Level.SEVERE, null, ex);
        }


        //Sort
        IndexTable indexTable = new IndexTable(knnVecList, new Comparator()
        {

            @Override
            public int compare(Object o1, Object o2)
            {
                List<VecPaired<Double, VecPaired<Integer, Vec>>> l1 = 
                        (List<VecPaired<Double, VecPaired<Integer, Vec>>>) o1;
                List<VecPaired<Double, VecPaired<Integer, Vec>>> l2 = 
                        (List<VecPaired<Double, VecPaired<Integer, Vec>>>) o2;
                
                return Double.compare(getEps(l1), getEps(l2));
            }
        });
        
        //Assign clusters, does very little computation. No need to parallelize expandCluster
        Arrays.fill(designations, UNCLASSIFIED);
        int clusterID = 0;
        for(int i = 0; i < indexTable.length(); i++)
        {
            int p = indexTable.index(i);
            if(designations[p] == UNCLASSIFIED && localMax(p, knnVecList))
                expandCluster(p, clusterID++, knnVecList, designations);
        }
        
        return designations;
    }

    /**
     * Performs the main clustering loop of expandCluster. 
     * @param neighbors the list of neighbors
     * @param i the index of <tt>neighbors</tt> to consider
     * @param designations the array of cluster designations
     * @param clusterID the current clusterID to assign
     * @param seeds the stack to hold all seeds in
     */
    private void addSeed(List<VecPaired<Double, VecPaired<Integer, Vec>>> neighbors, int i, int[] designations, int clusterID, Stack<Integer> seeds)
    {
        int index = neighbors.get(i).getVector().getPair();
        if (designations[index] != UNCLASSIFIED)
            return;
        designations[index] = clusterID;
        seeds.add(index);
    }

    /**
     * Convenience method. Gets the eps value for the given set of neighbors
     * @param neighbors the set of neighbors, with index 0 being the point itself
     * @return the eps of the <tt>k</tt><sup>th</sup> neighbor
     */
    private double getEps(List<VecPaired<Double, VecPaired<Integer, Vec>>> neighbors)
    {
        return neighbors.get(k).getPair();
    }

    /**
     * Returns true if the given point is a local maxima, meaning it is more dense then the density of all its neighbors
     * @param p the index of the data point in question
     * @param knnVecList the neighbor list
     * @return <tt>true</tt> if it is a local max, <tt> false</tt> otherwise. 
     */
    private boolean localMax(int p, List<List<VecPaired<Double, VecPaired<Integer, Vec>>>> knnVecList)
    {
        List<VecPaired<Double, VecPaired<Integer, Vec>>> neighbors = knnVecList.get(p);
        double myEps = getEps(neighbors);
        
        for(int i = 1; i < neighbors.size(); i++)
        {
            int neighborP = neighbors.get(i).getVector().getPair();
            if(getEps(knnVecList.get(neighborP)) < myEps)
                return false;
        }
        
        return true;
    }

    /**
     * Does the cluster assignment
     * @param p the current index of a data point to assign to a cluster
     * @param clusterID the current cluster ID to assign
     * @param knnVecList the in order list of every index and its nearest neighbors
     * @param designations the array to store cluster designations in
     */
    private void expandCluster(int p, int clusterID, List<List<VecPaired<Double, VecPaired<Integer, Vec>>>> knnVecList, int[] designations)
    {
        designations[p] = clusterID;
        double pointEps;
        int n;
        Stack<Integer> seeds = new Stack<Integer>();
        {
            List<VecPaired<Double, VecPaired<Integer, Vec>>> neighbors = knnVecList.get(p);
            for (int i = 1; i < neighbors.size(); i++)
                addSeed(neighbors, i, designations, clusterID, seeds);
            pointEps = getEps(neighbors);
            n = neighbors.get(k).length();
        }
        final double scale = Math.pow(2, alpha / n);


        while (!seeds.isEmpty())
        {
            int currentP = seeds.pop();
            List<VecPaired<Double, VecPaired<Integer, Vec>>> neighbors = knnVecList.get(currentP);
            double currentPEps = getEps(neighbors);
            if (currentPEps <= scale * pointEps)
            {
                for (int i = 1; i < neighbors.size(); i++)
                    addSeed(neighbors, i, designations, clusterID, seeds);
            }
        }
    }

    @Override
    public List<Parameter> getParameters()
    {
        return parameters;
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return parameterMap.get(paramName);
    }

}
