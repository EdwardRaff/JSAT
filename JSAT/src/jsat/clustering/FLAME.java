package jsat.clustering;

import java.util.*;
import java.util.concurrent.*;

import jsat.DataSet;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.linear.vectorcollection.DefaultVectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollectionUtils;
import jsat.math.OnLineStatistics;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.FakeExecutor;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.AtomicDoubleArray;

/**
 * Provides an implementation of the FLAME clustering algorithm. The original 
 * FLAME paper does not describe all necessary details for an implementation, so 
 * results may differ between implementations. <br><br>
 * FLAME is highly sensitive to the number of neighbors chosen. Increasing the 
 * neighbors tends to reduce the number of clusters formed.
 * <br><br>
 * See: Fu, L.,&amp;Medico, E. (2007). <i>FLAME, a novel fuzzy clustering method 
 * for the analysis of DNA microarray data</i>. BMC Bioinformatics, 8(1), 3. 
 * Retrieved from <a href="http://www.ncbi.nlm.nih.gov/pubmed/17204155">here</a>
 * 
 * @author Edward Raff
 */
public class FLAME extends ClustererBase implements Parameterized
{

	private static final long serialVersionUID = 2393091020100706517L;
	private DistanceMetric dm;
    private int k;
    private int maxIterations;
    private VectorCollectionFactory<VecPaired<Vec, Integer>> vectorCollectionFactory = new DefaultVectorCollectionFactory<VecPaired<Vec, Integer>>();
    private double stndDevs = 2.5;
    private double eps = 1e-6;

    /**
     * Creates a new FLAME clustering object
     * @param dm the distance metric to use
     * @param k the number of neighbors to consider
     * @param maxIterations the maximum number of iterations to perform
     */
    public FLAME(DistanceMetric dm, int k, int maxIterations)
    {
        setDistanceMetric(dm);
        setK(k);
        setMaxIterations(maxIterations);
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public FLAME(FLAME toCopy)
    {
        this.dm = toCopy.dm.clone();
        this.maxIterations = toCopy.maxIterations;
        this.vectorCollectionFactory = toCopy.vectorCollectionFactory;
        this.k = toCopy.k;
        this.stndDevs = toCopy.stndDevs;
        this.eps = toCopy.eps;
        
    }
  
    /**
     * Sets the maximum number of iterations to perform. FLAME can require far 
     * more iterations to converge than necessary to get the same hard 
     * clustering result. 
     * 
     * @param maxIterations the maximum number of iterations to perform
     */
    public void setMaxIterations(int maxIterations)
    {
        if(maxIterations < 1)
            throw new IllegalArgumentException("Must perform a positive number of iterations, not " + maxIterations);
        this.maxIterations = maxIterations;
    }

    /**
     * Returns the maximum number of iterations to perform
     * @return the maximum number of iterations to perform
     */
    public int getMaxIterations()
    {
        return maxIterations;
    }

    /**
     * Sets the number of neighbors that will be considered in determining 
     * Cluster Supporting Points and assignment contributions. 
     * @param k the number of neighbors to consider
     */
    public void setK(int k)
    {
        this.k = k;
    }

    /**
     * Returns the number of neighbors used
     * @return the number of neighbors used
     */
    public int getK()
    {
        return k;
    }

    /**
     * Sets the convergence goal for the minimum difference in score between 
     * rounds. Negative values are allowed to force all iterations to occur
     * @param eps the minimum difference in scores for convergence
     */
    public void setEps(double eps)
    {
        if(Double.isNaN(eps))
            throw new IllegalArgumentException("Eps can not be NaN");
        this.eps = eps;
    }

    /**
     * Returns the minimum difference in scores to consider FLAME converged
     * @return the minimum difference in scores to consider FLAMe converged
     */
    public double getEps()
    {
        return eps;
    }

    /**
     * Sets the number of standard deviations away from the mean density a
     * candidate outlier must be to be confirmed as an outlier. 
     * @param stndDevs the number of standard deviations away from the mean 
     * density an outlier must be
     */
    public void setStndDevs(double stndDevs)
    {
        if(stndDevs < 0 || Double.isInfinite(stndDevs) || Double.isNaN(stndDevs))
            throw new IllegalArgumentException("Standard Deviations must be non negative");
        this.stndDevs = stndDevs;
    }

    /**
     * Returns the number of standard deviations away from the mean 
     * density an outlier must be
     * @return the number of standard deviations away from the mean 
     * density an outlier must be
     */
    public double getStndDevs()
    {
        return stndDevs;
    }

    /**
     * Sets the distance metric to use for the nearest neighbor search
     * @param dm the distance metric to use
     */
    public void setDistanceMetric(DistanceMetric dm)
    {
        this.dm = dm;
    }

    /**
     * Returns the distance metric to use for the nearest neighbor search
     * @return the distance metric to use
     */
    public DistanceMetric getDistanceMetric()
    {
        return dm;
    }

    /**
     * Sets the vector collection factory used to accelerate the nearest 
     * neighbor search. The nearest neighbor only needs to be done once for each
     * point, so the collection should be faster than the naive method when 
     * considering both construction and search time. 
     * 
     * @param vectorCollectionFactory 
     */
    public void setVectorCollectionFactory(VectorCollectionFactory<VecPaired<Vec, Integer>> vectorCollectionFactory)
    {
        this.vectorCollectionFactory = vectorCollectionFactory;
    }
    
    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        return cluster(dataSet, new FakeExecutor(), designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        try
        {
            final int n = dataSet.getSampleSize();
            if (designations == null || designations.length < dataSet.getSampleSize())
                designations = new int[n];
            List<VecPaired<Vec, Integer>> vecs = new ArrayList<VecPaired<Vec, Integer>>(n);
            for (int i = 0; i < dataSet.getSampleSize(); i++)
                vecs.add(new VecPaired<Vec, Integer>(dataSet.getDataPoint(i).getNumericalValues(), i));
            
            TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadpool);
            VectorCollection<VecPaired<Vec, Integer>> vc;
            final List<List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>> allNNs;
            if (threadpool instanceof FakeExecutor)
            {
                vc = vectorCollectionFactory.getVectorCollection(vecs, dm);
                allNNs = VectorCollectionUtils.allNearestNeighbors(vc, vecs, k + 1);
            }
            else
            {
                vc = vectorCollectionFactory.getVectorCollection(vecs, dm, threadpool);
                allNNs = VectorCollectionUtils.allNearestNeighbors(vc, vecs, k + 1, threadpool);
            }

            //NOTE: Density is done in reverse, so large values indicate low density, small values indiciate high density. 
            //mark density as the sum of distances
            final double[] density = new double[vecs.size()];
            final double[][] weights = new double[n][k];
            OnLineStatistics densityStats = new OnLineStatistics();
            for (int i = 0; i < density.length; i++)
            {
                List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> knns = allNNs.get(i);
                for (int j = 1; j < knns.size(); j++)
                    density[i] += (weights[i][j - 1] = knns.get(j).getPair());
                densityStats.add(density[i]);
                
                double sum = 0;
                for (int j = 0; j < k; j++)
                    sum += (weights[i][j] = Math.min(1.0 / Math.pow(weights[i][j], 2), Double.MAX_VALUE / (k + 1)));
                
                for (int j = 0; j < k; j++)
                    weights[i][j] /= sum;
            }
            
            final Map<Integer, Integer> CSOs = new HashMap<Integer, Integer>();
            final Set<Integer> outliers = new IntSet();
            Arrays.fill(designations, -1);
            
            final double threshold = densityStats.getMean() + densityStats.getStandardDeviation() * stndDevs;
            
            for (int i = 0; i < density.length; i++)
            {
                List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> knns = allNNs.get(i);
                boolean lowest = true;//if my density score is lower then all neighbors, then i am a CSO
                boolean highest = true;//if heigher, then I am an outlier
                for (int j = 1; j < knns.size() && (highest || lowest); j++)
                {
                    int jNN = knns.get(j).getVector().getPair();
                    if (density[i] > density[jNN])
                        lowest = false;
                    else
                        highest = false;
                }
                
                if (lowest)
                    CSOs.put(i, CSOs.size());
                else if (highest && density[i] > threshold)
                    outliers.add(i);
            }
            
            //remove CSO that occur near outliers
            {
                int origSize = CSOs.size();
                Iterator<Integer> iter = CSOs.keySet().iterator();
                while (iter.hasNext())
                {
                    int i = iter.next();
                    List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> knns = allNNs.get(i);
                    for (int j = 1; j < knns.size(); j++)
                        if (outliers.contains(knns.get(j).getVector().getPair()))
                        {
                            iter.remove();
                            break;
                        }
                }
                
                if(origSize != CSOs.size())//we did a removal, re-order clusters
                {
                    Set<Integer> keys = new IntSet(CSOs.keySet());
                    CSOs.clear();
                    for(int i : keys)
                        CSOs.put(i, CSOs.size());
                }
                //May have gaps, will be fixed in final step
                for (int i : CSOs.keySet())
                    designations[i] = CSOs.get(i);
            }

            //outlier is implicit extra term
            double[][] fuzzy = new double[n][CSOs.size() + 1];
            for (int i = 0; i < n; i++)
                if (CSOs.containsKey(i))
                    fuzzy[i][CSOs.get(i)] = 1.0;//each CSO is full it itself
                else if (outliers.contains(i))
                    fuzzy[i][CSOs.size()] = 1.0;
                else
                    Arrays.fill(fuzzy[i], 1.0 / (CSOs.size() + 1));



            //iterate
            double[][] fuzzy2 = new double[n][CSOs.size() + 1];

            double prevScore = Double.POSITIVE_INFINITY;
            
            for (int iter = 0; iter < maxIterations; iter++)
            {
                final double[][] FROM = fuzzy, TO = fuzzy2;
                final AtomicDoubleArray score = new AtomicDoubleArray(1);
                final CountDownLatch cdl = new CountDownLatch(SystemInfo.LogicalCores);
                for (int id = 0; id < SystemInfo.LogicalCores; id++)
                {
                    final int ID = id;
                    threadpool.submit(new Runnable() 
                    {

                        @Override
                        public void run()
                        {
                            double localScore = 0;
                            for (int i = ID; i < FROM.length; i+=SystemInfo.LogicalCores)
                            {
                                if (outliers.contains(i) || CSOs.containsKey(i))
                                    continue;
                                final double[] fuzzy2_i = TO[i];
                                Arrays.fill(fuzzy2_i, 0);
                                List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> knns = allNNs.get(i);

                                double sum = 0;
                                for (int j = 1; j < weights[i].length; j++)
                                {
                                    int jNN = knns.get(j).getVector().getPair();
                                    final double[] fuzzy_jNN = FROM[jNN];
                                    double weight = weights[i][j - 1];
                                    for (int z = 0; z < FROM[jNN].length; z++)
                                        fuzzy2_i[z] += weight * fuzzy_jNN[z];
                                }

                                for (int z = 0; z < fuzzy2_i.length; z++)
                                    sum += fuzzy2_i[z];

                                for (int z = 0; z < fuzzy2_i.length; z++)
                                {
                                    fuzzy2_i[z] /= sum+1e-6;
                                    localScore += Math.abs(FROM[i][z] - fuzzy2_i[z]);
                                }
                            }
                            score.addAndGet(0, localScore);
                            cdl.countDown();
                        }
                    });
                }
             
                cdl.await();

                if (Math.abs(prevScore - score.get(0)) < eps)
                    break;
                prevScore = score.get(0);

                double[][] tmp = fuzzy;
                fuzzy = fuzzy2;
                fuzzy2 = tmp;
            }

            //Figure out final clsutering
            int[] clusterCounts = new int[n];
            for (int i = 0; i < fuzzy.length; i++)
            {
                int pos = -1;
                double maxVal = 0;
                for (int j = 0; j < fuzzy[i].length; j++)
                {
                    if (fuzzy[i][j] > maxVal)
                    {
                        maxVal = fuzzy[i][j];
                        pos = j;
                    }
                }

                if(pos == -1)//TODO how di this happen? Mark it as an outlier. Somehow your whole row became zeros to cause this
                    pos = CSOs.size();
                clusterCounts[pos]++;
                if (pos == CSOs.size())//outlier
                    pos = -1;
                designations[i] = pos;
            }
            
            /* Transform clusterCOunts to indicate the new cluster ID. If 
             * everyone gets there own id, no clusters removed. Else, people 
             * with a negative value know they need to remove themsleves 
             */
            int newCCount = 0;
            for(int i = 0; i < clusterCounts.length; i++)
                if(clusterCounts[i] > 1)
                    clusterCounts[i] = newCCount++;
                else
                    clusterCounts[i] = -1;
                    
            
            //Go through and remove clusters with a count of 1
            if(newCCount != clusterCounts.length)
            {
                double[] tmp = new double[CSOs.size()+1];
                for (int i = 0; i < fuzzy.length; i++)
                {
                    int d = designations[i];
                    if(d > 0)//not outlier
                    {
                        if (clusterCounts[d] == -1)//remove self
                        {
                            List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> knns = allNNs.get(i);
                            
                            for (int j = 1; j < weights[i].length; j++)
                            {
                                int jNN = knns.get(j).getVector().getPair();
                                final double[] fuzzy_jNN = fuzzy[jNN];
                                double weight = weights[i][j - 1];
                                for (int z = 0; z < fuzzy[jNN].length; z++)
                                    tmp[z] += weight * fuzzy_jNN[z];
                            }
                            
                            double maxVal = -1;
                            int maxIndx = -1;
                            for(int z = 0; z < tmp.length; z++)
                                if(tmp[z] > maxVal)
                                {
                                    maxVal =tmp[z];
                                    maxIndx = z;
                                }
                            if(maxIndx == CSOs.size())
                                designations[i] = -1;
                            else
                                designations[i] = clusterCounts[maxIndx];
                        }
                        else
                        {
                            designations[i] = clusterCounts[d];
                        }
                    }
                }
            }
            
            return designations;
        }
        catch (InterruptedException interruptedException)
        {
            throw new ClusterFailureException();
        }
    }

    @Override
    public FLAME clone()
    {
        return new FLAME(this);
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
