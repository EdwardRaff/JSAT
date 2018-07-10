
package jsat.clustering;

import jsat.linear.distancemetrics.TrainableDistanceMetric;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;

import jsat.DataSet;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.math.OnLineStatistics;
import static jsat.clustering.SeedSelectionMethods.*;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

/**
 *
 * @author Edward Raff
 */
public class PAM implements KClusterer
{

    private static final long serialVersionUID = 4787649180692115514L;
    protected DistanceMetric dm;
    protected Random rand;
    protected SeedSelection seedSelection;
    protected int iterLimit = 100;
    
    protected int[] medoids;
    protected boolean storeMedoids = true;

    public PAM(DistanceMetric dm, Random rand, SeedSelection seedSelection)
    {
        this.dm = dm;
        this.rand = rand;
        this.seedSelection = seedSelection;
    }

    public PAM(DistanceMetric dm, Random rand)
    {
        this(dm, rand, SeedSelection.KPP);
    }

    public PAM(DistanceMetric dm)
    {
        this(dm, RandomUtil.getRandom());
    }

    public PAM()
    {
        this(new EuclideanDistance());
    }

    /**
     * 
     * @param iterLimit the maximum number of iterations of the algorithm to perform
     */
    public void setMaxIterations(int iterLimit)
    {
        this.iterLimit = iterLimit;
    }

    /**
     * 
     * @return the maximum number of iterations of the algorithm to perform
     */
    public int getMaxIterations()
    {
        return iterLimit;
    }

    /**
     * Sets the distance metric used by this clustering algorithm
     * @param dm the distance metric to use
     */
    public void setDistanceMetric(DistanceMetric dm)
    {
        this.dm = dm;
    }

    /**
     * 
     * @return the distance metric to be used by this algorithm
     */
    public DistanceMetric getDistanceMetric()
    {
        return dm;
    }
    
    
    

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public PAM(PAM toCopy)
    {
        this.dm = toCopy.dm.clone();
        this.rand = RandomUtil.getRandom();
        this.seedSelection = toCopy.seedSelection;
        if(toCopy.medoids != null)
            this.medoids = Arrays.copyOf(toCopy.medoids, toCopy.medoids.length);
        this.storeMedoids = toCopy.storeMedoids;
        this.iterLimit = toCopy.iterLimit;
    }
    
    /**
     * If set to {@code true} the computed medoids will be stored after clustering
     * is completed, and can then be retrieved using {@link #getMedoids() }. 
     * @param storeMedoids {@code true} if the medoids should be stored for 
     * later, {@code false} to discard them once clustering is complete. 
     */
    public void setStoreMedoids(boolean storeMedoids)
    {
        this.storeMedoids = storeMedoids;
    }

    /**
     * Returns the raw array of indices that indicate which data point acted as 
     * the center for each cluster. 
     * @return the array of medeoid indices
     */
    public int[] getMedoids()
    {
        return medoids;
    }

    /**
     * Sets the method of seed selection used by this algorithm
     * @param seedSelection the method of seed selection to used
     */
    public void setSeedSelection(SeedSelection seedSelection)
    {
        this.seedSelection = seedSelection;
    }

    /**
     * 
     * @return the method of seed selection used by this algorithm
     */
    public SeedSelection getSeedSelection()
    {
        return seedSelection;
    }
   
    
    /**
     * Performs the actual work of PAM. 
     * 
     * @param data the data set to apply PAM to
     * @param doInit {@code true} if the initialization procedure of training the distance metric, initiating its cache, and selecting he seeds, should be done. 
     * @param medioids the array to store the indices that get chosen as the medoids. The length of the array indicates how many medoids should be obtained. 
     * @param assignments an array of the same length as <tt>data</tt>, each value indicating what cluster that point belongs to. 
     * @param cacheAccel the pre-computed distance acceleration cache. May be {@code null}.
     * @param parallel the value of parallel 
     * @return the double 
     */
    protected double cluster(DataSet data, boolean doInit, int[] medioids, int[] assignments, List<Double> cacheAccel, boolean parallel)
    {
        DoubleAdder totalDistance =new DoubleAdder();
        LongAdder changes = new LongAdder();
        Arrays.fill(assignments, -1);//-1, invalid category!
        
        int[] bestMedCand = new int[medioids.length];
        double[] bestMedCandDist = new double[medioids.length];
        List<Vec> X = data.getDataVectors();
        final List<Double> accel;
        
        if(doInit)
        {
            TrainableDistanceMetric.trainIfNeeded(dm, data);
            accel = dm.getAccelerationCache(X);
            selectIntialPoints(data, medioids, dm, accel, rand, seedSelection);
        }
        else
            accel = cacheAccel;

        int iter = 0;
        do
        {
            changes.reset();
            totalDistance.reset();
            
            ParallelUtils.run(parallel, data.size(), (start, end)->
            {
                for(int i = start; i < end; i++)
                {
                    int assignment = 0;
                    double minDist = dm.dist(medioids[0], i, X, accel);

                    for (int k = 1; k < medioids.length; k++)
                    {
                        double dist = dm.dist(medioids[k], i, X, accel);
                        if (dist < minDist)
                        {
                            minDist = dist;
                            assignment = k;
                        }
                    }

                    //Update which cluster it is in
                    if (assignments[i] != assignment)
                    {
                        changes.increment();
                        assignments[i] = assignment;
                    }
                    totalDistance.add(minDist * minDist);
                }
            });
            
            //TODO this update may be faster by using more memory, and actually moiving people about in the assignment loop above
            //Update the medoids
            Arrays.fill(bestMedCandDist, Double.MAX_VALUE);
            for(int i = 0; i < data.size(); i++)
            {
                double thisCandidateDistance;
                final int clusterID = assignments[i];
                final int medCandadate = i;
                
                final int ii = i;
                thisCandidateDistance = ParallelUtils.range(data.size(), parallel)
                        .filter(j -> j != ii && assignments[j] == clusterID)
                        .mapToDouble(j -> Math.pow(dm.dist(medCandadate, j, X, accel), 2))
                        .sum();
                
                if(thisCandidateDistance < bestMedCandDist[clusterID])
                {
                    bestMedCand[clusterID] = i;
                    bestMedCandDist[clusterID] = thisCandidateDistance;
                }
            }
            System.arraycopy(bestMedCand, 0, medioids, 0, medioids.length);
        }
        while( changes.sum() > 0 && iter++ < iterLimit);
        
        return totalDistance.sum();
    }

    @Override
    public int[] cluster(DataSet dataSet, boolean parallel, int[] designations)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.size()/2), parallel, designations);
    }
    
    @Override
    public int[] cluster(DataSet dataSet, int clusters, boolean parallel, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.size()];
        medoids = new int[clusters];
        
        cluster(dataSet, true, medoids, designations, null, parallel);
        
        if(!storeMedoids)
            medoids = null;
        
        return designations;
    }

    @Override
    public PAM clone()
    {
        return new PAM(this);
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, boolean parallel, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.size()];
        
        double[] totDistances = new double[highK-lowK+1];
        
        for(int k = lowK; k <= highK; k++)
        {
            totDistances[k-lowK] = cluster(dataSet, true, new int[k], designations, null, parallel);
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
        
        return cluster(dataSet, maxChangeK, parallel, designations);
    }
    
    /**
     * Computes the medoid of the data 
     * @param parallel whether or not the computation should be done using multiple cores
     * @param X the list of all data
     * @param dm the distance metric to get the medoid with respect to
     * @return the index of the point in <tt>X</tt> that is the medoid
     */
    public static int medoid(boolean parallel, List<? extends Vec> X, DistanceMetric dm)
    {
        IntList order = new IntList(X.size());
        ListUtils.addRange(order, 0, X.size(), 1);
        List<Double> accel = dm.getAccelerationCache(X, parallel);
        return medoid(parallel, order, X, dm, accel);
    }
    
    /**
     * Computes the medoid of a sub-set of data
     * @param parallel whether or not the computation should be done using multiple cores
     * @param indecies the indexes of the points to get the medoid of 
     * @param X the list of all data
     * @param dm the distance metric to get the medoid with respect to
     * @param accel the acceleration cache for the distance metric
     * @return the index value contained within indecies that is the medoid 
     */
    public static int medoid(boolean parallel, Collection<Integer> indecies, List<? extends Vec> X, DistanceMetric dm, List<Double> accel)
    {
        double bestDist = Double.POSITIVE_INFINITY;
        int bestIndex = -1;
        for (int i : indecies)
        {
            double thisCandidateDistance;
            final int medCandadate = i;

            thisCandidateDistance = ParallelUtils.range(indecies.size(), parallel)
                    .filter(j -> j != i)
                    .mapToDouble(j -> dm.dist(medCandadate, j, X, accel))
                    .sum();

            if (thisCandidateDistance < bestDist)
            {
                bestIndex = i;
                bestDist = thisCandidateDistance;
            }
        }
        
        return bestIndex;
    }
}
