
package jsat.clustering;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.linear.MatrixStatistics;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.*;
import static jsat.utils.SystemInfo.LogicalCores;
import jsat.utils.concurrent.ParallelUtils;

/**
 * This class provides methods for sampling a data set for a set of initial points to act as the seeds for a clustering algorithm. 
 * 
 * @author Edward Raff
 */
public class SeedSelectionMethods
{

    private SeedSelectionMethods()
    {
    }
    
    
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
        KPP,
        
        /**
         * The first seed is chosen randomly, and then all others are chosen
         * to be the farthest away from all other seeds
         */
        FARTHEST_FIRST,
        
        /**
         * Selects the seeds in one pass by selecting points as evenly 
         * distributed quantiles for the distance of each point from the mean 
         * of the whole data set. This makes the seed selection deterministic
         * <br><br>
         * See: J. A. Hartigan and M. A. Wong, "A k-means clustering algorithm", 
         * Applied Statistics, vol. 28, pp. 100–108, 1979.
         */
        MEAN_QUANTILES
    };
    
    /**
     * Selects seeds from a data set to use for a clustering algorithm. Copies of the vectors chosen will be returned.
     * 
     * @param d the data set to perform select from
     * @param k the number of seeds to choose 
     * @param dm the distance metric to used when selecting points
     * @param rand a source of randomness
     * @param selectionMethod  The method of seed selection to use. 
     * @return a list of the copies of the chosen vectors. 
     */
    static public List<Vec> selectIntialPoints(DataSet d, int k, DistanceMetric dm, Random rand, SeedSelection selectionMethod)
    {
        return selectIntialPoints(d, k, dm, null, rand, selectionMethod);
    }
    
    /**
     * 
     * @param d the data set to perform select from
     * @param k the number of seeds to choose 
     * @param dm the distance metric to used when selecting points
     * @param accelCache the cache of pre-generated acceleration information for the distance metric. May be null
     * @param rand a source of randomness
     * @param selectionMethod  The method of seed selection to use. 
     * @return a list of the copies of the chosen vectors. 
     */
    static public List<Vec> selectIntialPoints(DataSet d, int k, DistanceMetric dm, List<Double> accelCache, Random rand, SeedSelection selectionMethod)
    {
        int[] indicies = new int[k];
        selectIntialPoints(d, indicies, dm, accelCache, rand, selectionMethod, false);
        List<Vec> vecs = new ArrayList<>(k);
        for(Integer i : indicies)
            vecs.add(d.getDataPoint(i).getNumericalValues().clone());
        return vecs;
    }
    
    /**
     * Selects seeds from a data set to use for a clustering algorithm. Copies of the vectors chosen will be returned.
     * 
     * @param d the data set to perform select from
     * @param k the number of seeds to choose 
     * @param dm the distance metric to used when selecting points
     * @param rand a source of randomness
     * @param selectionMethod  The method of seed selection to use. 
     * @param parallel {@code true} if multiple threads should be used to
     * perform clustering. {@code false} if it should be done in a single
     * threaded manner.
     * @return a list of the copies of the chosen vectors. 
     */
    static public List<Vec> selectIntialPoints(DataSet d, int k, DistanceMetric dm, Random rand, SeedSelection selectionMethod, boolean parallel)
    {
        return selectIntialPoints(d, k, dm, null, rand, selectionMethod, parallel);
    }
    
    /**
     * Selects seeds from a data set to use for a clustering algorithm. Copies of the vectors chosen will be returned.
     * 
     * @param d the data set to perform select from
     * @param k the number of seeds to choose 
     * @param dm the distance metric to used when selecting points
     * @param accelCache the cache of pre-generated acceleration information for the distance metric. May be null
     * @param rand a source of randomness
     * @param selectionMethod  The method of seed selection to use. 
     * @param parallel {@code true} if multiple threads should be used to
     * perform clustering. {@code false} if it should be done in a single
     * threaded manner.
     * @return a list of the copies of the chosen vectors. 
     */
    static public List<Vec> selectIntialPoints(DataSet d, int k, DistanceMetric dm, List<Double> accelCache, Random rand, SeedSelection selectionMethod, boolean parallel)
    {
        int[] indicies = new int[k];
        selectIntialPoints(d, indicies, dm, accelCache, rand, selectionMethod, parallel);
        List<Vec> vecs = new ArrayList<Vec>(k);
        for(Integer i : indicies)
            vecs.add(d.getDataPoint(i).getNumericalValues().clone());
        return vecs;
    }
    
    /**
     * Selects seeds from a data set to use for a clustering algorithm. The indices of the chosen points will be placed in the <tt>indices</tt> array. 
     * 
     * @param d the data set to perform select from
     * @param indices a storage place to note the indices that were chosen as seed. The length of the array indicates how many seeds to select. 
     * @param dm the distance metric to used when selecting points
     * @param rand a source of randomness
     * @param selectionMethod  The method of seed selection to use. 
     */
    static public void selectIntialPoints(DataSet d, int[] indices, DistanceMetric dm, Random rand, SeedSelection selectionMethod)
    {
        selectIntialPoints(d, indices, dm, null, rand, selectionMethod);
    }
    
    /**
     * Selects seeds from a data set to use for a clustering algorithm. The indices of the chosen points will be placed in the <tt>indices</tt> array. 
     * 
     * @param d the data set to perform select from
     * @param indices a storage place to note the indices that were chosen as seed. The length of the array indicates how many seeds to select. 
     * @param dm the distance metric to used when selecting points
     * @param accelCache the cache of pre-generated acceleration information for the distance metric. May be null
     * @param rand a source of randomness
     * @param selectionMethod  The method of seed selection to use. 
     */
    static public void selectIntialPoints(DataSet d, int[] indices, DistanceMetric dm, List<Double> accelCache, Random rand, SeedSelection selectionMethod)
    {
        selectIntialPoints(d, indices, dm, accelCache, rand, selectionMethod, false);
    }
    
    /**
     * Selects seeds from a data set to use for a clustering algorithm. The indices of the chosen points will be placed in the <tt>indices</tt> array. 
     * 
     * @param d the data set to perform select from
     * @param indices a storage place to note the indices that were chosen as seed. The length of the array indicates how many seeds to select. 
     * @param dm the distance metric to used when selecting points
     * @param rand a source of randomness
     * @param selectionMethod  The method of seed selection to use. 
     * @param parallel {@code true} if multiple threads should be used to
     * perform clustering. {@code false} if it should be done in a single
     * threaded manner.
     */
    static public void selectIntialPoints(DataSet d, int[] indices, DistanceMetric dm, Random rand, SeedSelection selectionMethod, boolean parallel)
    {
        selectIntialPoints(d, indices, dm, null, rand, selectionMethod, parallel);
    }
    
    /**
     * Selects seeds from a data set to use for a clustering algorithm. The indices of the chosen points will be placed in the <tt>indices</tt> array. 
     *
     * @param d the data set to perform select from
     * @param indices a storage place to note the indices that were chosen as seed. The length of the array indicates how many seeds to select. 
     * @param dm the distance metric to used when selecting points
     * @param accelCache the cache of pre-generated acceleration information for the distance metric. May be null
     * @param rand a source of randomness
     * @param selectionMethod  The method of seed selection to use. 
     * @param parallel {@code true} if multiple threads should be used to
     * perform clustering. {@code false} if it should be done in a single
     * threaded manner.
     */
    static public void selectIntialPoints(DataSet d, int[] indices, DistanceMetric dm, List<Double> accelCache, Random rand, SeedSelection selectionMethod, boolean parallel)
    {

        int k = indices.length;

        if (null != selectionMethod)
            switch (selectionMethod)
            {
                case RANDOM:
                    Set<Integer> indecies = new IntSet(k);
                    while (indecies.size() != k)//Keep sampling, we cant use the same point twice.
                        indecies.add(rand.nextInt(d.size()));//TODO create method to do uniform sampleling for a select range
                    int j = 0;
                    for (Integer i : indecies)
                        indices[j++] = i;
                    break;
                case KPP:
                    kppSelection(indices, rand, d, k, dm, accelCache, parallel);
                    break;
                case FARTHEST_FIRST:
                    ffSelection(indices, rand, d, k, dm, accelCache, parallel);
                    break;
                case MEAN_QUANTILES:
                    mqSelection(indices, d, k, dm, accelCache, parallel);
                    break;
                default:
                    break;
            }
        
    }

    private static void kppSelection(int[] indices, Random rand, DataSet d, int k, DistanceMetric dm, List<Double> accelCache)
    {
        kppSelection(indices, rand, d, k, dm, accelCache, false);
    }
    
    private static void kppSelection(final int[] indices, Random rand, final DataSet d, final int k, final DistanceMetric dm, final List<Double> accelCache, boolean parallel)
    {
        /*
         * http://www.stanford.edu/~darthur/kMeansPlusPlus.pdf : k-means++: The Advantages of Careful Seeding
         *
         */
        //Initial random point
        indices[0] = rand.nextInt(d.size());

        final double[] closestDist = new double[d.size()];
        final List<Vec> X = d.getDataVectors();

        for (int j = 1; j < k; j++)
        {
            //Compute the distance from each data point to the closest mean
            final int newMeanIndx = indices[j - 1];//Only the most recently added mean needs to get distances computed. 

            final boolean forceCompute = j == 1;
            
            double sqrdDistSum = ParallelUtils.run(parallel, X.size(), (start, end) ->
            {
                double sqrdDistChanges = 0.0;
                for (int i = start; i < end; i++)
                {
                    double newDist = dm.dist(newMeanIndx, i, X, accelCache);

                    newDist *= newDist;
                    if (newDist < closestDist[i] || forceCompute)
                    {
                        sqrdDistChanges -= closestDist[i];//on inital, -= 0  changes nothing. on others, removed the old value
                        sqrdDistChanges += newDist;
                        closestDist[i] = newDist;
                    }
                }

                return sqrdDistChanges;
            }, 
           (t, u) -> t + u);
            
            if(sqrdDistSum <= 1e-6)//everyone is too close, randomly fill rest
            {
                Set<Integer> ind = new IntSet();
                for(int i = 0;i <j; i++)
                    ind.add(indices[i]);
                while(ind.size() < k)
                    ind.add(rand.nextInt(closestDist.length));
                int pos = 0;
                for(int i : ind)
                    indices[pos++] = i;
                return;
            }

            //Choose new x as weighted probablity by the squared distances
            double rndX = rand.nextDouble() * sqrdDistSum;
            double searchSum = closestDist[0];
            int i = 0;
            while(searchSum < rndX && i < d.size()-1)
                searchSum += closestDist[++i];
            
            indices[j] = i;
        }
    }
    
    private static void ffSelection(final int[] indices, Random rand, final DataSet d, final int k, final DistanceMetric dm, final List<Double> accelCache, boolean parallel)
    {
        //Initial random point
        indices[0] = rand.nextInt(d.size());

        final double[] closestDist = new double[d.size()];
        Arrays.fill(closestDist, Double.POSITIVE_INFINITY);
        final List<Vec> X = d.getDataVectors();

        for (int j = 1; j < k; j++)
        {
            //Compute the distance from each data point to the closest mean
            final int newMeanIndx = indices[j - 1];//Only the most recently added mean needs to get distances computed. 

            //Atomic integer storres the index of the vector with the current maximum  minimum distance to a selected centroid
            final AtomicInteger maxDistIndx = new AtomicInteger(0);
            
            ParallelUtils.run(parallel, d.size(), (start, end)->
            {
                double maxDist = Double.NEGATIVE_INFINITY;
                int max = indices[0];//set to some lazy value, it will be changed
                for (int i = start; i < end; i++)
                {
                    double newDist = dm.dist(newMeanIndx, i, X, accelCache);
                    closestDist[i] = Math.min(newDist, closestDist[i]);

                    if (closestDist[i] > maxDist)
                    {
                        maxDist = closestDist[i];
                        max = i;
                    }
                }
                
                synchronized(maxDistIndx)
                {
                    if(closestDist[max] > closestDist[maxDistIndx.get()])
                        maxDistIndx.set(max);
                }
            });
            
            indices[j] = maxDistIndx.get();
        }
    }
    
    private static void mqSelection(final int[] indices, final DataSet d, final int k, final DistanceMetric dm, final List<Double> accelCache, boolean parallel)
    {
        final double[] meanDist = new double[d.size()];

        //Compute the distance from each data point to the closest mean
        final Vec newMean = MatrixStatistics.meanVector(d);
        final List<Double> meanQI = dm.getQueryInfo(newMean);
        final List<Vec> X = d.getDataVectors();

        ParallelUtils.run(parallel, d.size(), (start, end)->
        {
            for (int i = start; i < end; i++)
                meanDist[i] = dm.dist(i, newMean, meanQI, X, accelCache);
        });
        
        IndexTable indxTbl = new IndexTable(meanDist);
        for(int l = 0; l < k; l++)
            indices[l] = indxTbl.index(l*d.size()/k);
    }
    
}
