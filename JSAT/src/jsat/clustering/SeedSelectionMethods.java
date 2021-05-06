
package jsat.clustering;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.MatrixStatistics;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.VPTreeMV;
import jsat.utils.*;
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
	 * Faster version of the k-means++ seeding algorithm. <br>
	 * <br>
	 * See: "Exact Acceleration of K-Means++ and K-Means‖" IJAI 2021
	 */
	KPP_TIA,
	
	/**
	 * The K-Means|| algorithm <br>
	 * <br>
	 * See: ﻿Bahmani, B., Moseley, B., Vattani, A., Kumar, R., and
	 * Vassilvitskii, S. (2012). Scalable K-means++. Proc. VLDB Endow.,
	 * 5(7), 622–633. 
	 */
	KBB,
	/**
	 * Faster version of the K-Means|| seeding algorithm. <br>
	 * <br>
	 * See: "Exact Acceleration of K-Means++ and K-Means‖" IJAI 2021
	 */
	KBB_TIA,
        
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
		case KPP_TIA:
                    kppSelectionTIA(indices, rand, d, k, dm, accelCache, parallel);
                    break;
                case KPP:
                    kppSelection(indices, rand, d, k, dm, accelCache, parallel);
                    break;
		case KBB_TIA:
                    kbbSelectionTIA(indices, rand, d, k, dm, accelCache, parallel);
                    break;
		case KBB:
                    kbbSelection(indices, rand, d, k, dm, accelCache, parallel);
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
	
	Vec w = d.getDataWeights();

        final double[] closestDist = new double[d.size()];
        Arrays.fill(closestDist, Double.POSITIVE_INFINITY);
        final List<Vec> X = d.getDataVectors();

        for (int j = 1; j < k; j++)
        {
            //Compute the distance from each data point to the closest mean
            final int newMeanIndx = indices[j - 1];//Only the most recently added mean needs to get distances computed. 
            
            double sqrdDistSum = ParallelUtils.run(parallel, X.size(), (start, end) ->
            {
                double partial_sqrd_dist = 0.0;
                for (int i = start; i < end; i++)
                {
                    double newDist = dm.dist(newMeanIndx, i, X, accelCache);

                    newDist *= newDist;
                    if (newDist < closestDist[i])
                        closestDist[i] = newDist;
                    partial_sqrd_dist += closestDist[i]*w.get(i);
                }

                return partial_sqrd_dist;
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
            double searchSum = closestDist[0]*w.get(0);
            int i = 0;
            while(searchSum < rndX && i < d.size()-1)
                searchSum += closestDist[++i]*w.get(i);
            
            indices[j] = i;
        }
    }
    //accelerated variant from Exact Acceleration of K-Means++ and K-Means‖
    private static void kppSelectionTIA(final int[] indices, Random rand, final DataSet d, final int k, final DistanceMetric dm, final List<Double> accelCache, boolean parallel)
    {
        /*
         * http://www.stanford.edu/~darthur/kMeansPlusPlus.pdf : k-means++: The Advantages of Careful Seeding
         *
         */

        final double[] closestDist = new double[d.size()];
	Arrays.fill(closestDist, Double.POSITIVE_INFINITY);
	final int[] closest_mean = new int[d.size()];
	Arrays.fill(closest_mean, 0);
	
	
	Vec w = d.getDataWeights();
	final double[] expo_sample = new double[d.size()];
	indices[0] = 0;//First initial seed
	for(int i = 0; i < d.size(); i++)
	{
	    double p = rand.nextDouble();
	    expo_sample[i] = -Math.log(1-p)/w.get(i);//dont use FastMath b/c we need to make sure all values are strictly positive
	    if(expo_sample[i] < expo_sample[indices[0]])
		indices[0] = i;
	}
	
	
	
	
	final double[] sample_weight = new double[d.size()];
	PriorityQueue<Integer> nextSample = new PriorityQueue<>(expo_sample.length, (a, b) -> Double.compare(sample_weight[a], sample_weight[b])); 
	
	IntList dirtyItemsToFix = new IntList();
	boolean[] dirty = new boolean[d.size()];
	Arrays.fill(dirty, false);
	
	
        //Initial random point
	closestDist[indices[0]] = 0.0;
        
        final List<Vec> X = d.getDataVectors();
	
	double[] gamma = new double[k];
	Arrays.fill(gamma, Double.MAX_VALUE);

	double prev_partial = 0;
        for (int j = 1; j < k; j++)
        {
	    final int jj = j;
            //Compute the distance from each data point to the closest mean
            final int newMeanIndx = indices[j - 1];//Only the most recently added mean needs to get distances computed. 
	    
            double sqrdDistSum = ParallelUtils.run(parallel, X.size(), (start, end) ->
            {
                double partial_sqrd_dist = 0.0;
                for (int i = start; i < end; i++)
                {
		    //mul by 4 b/c gamma and closestDist are the _squared_ distances, not raw. 
		    if(gamma[closest_mean[i]] < 4* closestDist[i])
		    {
			double newDist = dm.dist(newMeanIndx, i, X, accelCache);
			
			newDist *= newDist;
			if (newDist < closestDist[i])
			{	    
			    
			    if(jj > 1)
			    {
				partial_sqrd_dist -= closestDist[i]*w.get(i);
				dirty[i] = true;
			    }
			    else
			    {
				sample_weight[i] = expo_sample[i]/(newDist);
				nextSample.add(i);
			    }
			    closest_mean[i] = jj-1;
			    closestDist[i] = newDist;
			    partial_sqrd_dist += closestDist[i]*w.get(i);
			    
			}
		    }
                }

                return partial_sqrd_dist;
            }, 
           (t, u) -> t + u);
	    
	    if(prev_partial != 0)
	    {
		sqrdDistSum = prev_partial + sqrdDistSum;
	    }
            prev_partial = sqrdDistSum;
	    
            if(sqrdDistSum <= 1e-6)//everyone is too close, randomly fill rest
            {
//		System.out.println("BAILL");
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

	    int tries = 0;//for debugging 
	    
	    //Search till we find first clean item
	    while(!nextSample.isEmpty() && dirty[nextSample.peek()])
		dirtyItemsToFix.add(nextSample.poll());
	    for(int i : dirtyItemsToFix)//fix all the dirty items!
		sample_weight[i] = expo_sample[i]/(closestDist[i]);
	    nextSample.addAll(dirtyItemsToFix);//put them back in the Q
	    dirtyItemsToFix.clear();//done, clean up
            while(true)//this should only happen once, kept for debugging purposes
	    {
		tries++;
		int next_indx = nextSample.poll();
		if(dirty[next_indx])//this should not enter, kept for debugging purposes
		{
		    sample_weight[next_indx] = expo_sample[next_indx]/(closestDist[next_indx]);
		    dirty[next_indx] = false;
		    nextSample.add(next_indx);
		}
		else
		{
		    indices[j] = next_indx;
		    break;
		}
	    }
	    //now we have new index, determine dists to prev means
	    if(j+1 < k)
	    {
		//for(k_prev = 0; k_prev < j; k_prev++)
		ParallelUtils.run(parallel, j, (k_prev, end) ->
		{
		    for(; k_prev < end; k_prev++)
		    {
			gamma[k_prev] = Math.pow(dm.dist(indices[k_prev], indices[jj], X, accelCache), 2);
		    }
		});
	    }
        }
    }
    
    private static void kbbSelection(final int[] indices, Random rand, final DataSet d, final int k, final DistanceMetric dm, final List<Double> accelCache, boolean parallel)
    {
	
	int trials = 5;
	int oversample = 2*k;
	
	//Initial random point
//        indices[0] = rand.nextInt(d.size());
	int[] assigned_too = new int[d.size()];
	IntList C = new IntList(trials*oversample);
	C.add(rand.nextInt(d.size()));//Initial random point
	
	Vec w = d.getDataWeights();

        final double[] closestDist = new double[d.size()];
        Arrays.fill(closestDist, Double.POSITIVE_INFINITY);
        final List<Vec> X = d.getDataVectors();
	
	
	//init poitns to initial center
	double sqrdDistSum = ParallelUtils.run(parallel, X.size(), (start, end) ->
	{
	    double partial_sqrd_dist = 0.0;
	    for (int i = start; i < end; i++)
	    {
		double newDist = dm.dist(C.getI(0), i, X, accelCache);

		newDist *= newDist;
		if (newDist < closestDist[i])
		    closestDist[i] = newDist;
		partial_sqrd_dist += closestDist[i]*w.get(i);
	    }

	    return partial_sqrd_dist;
	}, 
       (z, u) -> z + u);
	
	for(int t = 0; t < trials; t++)
        {
            //Lets sample some new points
	    int orig_size = C.size();
	    for(int i = 0; i < X.size(); i++)
		if(w.get(i)*oversample*closestDist[i]/sqrdDistSum > rand.nextDouble())//sample!
		    C.add(i);
	    
	    sqrdDistSum = ParallelUtils.run(parallel, X.size(), (start, end) ->
	    {
		double partial_sqrd_dist = 0.0;
		for (int i = start; i < end; i++)
		{
		    if(closestDist[i] == 0)
			continue;
		    for(int j = orig_size; j < C.size(); j++)
		    {
			double newDist = dm.dist(C.get(j), i, X, accelCache);

			newDist *= newDist;
			if (newDist < closestDist[i])
			{
			    closestDist[i] = newDist;
			    assigned_too[i] = j;
			}
		    }
		    partial_sqrd_dist += closestDist[i]*w.get(i);
		}

		return partial_sqrd_dist;
	    }, 
	   (z, u) -> z + u);
        }
	
	
	Vec weights = new DenseVector(C.size());
	for(int i = 0; i < X.size(); i++)
	    weights.increment(assigned_too[i], w.get(i));
	SimpleDataSet sds = new SimpleDataSet(d.getNumNumericalVars(), new CategoricalData[0]);
	for(int j : C)
	{
	    sds.add(new DataPoint(X.get(j)));
	    sds.setWeight(sds.size()-1, weights.get(sds.size()-1));
	}
	
	//run k-means++ on the weighted set of selected over-samples
	kppSelection(indices, rand, sds, k, dm, dm.getAccelerationCache(sds.getDataVectors(), parallel), parallel);
	//map final seeds back to original vectors
	for(int i = 0; i < k; i++)
	    indices[i] = C.getI(indices[i]);
    }
    
    //accelerated variant from Exact Acceleration of K-Means++ and K-Means‖
    private static void kbbSelectionTIA(final int[] indices, Random rand, final DataSet d, final int k, final DistanceMetric dm, final List<Double> accelCache, boolean parallel)
    {
	
	int trials = 5;
	int oversample = 2*k;
	
	//Initial random point
	int[] assigned_too = new int[d.size()];
	IntList C = new IntList(trials*oversample);
	C.add(rand.nextInt(d.size()));//Initial random point
	
	Vec w = d.getDataWeights();

        final double[] closestDist = new double[d.size()];
        Arrays.fill(closestDist, Double.POSITIVE_INFINITY);
        final List<Vec> X = d.getDataVectors();
	
	
	//init poitns to initial center
	double sqrdDistSum = ParallelUtils.run(parallel, X.size(), (start, end) ->
	{
	    double partial_sqrd_dist = 0.0;
	    for (int i = start; i < end; i++)
	    {
		double newDist = dm.dist(C.getI(0), i, X, accelCache);

		newDist *= newDist;
		if (newDist < closestDist[i])
		    closestDist[i] = newDist;
		partial_sqrd_dist += closestDist[i]*w.get(i);
	    }

	    return partial_sqrd_dist;
	}, 
       (z, u) -> z + u);
	
	for(int t = 0; t < trials; t++)
        {
            //Lets sample some new points
	    int orig_size = C.size();
	    for(int i = 0; i < X.size(); i++)
		if(w.get(i)*oversample*closestDist[i]/sqrdDistSum > rand.nextDouble())//sample!
		    C.add(i);
	    
	    List<Integer> to_assign = C.subList(orig_size, C.size());
	    List<Vec> X_new_means = new ArrayList<>(to_assign.size());
	    for(int j : to_assign)
		X_new_means.add(X.get(j));
	    
	    VPTreeMV<Vec> vp = new VPTreeMV<>(X_new_means, dm, parallel);
            
	    sqrdDistSum = ParallelUtils.run(parallel, X.size(), (start, end) ->
	    {
		double partial_sqrd_dist = 0.0;
		
		IntList neighbors = new IntList();
		DoubleList distances = new DoubleList();
		
		for (int i = start; i < end; i++)
		{
		    if(closestDist[i] == 0)
			continue;
		    
		    neighbors.clear();
		    distances.clear();
		    vp.search(X.get(i), 1, Math.sqrt(closestDist[i]), neighbors, distances);
		    
		    if(distances.isEmpty())//no one within radius!
			continue;
		    
		    double newDist = distances.getD(0);

		    newDist *= newDist;
		    if (newDist < closestDist[i])
		    {
			closestDist[i] = newDist;
			assigned_too[i] = orig_size + neighbors.getI(0);
		    }
		    
		    partial_sqrd_dist += closestDist[i]*w.get(i);
		}

		return partial_sqrd_dist;
	    }, 
	   (z, u) -> z + u);
        }
	
	
	Vec weights = new DenseVector(C.size());
	for(int i = 0; i < X.size(); i++)
	    weights.increment(assigned_too[i], w.get(i));
	SimpleDataSet sds = new SimpleDataSet(d.getNumNumericalVars(), new CategoricalData[0]);
	for(int j : C)
	{
	    sds.add(new DataPoint(X.get(j)));
	    sds.setWeight(sds.size()-1, weights.get(sds.size()-1));
	}
	//run k-means++ on the weighted set of selected over-samples
	kppSelectionTIA(indices, rand, sds, k, dm, dm.getAccelerationCache(sds.getDataVectors(), parallel), parallel);
//	kppSelection(indices, rand, sds, k, dm, dm.getAccelerationCache(sds.getDataVectors(), parallel), parallel);
	//map final seeds back to original vectors
	for(int i = 0; i < k; i++)
	    indices[i] = C.getI(indices[i]);
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
