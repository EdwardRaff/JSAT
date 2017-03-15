
package jsat.clustering;

import java.util.*;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.utils.DoubleList;
import jsat.utils.random.RandomUtil;

/**
 *
 * @author Edward Raff
 */
public class CLARA extends PAM
{

	private static final long serialVersionUID = 174392533688953706L;
	/**
     * The number of samples to take
     */
    private int sampleSize;
    /**
     * The number of times to do sampling
     */
    private int sampleCount;
    private boolean autoSampleSize;

    public CLARA(int sampleSize, int sampleCount, DistanceMetric dm, Random rand, SeedSelection seedSelection)
    {
        super(dm, rand, seedSelection);
        this.sampleSize = sampleSize;
        this.sampleCount = sampleCount;
        this.autoSampleSize = false;
    }
    
    public CLARA(int sampleCount, DistanceMetric dm, Random rand, SeedSelection seedSelection)
    {
        super(dm, rand, seedSelection);
        this.sampleSize = -1;
        this.sampleCount = sampleCount;
        this.autoSampleSize = true;
    }

    public CLARA(DistanceMetric dm, Random rand, SeedSelection seedSelection)
    {
        this(5, dm, rand, seedSelection);
    }

    public CLARA(DistanceMetric dm, Random rand)
    {
        this(dm, rand, SeedSelection.KPP);
    }

    public CLARA(DistanceMetric dm)
    {
        this(dm, RandomUtil.getRandom());
    }

    public CLARA()
    {
        this(new EuclideanDistance());
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public CLARA(CLARA toCopy)
    {
        super(toCopy);
        this.sampleSize = toCopy.sampleSize;
        this.sampleCount = toCopy.sampleCount;
        this.autoSampleSize = toCopy.autoSampleSize;
    }
    
    
    
    /**
     * 
     * @return the number of times {@link PAM} will be applied to a sample from the data set. 
     */
    public int getSampleCount()
    {
        return sampleCount;
    }

    /**
     * Sets the number of times {@link PAM} will be applied to different samples from the data set. 
     * 
     * @param sampleCount the number of times to apply sampeling. 
     */
    public void setSampleCount(int sampleCount)
    {
        this.sampleCount = sampleCount;
    }

    /**
     * 
     * @return the number of samples that will be taken to perform {@link PAM} on. 
     */
    public int getSampleSize()
    {
        return sampleSize;
    }

    /**
     * Sets the number of samples CLARA should take from the data set to perform {@link PAM} on. 
     * 
     * @param sampleSize the number of samples to take
     */
    public void setSampleSize(int sampleSize)
    {
        if(sampleSize >= 0)
        {
            autoSampleSize = false;
            this.sampleSize = sampleSize;
        }
        else 
            autoSampleSize = true;
    }
    
    @Override
    protected double cluster(DataSet data, boolean doInit, int[] medioids, int[] assignments, List<Double> cacheAccel)
    {
        int k = medioids.length;
        int[] bestMedoids = new int[medioids.length];
        int[] bestAssignments = new int[assignments.length];
        double bestMedoidsDist = Double.MAX_VALUE;
        List<Vec> X = data.getDataVectors();
        
        
        if(sampleSize >= data.getSampleSize())//Then we might as well just do one round of PAM
        {
            return super.cluster(data, true, medioids, assignments, cacheAccel);
        }
        else if(doInit)
        {
            TrainableDistanceMetric.trainIfNeeded(dm, data);
            cacheAccel = dm.getAccelerationCache(X);
        }
        
        int sampSize = autoSampleSize ? 40+2*k : sampleSize;
        int[] sampleAssignments = new int[sampSize];
        
        List<DataPoint> sample = new ArrayList<DataPoint>(sampSize);
        /**
         * We need the mapping to be able to go from the sample indicies back to their position in the full data set
         * Key is the sample index [1, 2, 3, ..., sampSize]
         * Value is the coresponding index in the full data set
         */
        Map<Integer, Integer> samplePoints = new LinkedHashMap<Integer, Integer>();
        DoubleList subCache = new DoubleList(sampSize);
        
        for(int i = 0; i < sampleCount; i++)
        {
            //Take a sample and use PAM on it to get medoids
            samplePoints.clear();
            sample.clear();
            subCache.clear();
            
            while (samplePoints.size() < sampSize)
            {
                int indx = rand.nextInt(data.getSampleSize());
                if (!samplePoints.containsValue(indx))
                    samplePoints.put(samplePoints.size(), indx);
            }
            for (Integer j : samplePoints.values())
            {
                sample.add(data.getDataPoint(j));
                subCache.add(cacheAccel.get(j));
            }

            DataSet sampleSet = new SimpleDataSet(sample);
            
            //Sampling done, now apply PAM
            SeedSelectionMethods.selectIntialPoints(sampleSet, medioids, dm, subCache, rand, getSeedSelection());
            super.cluster(sampleSet, false, medioids, sampleAssignments, subCache);
            
            //Map the sample medoids back to the full data set
            for(int j = 0; j < medioids.length; j++)
                medioids[j] = samplePoints.get(medioids[j]);
            
            //Now apply the sample medoids to the full data set
            double sqrdDist = 0.0;
            for(int j = 0; j < data.getSampleSize(); j++)
            {
                double smallestDist = Double.MAX_VALUE;
                int assignment = -1;
                
                for(int z = 0; z < k; z++)
                {
                    double tmp = dm.dist(medioids[z], j, X, cacheAccel);
                    if(tmp < smallestDist)
                    {
                        assignment = z;
                        smallestDist = tmp;
                    }
                }
                assignments[j] = assignment;
                sqrdDist += smallestDist*smallestDist;
            }
            
            if(sqrdDist < bestMedoidsDist)
            {
                bestMedoidsDist = sqrdDist;
                System.arraycopy(medioids, 0, bestMedoids, 0, k);
                System.arraycopy(assignments, 0, bestAssignments, 0, assignments.length);
            }
        }
        
        System.arraycopy(bestMedoids, 0, medioids, 0, k);
        System.arraycopy(bestAssignments, 0, assignments, 0, assignments.length);
        
        return bestMedoidsDist;
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        medoids = new int[clusters];
        
        this.cluster(dataSet, true, medoids, designations, null);
        if(!storeMedoids)
            medoids = null;
        
        return designations;
    }

    @Override
    public CLARA clone()
    {
        return new CLARA(this);
    }
    
}
