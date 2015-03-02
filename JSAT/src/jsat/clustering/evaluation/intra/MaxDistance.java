
package jsat.clustering.evaluation.intra;

import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * Evaluates a cluster's validity by returning the 
 * maximum distance between any two points in the cluster. 
 * 
 * @author Edward Raff
 */
public class MaxDistance implements IntraClusterEvaluation
{
    private DistanceMetric dm;

    /**
     * Creates a new MaxDistance measure using the {@link EuclideanDistance}
     */
    public MaxDistance()
    {
        this(new EuclideanDistance());
    }
    /**
     * Creates a new MaxDistance
     * @param dm the metric to measure the distance between two points by
     */
    public MaxDistance(DistanceMetric dm)
    {
        this.dm = dm;
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public MaxDistance(MaxDistance toCopy)
    {
        this(toCopy.dm.clone());
    }
    
    @Override
    public double evaluate(int[] designations, DataSet dataSet, int clusterID)
    {
        double maxDistance = 0;
        for (int i = 0; i < dataSet.getSampleSize(); i++)
            for (int j = i + 1; j < dataSet.getSampleSize(); j++)
                if (designations[i] == clusterID)
                    maxDistance = Math.max(
                            dm.dist(dataSet.getDataPoint(i).getNumericalValues(),
                                    dataSet.getDataPoint(j).getNumericalValues()),
                            maxDistance);
        return maxDistance;
    }

    @Override
    public double evaluate(List<DataPoint> dataPoints)
    {
        double maxDistance = 0;
        for(int i = 0; i < dataPoints.size(); i++)
            for(int j = i+1; j < dataPoints.size(); j++ )
                maxDistance = Math.max(
                        dm.dist(dataPoints.get(i).getNumericalValues(), 
                                dataPoints.get(j).getNumericalValues()), 
                        maxDistance);
        
        return maxDistance;
    }

    @Override
    public MaxDistance clone()
    {
        return new MaxDistance(this);
    }
    
}
