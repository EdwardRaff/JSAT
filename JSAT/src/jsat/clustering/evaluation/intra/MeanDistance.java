package jsat.clustering.evaluation.intra;

import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * Evaluates a cluster's validity by computing the mean distance between all 
 * combinations of points. 
 * 
 * @author Edwar Raff
 */
public class MeanDistance implements IntraClusterEvaluation
{
    private DistanceMetric dm;

    /**
     * Creates a new MeanDistance using the {@link EuclideanDistance}
     */
    public MeanDistance()
    {
        this(new EuclideanDistance());
    }

    /**
     * Creates a new MeanDistance
     * @param dm the metric to measure the distance between two points by
     */
    public MeanDistance(DistanceMetric dm)
    {
        this.dm = dm;
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public MeanDistance(MeanDistance toCopy)
    {
        this(toCopy.dm.clone());
    }
    
    @Override
    public double evaluate(int[] designations, DataSet dataSet, int clusterID)
    {
        double distances = 0;
        for (int i = 0; i < dataSet.getSampleSize(); i++)
            for (int j = i + 1; j < dataSet.getSampleSize(); j++)
                if (designations[i] == clusterID)
                    distances += dm.dist(dataSet.getDataPoint(i).getNumericalValues(),
                                         dataSet.getDataPoint(j).getNumericalValues());
        return distances/(dataSet.getSampleSize()*(dataSet.getSampleSize()-1));
    }

    @Override
    public double evaluate(List<DataPoint> dataPoints)
    {
        double distances = 0.0;
        for(int i = 0; i < dataPoints.size(); i++)
            for(int j = i+1; j < dataPoints.size(); j++ )
                distances += dm.dist(dataPoints.get(i).getNumericalValues(),
                                     dataPoints.get(j).getNumericalValues());
        
        return distances/(dataPoints.size()*(dataPoints.size()-1));
    }

    @Override
    public MeanDistance clone()
    {
        return new MeanDistance(this);
    }
    
}
