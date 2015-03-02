package jsat.clustering.evaluation.intra;

import java.util.List;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.*;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * Evaluates a cluster's validity by computing the sum of squared distances from
 * each point to the mean of the cluster. 
 * 
 * @author Edward Raff
 */
public class SoSCentroidDistance implements IntraClusterEvaluation
{
    private DistanceMetric dm;

    /**
     * Creates a new MeanCentroidDistance using the {@link EuclideanDistance}
     */
    public SoSCentroidDistance()
    {
        this(new EuclideanDistance());
    }
    
    /**
     * Creates a new MeanCentroidDistance. 
     * @param dm the metric to measure the distance between two points by
     */
    public SoSCentroidDistance(DistanceMetric dm)
    {
        this.dm = dm;
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public SoSCentroidDistance(SoSCentroidDistance toCopy)
    {
        this(toCopy.dm.clone());
    }
    
    @Override
    public double evaluate(int[] designations, DataSet dataSet, int clusterID)
    {
        Vec mean = new DenseVector(dataSet.getNumNumericalVars());
        
        int clusterSize = 0;
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            if(designations[i] == clusterID)
            {
                clusterSize++;
                mean.mutableAdd(dataSet.getDataPoint(i).getNumericalValues());
            }
        mean.mutableDivide(clusterSize);
        
        
        double score = 0.0;
        
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            if(designations[i] == clusterID)
                score += Math.pow(dm.dist(dataSet.getDataPoint(i).getNumericalValues(), mean), 2);
        
        return score;
    }

    @Override
    public double evaluate(List<DataPoint> dataPoints)
    {
        if(dataPoints.isEmpty())
            return 0;
        Vec mean = MatrixStatistics.meanVector(new SimpleDataSet(dataPoints));
        
        double score = 0.0;
        for(DataPoint dp : dataPoints)
            score += Math.pow(dm.dist(dp.getNumericalValues(), mean), 2);
        
        return score;
    }

    @Override
    public SoSCentroidDistance clone()
    {
        return new SoSCentroidDistance(this);
    }
    
}
