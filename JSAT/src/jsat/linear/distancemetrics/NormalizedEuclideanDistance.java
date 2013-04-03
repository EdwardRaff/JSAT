package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.datatransform.UnitVarianceTransform;
import jsat.linear.MatrixStatistics;
import jsat.linear.Vec;
import jsat.linear.VecOps;
import jsat.math.FunctionBase;
import jsat.math.MathTricks;
import jsat.regression.RegressionDataSet;

/**
 * Implementation of the Normalized Euclidean Distance Metric. The normalized 
 * version divides each variable by its standard deviation, and then continues 
 * as the normal {@link EuclideanDistance}. <br>
 * The same results can be achieved by first applying 
 * {@link UnitVarianceTransform} to a data set before using the 
 * L2 norm.<br> 
 * It is equivalent to the {@link MahalanobisDistance} if only the diagonal 
 * values were used. 
 * 
 * 
 * @author Edward Raff
 */
public class NormalizedEuclideanDistance extends TrainableDistanceMetric
{
    private Vec invStndDevs;

    /**
     * Creates a new Normalized Euclidean distance metric
     */
    public NormalizedEuclideanDistance()
    {
    }

    @Override
    public <V extends Vec> void train(List<V> dataSet)
    {
        invStndDevs = MatrixStatistics.covarianceDiag(MatrixStatistics.meanVector(dataSet), dataSet);
        invStndDevs.applyFunction(MathTricks.sqrdFunc);
        invStndDevs.applyFunction(MathTricks.invsFunc);
    }

    @Override
    public <V extends Vec> void train(List<V> dataSet, ExecutorService threadpool)
    {
        train(dataSet);
    }

    @Override
    public void train(DataSet dataSet)
    {
        invStndDevs = dataSet.getColumnMeanVariance()[1];
        invStndDevs.applyFunction(MathTricks.sqrdFunc);
        invStndDevs.applyFunction(MathTricks.invsFunc);
    }

    @Override
    public void train(DataSet dataSet, ExecutorService threadpool)
    {
        train(dataSet);
    }

    @Override
    public void train(ClassificationDataSet dataSet)
    {
        train((DataSet)dataSet);
    }

    @Override
    public void train(ClassificationDataSet dataSet, ExecutorService threadpool)
    {
        train(dataSet);
    }

    @Override
    public boolean supportsClassificationTraining()
    {
        return true;
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train((DataSet)dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadpool)
    {
        train(dataSet);
    }

    @Override
    public boolean supportsRegressionTraining()
    {
        return true;
    }

    @Override
    public boolean needsTraining()
    {
        return invStndDevs == null;
    }

    @Override
    public TrainableDistanceMetric clone()
    {
        NormalizedEuclideanDistance clone = new NormalizedEuclideanDistance();
        if(this.invStndDevs != null)
            clone.invStndDevs = this.invStndDevs.clone();
        return clone;
    }

    @Override
    public double dist(Vec a, Vec b)
    {
        double r = VecOps.accumulateSum(invStndDevs, a, b, new FunctionBase() 
        {
            @Override
            public double f(Vec x)
            {
                return Math.pow(x.get(0), 2);
            }
        });
        
        return Math.sqrt(r);
    }

    @Override
    public boolean isSymmetric()
    {
        return true;
    }

    @Override
    public boolean isSubadditive()
    {
        return true;
    }

    @Override
    public boolean isIndiscemible()
    {
        return true;
    }

    @Override
    public double metricBound()
    {
        return Double.POSITIVE_INFINITY;
    }
    
}
