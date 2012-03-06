
package jsat.linear.distancemetrics;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.LUPDecomposition;
import jsat.linear.Matrix;
import jsat.linear.MatrixStatistics;
import jsat.linear.SingularValueDecomposition;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;

/**
 * The Mahalanobis Distance is a metric that takes into account the variance of the data. This requires 
 * training the metric with the data set to learn the variance of. The extra work involved adds 
 * computation time to training and prediction. However, improvements in accuracy can be obtained for 
 * many data sets. At the same time, the Mahalanobis Distance can also be detrimental to accuracy. 
 * 
 * @author Edward Raff
 */
public class MahalanobisDistance extends TrainableDistanceMetric
{
    /**
     * Used first, faster
     */
    private LUPDecomposition lup;
    /**
     * Used when LUP fails 
     */
    private SingularValueDecomposition svd;
    private boolean reTrain;

    public MahalanobisDistance()
    {
        reTrain = true;
    }

    /**
     * Returns <tt>true</tt> if this metric will indicate a need to be retrained 
     * once it has been trained once. This will mean {@link #needsTraining() } 
     * will always return true. <tt>false</tt> means the metric will not indicate
     * a need to be retrained once it has been trained once.
     * 
     * @return <tt>true</tt> if the data should always be retrained, <tt>false</tt> if it should not. 
     */
    public boolean isReTrain()
    {
        return reTrain;
    }

    /**
     * It may be desirable to have the metric trained only once, and use the same parameters
     * for all other training sessions of the learning algorithm using the metric. This can 
     * be controlled through this boolean. Setting <tt>true</tt> if this metric will indicate
     * a need to be retrained  once it has been trained once. This will mean {@link #needsTraining() } 
     * will always return true. <tt>false</tt> means the metric will not indicate
     * a need to be retrained once it has been trained once.
     * 
     * @param reTrain <tt>true</tt> to make the metric always request retraining, <tt>false</tt> so it will not. 
     */
    public void setReTrain(boolean reTrain)
    {
        this.reTrain = reTrain;
    }
    
    
    public <V extends Vec> void train(List<V> dataSet)
    {
        train(dataSet, null);
    }
    
    public <V extends Vec> void train(List<V> dataSet, ExecutorService threadpool)
    {
        Vec mean = MatrixStatistics.MeanVector(dataSet);
        Matrix covariance = MatrixStatistics.CovarianceMatrix(mean, dataSet);
        
        if(threadpool != null)
            lup = new LUPDecomposition(covariance.clone(), threadpool);
        else
            lup = new LUPDecomposition(covariance.clone());
        if(Math.abs(lup.det()) <= 1e-13)//Bad problem, use the SVD instead
        {
            lup = null;
            svd = new SingularValueDecomposition(covariance);
        }
    }
    
    public void train(DataSet dataSet)
    {
        train(dataSet, null);
    }
    
    public void train(DataSet dataSet, ExecutorService threadpool)
    {
        List<Vec> dataVecs = new ArrayList<Vec>(dataSet.getSampleSize());
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            dataVecs.add(dataSet.getDataPoint(i).getNumericalValues());
        
        train(dataVecs, threadpool);
    }

    public void train(ClassificationDataSet dataSet)
    {
        train( (DataSet) dataSet);
    }

    public void train(ClassificationDataSet dataSet, ExecutorService threadpool)
    {
        train( (DataSet) dataSet, threadpool);
    }

    public boolean supportsClassificationTraining()
    {
        return true;
    }

    public void train(RegressionDataSet dataSet)
    {
        train( (DataSet) dataSet);
    }

    public void train(RegressionDataSet dataSet, ExecutorService threadpool)
    {
        train( (DataSet) dataSet, threadpool);
    }

    public boolean supportsRegressionTraining()
    {
        return true;
    }

    public boolean needsTraining()
    {
        if(svd == null && lup == null)
            return true;
        else
            return isReTrain();
    }

    public double dist(Vec a, Vec b)
    {
        Vec aMb = a.subtract(b);
        Vec rightSide;
        if(lup != null)
            rightSide = lup.solve(aMb);
        else if (svd != null)
            rightSide = svd.solve(aMb);
        else
            throw new UntrainedModelException("Metric has not yet been trained");
        return Math.sqrt(aMb.dot(rightSide));
    }

    public boolean isSymmetric()
    {
        return true;
    }

    public boolean isSubadditive()
    {
        return true;
    }

    public boolean isIndiscemible()
    {
        return true;
    }

    public double metricBound()
    {
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public MahalanobisDistance clone()
    {
        MahalanobisDistance clone = new MahalanobisDistance();
        clone.reTrain = this.reTrain;
        if(this.lup != null)
            clone.lup = this.lup.clone();
        if(this.svd != null)
            clone.svd = this.svd.clone();
        return clone;
    }
}
