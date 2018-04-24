
package jsat.linear.distancemetrics;

import java.util.List;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.*;
import jsat.regression.RegressionDataSet;
import jsat.utils.concurrent.ParallelUtils;

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

    private static final long serialVersionUID = 7878528119699276817L;
    private boolean reTrain;
    /**
     * The inverse of the covariance matrix 
     */
    private Matrix S;

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
    
    /**
     * Sets the Inverse Covariance Matrix used as the distance matrix by this
     * distance metric.
     *
     * @param S the matrix to use as the distance matrix
     */
    public void setInverseCovariance(Matrix S)
    {
        this.S = S;
    }
    
    
    @Override
    public <V extends Vec> void train(List<V> dataSet)
    {
        train(dataSet, false);
    }
    
    @Override
    public <V extends Vec> void train(List<V> dataSet, boolean parallel)
    {
        Vec mean = MatrixStatistics.meanVector(dataSet);
        Matrix covariance = MatrixStatistics.covarianceMatrix(mean, dataSet);
        LUPDecomposition lup;
        SingularValueDecomposition svd;
        if(parallel)
            lup = new LUPDecomposition(covariance.clone(), ParallelUtils.CACHED_THREAD_POOL);
        else
            lup = new LUPDecomposition(covariance.clone());
        double det = lup.det();
        if(Double.isNaN(det) || Double.isInfinite(det) || Math.abs(det) <= 1e-13)//Bad problem, use the SVD instead
        {
            lup = null;
            svd = new SingularValueDecomposition(covariance);
            S = svd.getPseudoInverse();
        }
        else if(parallel)
            S = lup.solve(Matrix.eye(covariance.cols()), ParallelUtils.CACHED_THREAD_POOL);
        else
            S = lup.solve(Matrix.eye(covariance.cols()));
    }
    
    @Override
    public void train(DataSet dataSet)
    {
        train(dataSet, false);
    }
    
    @Override
    public void train(DataSet dataSet, boolean parallel)
    {
        train(dataSet.getDataVectors(), parallel);
    }

    @Override
    public void train(ClassificationDataSet dataSet)
    {
        train( (DataSet) dataSet);
    }

    @Override
    public void train(ClassificationDataSet dataSet, boolean parallel)
    {
        train((DataSet) dataSet, parallel);
    }

    @Override
    public boolean supportsClassificationTraining()
    {
        return true;
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train( (DataSet) dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet, boolean parallel)
    {
        train((DataSet) dataSet, parallel);
    }

    @Override
    public boolean supportsRegressionTraining()
    {
        return true;
    }

    @Override
    public boolean needsTraining()
    {
        if(S == null)
            return true;
        else
            return isReTrain();
    }

    @Override
    public double dist(Vec a, Vec b)
    {
        Vec aMb = a.subtract(b);
        return Math.sqrt(aMb.dot(S.multiply(aMb)));
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

    @Override
    public String toString()
    {
        return "Mahalanobis Distance";
    }
    
    @Override
    public MahalanobisDistance clone()
    {
        MahalanobisDistance clone = new MahalanobisDistance();
        clone.reTrain = this.reTrain;
        if(this.S != null)
            clone.S = this.S.clone();
        return clone;
    }
}
