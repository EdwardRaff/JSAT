
package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.*;
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
    
    
    @Override
    public <V extends Vec> void train(List<V> dataSet)
    {
        train(dataSet, null);
    }
    
    @Override
    public <V extends Vec> void train(List<V> dataSet, ExecutorService threadpool)
    {
        Vec mean = MatrixStatistics.meanVector(dataSet);
        Matrix covariance = MatrixStatistics.covarianceMatrix(mean, dataSet);
        LUPDecomposition lup;
        SingularValueDecomposition svd;
        if(threadpool != null)
            lup = new LUPDecomposition(covariance.clone(), threadpool);
        else
            lup = new LUPDecomposition(covariance.clone());
        double det = lup.det();
        if(Double.isNaN(det) || Double.isInfinite(det) || Math.abs(det) <= 1e-13)//Bad problem, use the SVD instead
        {
            lup = null;
            svd = new SingularValueDecomposition(covariance);
            S = svd.getPseudoInverse();
        }
        else if(threadpool != null)
            S = lup.solve(Matrix.eye(covariance.cols()), threadpool);
        else
            S = lup.solve(Matrix.eye(covariance.cols()));
    }
    
    @Override
    public void train(DataSet dataSet)
    {
        train(dataSet, null);
    }
    
    @Override
    public void train(DataSet dataSet, ExecutorService threadpool)
    {
        train(dataSet.getDataVectors(), threadpool);
    }

    @Override
    public void train(ClassificationDataSet dataSet)
    {
        train( (DataSet) dataSet);
    }

    @Override
    public void train(ClassificationDataSet dataSet, ExecutorService threadpool)
    {
        train( (DataSet) dataSet, threadpool);
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
    public void train(RegressionDataSet dataSet, ExecutorService threadpool)
    {
        train( (DataSet) dataSet, threadpool);
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

    @Override
    public boolean supportsAcceleration()
    {
        return false;
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> vecs)
    {
        return null;
    }

    @Override
    public double dist(int a, int b, List<? extends Vec> vecs, List<Double> cache)
    {
        return dist(vecs.get(a), vecs.get(b));
    }

    @Override
    public double dist(int a, Vec b, List<? extends Vec> vecs, List<Double> cache)
    {
        return dist(vecs.get(a), b);
    }

    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        return null;
    }
    
    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> vecs, ExecutorService threadpool)
    {
        return null;
    }

    @Override
    public double dist(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        return dist(vecs.get(a), b);
    }

    
}
