package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.LUPDecomposition;
import jsat.linear.Matrix;
import jsat.linear.MatrixStatistics;
import jsat.linear.SingularValueDecomposition;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;

/**
 * The Mahalanobis Distance is a metric that takes into account the variance of
 * the data. This requires training the metric with the data set to learn the
 * variance of. The extra work involved adds computation time to training and
 * prediction. However, improvements in accuracy can be obtained for many data
 * sets. At the same time, the Mahalanobis Distance can also be detrimental to
 * accuracy.
 *
 * @author Edward Raff
 */
public class MahalanobisDistance extends TrainableDistanceMetric {

  private static final long serialVersionUID = 7878528119699276817L;
  private boolean reTrain;
  /**
   * The inverse of the covariance matrix
   */
  private Matrix S;

  public MahalanobisDistance() {
    reTrain = true;
  }

  @Override
  public MahalanobisDistance clone() {
    final MahalanobisDistance clone = new MahalanobisDistance();
    clone.reTrain = reTrain;
    if (S != null) {
      clone.S = S.clone();
    }
    return clone;
  }

  @Override
  public double dist(final int a, final int b, final List<? extends Vec> vecs, final List<Double> cache) {
    return dist(vecs.get(a), vecs.get(b));
  }

  @Override
  public double dist(final int a, final Vec b, final List<? extends Vec> vecs, final List<Double> cache) {
    return dist(vecs.get(a), b);
  }

  @Override
  public double dist(final int a, final Vec b, final List<Double> qi, final List<? extends Vec> vecs,
      final List<Double> cache) {
    return dist(vecs.get(a), b);
  }

  @Override
  public double dist(final Vec a, final Vec b) {
    final Vec aMb = a.subtract(b);
    return Math.sqrt(aMb.dot(S.multiply(aMb)));
  }

  @Override
  public List<Double> getAccelerationCache(final List<? extends Vec> vecs) {
    return null;
  }

  @Override
  public List<Double> getAccelerationCache(final List<? extends Vec> vecs, final ExecutorService threadpool) {
    return null;
  }

  @Override
  public List<Double> getQueryInfo(final Vec q) {
    return null;
  }

  @Override
  public boolean isIndiscemible() {
    return true;
  }

  /**
   * Returns <tt>true</tt> if this metric will indicate a need to be retrained
   * once it has been trained once. This will mean {@link #needsTraining() }
   * will always return true. <tt>false</tt> means the metric will not indicate
   * a need to be retrained once it has been trained once.
   *
   * @return <tt>true</tt> if the data should always be retrained,
   *         <tt>false</tt> if it should not.
   */
  public boolean isReTrain() {
    return reTrain;
  }

  @Override
  public boolean isSubadditive() {
    return true;
  }

  @Override
  public boolean isSymmetric() {
    return true;
  }

  @Override
  public double metricBound() {
    return Double.POSITIVE_INFINITY;
  }

  @Override
  public boolean needsTraining() {
    if (S == null) {
      return true;
    } else {
      return isReTrain();
    }
  }

  /**
   * It may be desirable to have the metric trained only once, and use the same
   * parameters for all other training sessions of the learning algorithm using
   * the metric. This can be controlled through this boolean. Setting
   * <tt>true</tt> if this metric will indicate a need to be retrained once it
   * has been trained once. This will mean {@link #needsTraining() } will always
   * return true. <tt>false</tt> means the metric will not indicate a need to be
   * retrained once it has been trained once.
   *
   * @param reTrain
   *          <tt>true</tt> to make the metric always request retraining,
   *          <tt>false</tt> so it will not.
   */
  public void setReTrain(final boolean reTrain) {
    this.reTrain = reTrain;
  }

  @Override
  public boolean supportsAcceleration() {
    return false;
  }

  @Override
  public boolean supportsClassificationTraining() {
    return true;
  }

  @Override
  public boolean supportsRegressionTraining() {
    return true;
  }

  @Override
  public String toString() {
    return "Mahalanobis Distance";
  }

  @Override
  public void train(final ClassificationDataSet dataSet) {
    train((DataSet) dataSet);
  }

  @Override
  public void train(final ClassificationDataSet dataSet, final ExecutorService threadpool) {
    train((DataSet) dataSet, threadpool);
  }

  @Override
  public void train(final DataSet dataSet) {
    train(dataSet, null);
  }

  @Override
  public void train(final DataSet dataSet, final ExecutorService threadpool) {
    train(dataSet.getDataVectors(), threadpool);
  }

  @Override
  public <V extends Vec> void train(final List<V> dataSet) {
    train(dataSet, null);
  }

  @Override
  public <V extends Vec> void train(final List<V> dataSet, final ExecutorService threadpool) {
    final Vec mean = MatrixStatistics.meanVector(dataSet);
    final Matrix covariance = MatrixStatistics.covarianceMatrix(mean, dataSet);
    LUPDecomposition lup;
    SingularValueDecomposition svd;
    if (threadpool != null) {
      lup = new LUPDecomposition(covariance.clone(), threadpool);
    } else {
      lup = new LUPDecomposition(covariance.clone());
    }
    final double det = lup.det();
    if (Double.isNaN(det) || Double.isInfinite(det) || Math.abs(det) <= 1e-13) // Bad
                                                                               // problem,
                                                                               // use
                                                                               // the
                                                                               // SVD
                                                                               // instead
    {
      lup = null;
      svd = new SingularValueDecomposition(covariance);
      S = svd.getPseudoInverse();
    } else if (threadpool != null) {
      S = lup.solve(Matrix.eye(covariance.cols()), threadpool);
    } else {
      S = lup.solve(Matrix.eye(covariance.cols()));
    }
  }

  @Override
  public void train(final RegressionDataSet dataSet) {
    train((DataSet) dataSet);
  }

  @Override
  public void train(final RegressionDataSet dataSet, final ExecutorService threadpool) {
    train((DataSet) dataSet, threadpool);
  }

}
