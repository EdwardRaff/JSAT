package jsat.distributions.multivariate;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * Common class for implementing a multivariate distribution. A number of
 * methods are pre implemented, building off of the implementation of the
 * remaining methods. <br>
 * Note: the default implementation for the multithreaded methods calls the non
 * threaded version of the method. The exception to this is the
 * {@link #setUsingData(jsat.DataSet, java.util.concurrent.ExecutorService) }
 * method, which calls
 * {@link #setUsingData(java.util.List, java.util.concurrent.ExecutorService) }
 *
 * @author Edward Raff
 */
public abstract class MultivariateDistributionSkeleton implements MultivariateDistribution {

  private static final long serialVersionUID = 4080753806798149915L;

  @Override
  abstract public MultivariateDistribution clone();

  @Override
  public double logPdf(final double... x) {
    return logPdf(DenseVector.toDenseVec(x));
  }

  @Override
  public double logPdf(final Vec x) {
    final double logPDF = Math.log(pdf(x));
    if (Double.isInfinite(logPDF) && logPDF < 0) {// log(0) == -Infinty
      return -Double.MAX_VALUE;
    }
    return logPDF;
  }

  @Override
  public double pdf(final double... x) {
    return pdf(DenseVector.toDenseVec(x));
  }

  @Override
  public boolean setUsingData(final DataSet dataSet) {
    return setUsingDataList(dataSet.getDataPoints());
  }

  @Override
  public boolean setUsingData(final DataSet dataSet, final ExecutorService threadpool) {
    return setUsingDataList(dataSet.getDataPoints(), threadpool);
  }

  @Override
  public <V extends Vec> boolean setUsingData(final List<V> dataSet, final ExecutorService threadpool) {
    return setUsingData(dataSet);
  }

  @Override
  public boolean setUsingDataList(final List<DataPoint> dataPoints, final ExecutorService threadpool) {
    return setUsingDataList(dataPoints);
  }
}
