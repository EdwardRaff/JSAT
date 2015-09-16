package jsat.clustering.evaluation.intra;

import java.util.List;

import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.MatrixStatistics;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * Evaluates a cluster's validity by computing the mean distance of each point
 * in the cluster from the cluster's centroid.
 *
 * @author Edward Raff
 */
public class MeanCentroidDistance implements IntraClusterEvaluation {

  private final DistanceMetric dm;

  /**
   * Creates a new MeanCentroidDistance using the {@link EuclideanDistance}
   */
  public MeanCentroidDistance() {
    this(new EuclideanDistance());
  }

  /**
   * Creates a new MeanCentroidDistance.
   *
   * @param dm
   *          the metric to measure the distance between two points by
   */
  public MeanCentroidDistance(final DistanceMetric dm) {
    this.dm = dm;
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public MeanCentroidDistance(final MeanCentroidDistance toCopy) {
    this(toCopy.dm.clone());
  }

  @Override
  public MeanCentroidDistance clone() {
    return new MeanCentroidDistance(this);
  }

  @Override
  public double evaluate(final int[] designations, final DataSet dataSet, final int clusterID) {
    final Vec mean = new DenseVector(dataSet.getNumNumericalVars());

    int clusterSize = 0;
    for (int i = 0; i < dataSet.getSampleSize(); i++) {
      if (designations[i] == clusterID) {
        clusterSize++;
        mean.mutableAdd(dataSet.getDataPoint(i).getNumericalValues());
      }
    }
    mean.mutableDivide(clusterSize);

    double dists = 0.0;

    for (int i = 0; i < dataSet.getSampleSize(); i++) {
      if (designations[i] == clusterID) {
        dists += dm.dist(dataSet.getDataPoint(i).getNumericalValues(), mean);
      }
    }

    return dists / dataSet.getSampleSize();
  }

  @Override
  public double evaluate(final List<DataPoint> dataPoints) {
    final Vec mean = MatrixStatistics.meanVector(new SimpleDataSet(dataPoints));

    double dists = 0.0;
    for (final DataPoint dp : dataPoints) {
      dists += dm.dist(dp.getNumericalValues(), mean);
    }

    return dists / dataPoints.size();
  }

}
