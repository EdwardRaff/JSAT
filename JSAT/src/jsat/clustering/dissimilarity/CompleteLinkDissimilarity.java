package jsat.clustering.dissimilarity;

import java.util.List;
import java.util.Set;

import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * Measures the dissimilarity of two clusters by returning the value of the
 * maximal dissimilarity of any two pairs of data points where one is from each
 * cluster.
 *
 * @author Edward Raff
 */
public class CompleteLinkDissimilarity extends DistanceMetricDissimilarity implements UpdatableClusterDissimilarity {

  /**
   * Creates a new CompleteLinkDissimilarity using the {@link EuclideanDistance}
   */
  public CompleteLinkDissimilarity() {
    this(new EuclideanDistance());
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public CompleteLinkDissimilarity(final CompleteLinkDissimilarity toCopy) {
    this(toCopy.dm.clone());
  }

  /**
   * Creates a new CompleteLinkDissimilarity
   *
   * @param dm
   *          the distance metric to use between individual points
   */
  public CompleteLinkDissimilarity(final DistanceMetric dm) {
    super(dm);
  }

  @Override
  public CompleteLinkDissimilarity clone() {
    return new CompleteLinkDissimilarity(this);
  }

  @Override
  public double dissimilarity(final int i, final int ni, final int j, final int nj, final double[][] distanceMatrix) {
    return getDistance(distanceMatrix, i, j);
  }

  @Override
  public double dissimilarity(final int i, final int ni, final int j, final int nj, final int k, final int nk,
      final double[][] distanceMatrix) {
    return Math.max(getDistance(distanceMatrix, i, k), getDistance(distanceMatrix, j, k));
  }

  @Override
  public double dissimilarity(final List<DataPoint> a, final List<DataPoint> b) {
    double maxDiss = Double.MIN_VALUE;

    double tmpDist;
    for (final DataPoint ai : a) {
      for (final DataPoint bi : b) {
        if ((tmpDist = distance(ai, bi)) > maxDiss) {
          maxDiss = tmpDist;
        }
      }
    }

    return maxDiss;
  }

  @Override
  public double dissimilarity(final Set<Integer> a, final Set<Integer> b, final double[][] distanceMatrix) {
    double maxDiss = Double.MIN_VALUE;

    for (final int ai : a) {
      for (final int bi : b) {
        if (getDistance(distanceMatrix, ai, bi) > maxDiss) {
          maxDiss = getDistance(distanceMatrix, ai, bi);
        }
      }
    }

    return maxDiss;
  }

}
