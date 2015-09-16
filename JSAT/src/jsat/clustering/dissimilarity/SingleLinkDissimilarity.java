package jsat.clustering.dissimilarity;

import java.util.List;
import java.util.Set;
import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * Measures the dissimilarity of two clusters by returning the minimum
 * dissimilarity between the two closest data points from the clusters, ie: the
 * minimum distance needed to link the two clusters.
 *
 * @author Edward Raff
 */
public class SingleLinkDissimilarity extends DistanceMetricDissimilarity implements UpdatableClusterDissimilarity {

  /**
   * Creates a new SingleLinkDissimilarity using the {@link EuclideanDistance}
   */
  public SingleLinkDissimilarity() {
    this(new EuclideanDistance());
  }

  /**
   * Creates a new SingleLinkDissimilarity
   *
   * @param dm
   *          the distance metric to use between individual points
   */
  public SingleLinkDissimilarity(final DistanceMetric dm) {
    super(dm);
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public SingleLinkDissimilarity(final SingleLinkDissimilarity toCopy) {
    this(toCopy.dm.clone());
  }

  @Override
  public SingleLinkDissimilarity clone() {
    return new SingleLinkDissimilarity(this);
  }

  @Override
  public double dissimilarity(final int i, final int ni, final int j, final int nj, final double[][] distanceMatrix) {
    return getDistance(distanceMatrix, i, j);
  }

  @Override
  public double dissimilarity(final int i, final int ni, final int j, final int nj, final int k, final int nk,
      final double[][] distanceMatrix) {
    return Math.min(getDistance(distanceMatrix, i, k), getDistance(distanceMatrix, j, k));
  }

  @Override
  public double dissimilarity(final List<DataPoint> a, final List<DataPoint> b) {
    double minDiss = Double.MAX_VALUE;

    double tmpDist;
    for (final DataPoint ai : a) {
      for (final DataPoint bi : b) {
        if ((tmpDist = distance(ai, bi)) < minDiss) {
          minDiss = tmpDist;
        }
      }
    }

    return minDiss;
  }

  @Override
  public double dissimilarity(final Set<Integer> a, final Set<Integer> b, final double[][] distanceMatrix) {
    double minDiss = Double.MAX_VALUE;

    for (final int ai : a) {
      for (final int bi : b) {
        if (getDistance(distanceMatrix, ai, bi) < minDiss) {
          minDiss = getDistance(distanceMatrix, ai, bi);
        }
      }
    }

    return minDiss;
  }

}
