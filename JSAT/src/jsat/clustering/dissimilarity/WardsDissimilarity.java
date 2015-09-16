package jsat.clustering.dissimilarity;

import jsat.linear.distancemetrics.SquaredEuclideanDistance;

/**
 * An implementation of Ward's method for hierarchical clustering. This method
 * merges clusters based on the minimum total variance of the resulting
 * clusters.
 *
 * @author Edward Raff
 */
public class WardsDissimilarity extends LanceWilliamsDissimilarity {

  public WardsDissimilarity() {
    super(new SquaredEuclideanDistance());
  }

  @Override
  protected double aConst(final boolean iFlag, final int ni, final int nj, final int nk) {
    final double totalPoints = ni + nj + nk;
    if (iFlag) {
      return (ni + nk) / totalPoints;
    } else {
      return (nj + nk) / totalPoints;
    }
  }

  @Override
  protected double bConst(final int ni, final int nj, final int nk) {
    final double totalPoints = ni + nj + nk;
    return -nk / totalPoints;
  }

  @Override
  protected double cConst(final int ni, final int nj, final int nk) {
    return 0;
  }

  @Override
  public WardsDissimilarity clone() {
    return new WardsDissimilarity();
  }

}
