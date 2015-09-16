package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;

import jsat.linear.IndexValue;
import jsat.linear.Vec;

/**
 * Minkowski Distance is the L<sub>p</sub> norm.
 *
 * @author Edward Raff
 */
public class MinkowskiDistance implements DenseSparseMetric {

  private static final long serialVersionUID = 8976696315441171045L;
  private double p;

  /**
   *
   * @param p
   *          the norm to use as the distance
   */
  public MinkowskiDistance(final double p) {
    if (p <= 0 || Double.isNaN(p)) {
      throw new ArithmeticException("The pNorm exists only for p > 0");
    } else if (Double.isInfinite(p)) {
      throw new ArithmeticException("Infinity norm is a special case, use ChebyshevDistance for infinity norm");
    }

    setP(p);
  }

  @Override
  public MinkowskiDistance clone() {
    return new MinkowskiDistance(p);
  }

  @Override
  public double dist(final double summaryConst, final Vec main, final Vec target) {
    if (!target.isSparse()) {
      return dist(main, target);
    }
    /**
     * Summary contains the differences^p to the zero vec, only a few of the
     * indices are actually non zero - we correct those values
     */
    double addBack = 0.0;
    double takeOut = 0.0;
    for (final IndexValue iv : target) {
      final int i = iv.getIndex();
      final double mainVal = main.get(i);
      takeOut += Math.pow(mainVal, p);
      addBack += Math.pow(mainVal - iv.getValue(), p);
    }
    return Math.pow(summaryConst - takeOut + addBack, 1 / p);
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
    return a.pNormDist(p, b);
  }

  @Override
  public List<Double> getAccelerationCache(final List<? extends Vec> vecs) {
    return null;
  }

  @Override
  public List<Double> getAccelerationCache(final List<? extends Vec> vecs, final ExecutorService threadpool) {
    return null;
  }

  /**
   *
   * @return the norm to use for this metric.
   */
  public double getP() {
    return p;
  }

  @Override
  public List<Double> getQueryInfo(final Vec q) {
    return null;
  }

  @Override
  public double getVectorConstant(final Vec vec) {
    return Math.pow(vec.pNorm(p), p);
  }

  @Override
  public boolean isIndiscemible() {
    return true;
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

  /**
   *
   * @param p
   *          the norm to use for this metric
   */
  public void setP(final double p) {
    if (p <= 0 || Double.isNaN(p) || Double.isInfinite(p)) {
      throw new IllegalArgumentException("p must be a positive value, not " + p);
    }
    this.p = p;
  }

  @Override
  public boolean supportsAcceleration() {
    return false;
  }

  @Override
  public String toString() {
    return "Minkowski Distance (p=" + p + ")";
  }

}
