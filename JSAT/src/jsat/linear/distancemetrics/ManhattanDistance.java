package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;

import jsat.linear.IndexValue;
import jsat.linear.Vec;

/**
 * Manhattan Distance is the L<sub>1</sub> norm.
 *
 * @author Edward Raff
 */
public class ManhattanDistance implements DenseSparseMetric {

  private static final long serialVersionUID = 3028834823742743351L;

  @Override
  public ManhattanDistance clone() {
    return new ManhattanDistance();
  }

  @Override
  public double dist(final double summaryConst, final Vec main, final Vec target) {
    if (!target.isSparse()) {
      return dist(main, target);
    }
    /**
     * Summary contains the differences to the zero vec, only a few of the
     * indices are actually non zero - we correct those values
     */
    double takeOut = 0.0;
    for (final IndexValue iv : target) {
      final int i = iv.getIndex();
      final double mainVal = main.get(i);
      takeOut += mainVal - Math.abs(mainVal - iv.getValue());
    }
    return summaryConst - takeOut;
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
    return a.pNormDist(1, b);
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
  public double getVectorConstant(final Vec vec) {
    return vec.pNorm(1);
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

  @Override
  public boolean supportsAcceleration() {
    return false;
  }

  @Override
  public String toString() {
    return "Manhattan Distance";
  }

}
