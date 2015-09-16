package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;

import jsat.linear.Vec;

/**
 * Chebyshev Distance is the L<sub>&#8734;</sub> norm.
 *
 * @author Edward Raff
 */
public class ChebyshevDistance implements DistanceMetric {

  private static final long serialVersionUID = 2528153647402824790L;

  @Override
  public ChebyshevDistance clone() {
    return new ChebyshevDistance();
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
    if (a.length() != b.length()) {
      throw new ArithmeticException("Vectors must have the same length");
    }
    double max = 0;

    for (int i = 0; i < a.length(); i++) {
      max = Math.max(max, Math.abs(a.get(i) - b.get(i)));
    }

    return max;
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
    return "Chebyshev Distance";
  }
}
