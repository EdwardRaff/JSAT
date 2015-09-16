package jsat.distributions.kernels;

import java.util.List;

import jsat.linear.Vec;

/**
 * This provides a simple base implementation for the cache related methods in
 * Kernel Trick. By default they will all call
 * {@link #eval(jsat.linear.Vec, jsat.linear.Vec) } directly. For this reason
 * {@link #supportsAcceleration() } defaults to returning false. If the Kernel
 * supports cache acceleration,
 * {@link #evalSum(java.util.List, java.util.List, double[], jsat.linear.Vec, int, int) }
 * will make use of the acceleration. All other methods must be overloaded.
 *
 * @author Edward Raff
 */
public abstract class BaseKernelTrick implements KernelTrick {

  private static final long serialVersionUID = 7230585838672226751L;

  @Override
  public void addToCache(final Vec newVec, final List<Double> cache) {

  }

  @Override
  abstract public KernelTrick clone();

  @Override
  public double eval(final int a, final int b, final List<? extends Vec> trainingSet, final List<Double> cache) {
    return eval(trainingSet.get(a), trainingSet.get(b));
  }

  @Override
  public double eval(final int a, final Vec b, final List<Double> qi, final List<? extends Vec> vecs,
      final List<Double> cache) {
    return eval(vecs.get(a), b);
  }

  @Override
  public double evalSum(final List<? extends Vec> finalSet, final List<Double> cache, final double[] alpha, final Vec y,
      final int start, final int end) {
    return evalSum(finalSet, cache, alpha, y, getQueryInfo(y), start, end);
  }

  @Override
  public double evalSum(final List<? extends Vec> finalSet, final List<Double> cache, final double[] alpha, final Vec y,
      final List<Double> qi, final int start, final int end) {
    double sum = 0;

    for (int i = start; i < end; i++) {
      sum += alpha[i] * eval(i, y, qi, finalSet, cache);
    }

    return sum;
  }

  @Override
  public List<Double> getAccelerationCache(final List<? extends Vec> trainingSet) {
    return null;
  }

  @Override
  public List<Double> getQueryInfo(final Vec q) {
    return null;
  }

  @Override
  public boolean supportsAcceleration() {
    return false;
  }
}
