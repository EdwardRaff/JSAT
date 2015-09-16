package jsat.distributions.kernels;

import java.util.List;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;

/**
 * This abstract class provides the means of implementing a Kernel based off
 * some {@link DistanceMetric}. This will pre-implement most of the methods of
 * the KernelTrick interface, including using the distance acceleration of the
 * metric (if supported) when appropriate.
 *
 * @author Edward Raff
 */
public abstract class DistanceMetricBasedKernel implements KernelTrick {

  private static final long serialVersionUID = 8395066824809874527L;
  /**
   * the distance metric to use for the Kernel
   */
  @ParameterHolder
  protected DistanceMetric d;

  /**
   * Creates a new distance based kerenel
   *
   * @param d
   *          the distance metric to use
   */
  public DistanceMetricBasedKernel(final DistanceMetric d) {
    this.d = d;
  }

  @Override
  public void addToCache(final Vec newVec, final List<Double> cache) {
    cache.addAll(d.getQueryInfo(newVec));
  }

  @Override
  abstract public KernelTrick clone();

  @Override
  public double evalSum(final List<? extends Vec> finalSet, final List<Double> cache, final double[] alpha, final Vec y,
      final int start, final int end) {
    return evalSum(finalSet, cache, alpha, y, d.getQueryInfo(y), start, end);
  }

  @Override
  public double evalSum(final List<? extends Vec> finalSet, final List<Double> cache, final double[] alpha, final Vec y,
      final List<Double> qi, final int start, final int end) {
    double sum = 0;

    for (int i = start; i < end; i++) {
      if (alpha[i] != 0) {
        sum += alpha[i] * eval(i, y, qi, finalSet, cache);
      }
    }

    return sum;
  }

  @Override
  public List<Double> getAccelerationCache(final List<? extends Vec> trainingSet) {
    return d.getAccelerationCache(trainingSet);
  }

  @Override
  public Parameter getParameter(final String paramName) {
    return Parameter.toParameterMap(getParameters()).get(paramName);
  }

  @Override
  public List<Parameter> getParameters() {
    return Parameter.getParamsFromMethods(this);
  }

  @Override
  public List<Double> getQueryInfo(final Vec q) {
    return d.getQueryInfo(q);
  }

  @Override
  public boolean supportsAcceleration() {
    return d.supportsAcceleration();
  }

}
