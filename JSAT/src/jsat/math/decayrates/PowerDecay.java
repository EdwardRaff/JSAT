package jsat.math.decayrates;

import java.util.List;

import jsat.math.FastMath;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;

/**
 *
 * Decays an input by power of the amount of time that has occurred, the max
 * time being irrelevant. More specifically as &eta; ({@link #setTau(double)
 * &tau;} + time)<sup>-{@link #setAlpha(double) &alpha;}</sup><br>
 * <br>
 * {@link InverseDecay} is a special case of this decay when &alpha;=1. <br>
 * {@link NoDecay} is a special case of this decay when &alpha; = 0 <br>
 *
 * @author Edward Raff
 */
public class PowerDecay implements DecayRate, Parameterized {

  private static final long serialVersionUID = 6075066391550611699L;
  private double tau;
  private double alpha;

  /**
   * Creates a new Power Decay rate
   */
  public PowerDecay() {
    this(10, 0.5);
  }

  /**
   * Creates a new Power decay rate
   *
   * @param tau
   *          the initial time offset
   * @param alpha
   *          the time scaling
   */
  public PowerDecay(final double tau, final double alpha) {
    setTau(tau);
    setAlpha(alpha);
  }

  @Override
  public DecayRate clone() {
    return new PowerDecay(tau, alpha);
  }

  /**
   * Returns the scaling parameter
   *
   * @return the scaling parameter
   */
  public double getAlpha() {
    return alpha;
  }

  @Override
  public Parameter getParameter(final String paramName) {
    return Parameter.toParameterMap(getParameters()).get(paramName);
  }

  @Override
  public List<Parameter> getParameters() {
    return Parameter.getParamsFromMethods(this);
  }

  /**
   * Returns the early rate dampening parameter
   *
   * @return the early rate dampening parameter
   */
  public double getTau() {
    return tau;
  }

  @Override
  public double rate(final double time, final double initial) {
    return initial * FastMath.pow(tau + time, -alpha);
  }

  @Override
  public double rate(final double time, final double maxTime, final double initial) {
    return rate(time, initial);
  }

  /**
   * Controls the scaling via exponentiation, increasing &alpha; increases the
   * rate at which the rate decays. As &alpha; goes to zero, the decay rate goes
   * toward zero (meaning the value returned becomes constant),
   *
   * @param alpha
   *          the scaling parameter in [0, &infin;), but should generally be
   *          kept in (0, 1).
   */
  public void setAlpha(final double alpha) {
    if (alpha < 0 || Double.isInfinite(alpha) || Double.isNaN(alpha)) {
      throw new IllegalArgumentException("alpha must be a non negative constant, not " + alpha);
    }
    this.alpha = alpha;
  }

  /**
   * Controls the rate early in time, but has a decreasing impact on the rate
   * returned as time goes forward. Larger values of &tau; dampen the initial
   * rates returned, while lower values let the initial rates start higher.
   *
   * @param tau
   *          the early rate dampening parameter
   */
  public void setTau(final double tau) {
    if (tau <= 0 || Double.isInfinite(tau) || Double.isNaN(tau)) {
      throw new IllegalArgumentException("tau must be a positive constant, not " + tau);
    }
    this.tau = tau;
  }

  @Override
  public String toString() {
    return "Power Decay";
  }
}
