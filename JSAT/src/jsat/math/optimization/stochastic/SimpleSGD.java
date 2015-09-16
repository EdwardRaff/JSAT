package jsat.math.optimization.stochastic;

import jsat.linear.Vec;

/**
 * Performs unaltered Stochastic Gradient Decent updates computing <i>x = x-
 * &eta; grad</i><br>
 * <br>
 * Because the SimpleSGD requires no internal state, it is not necessary to call
 * {@link #setup(int) }.
 *
 * @author Edward Raff
 */
public class SimpleSGD implements GradientUpdater {

  private static final long serialVersionUID = 4022442467298319553L;

  /**
   * Creates a new SGD updater
   */
  public SimpleSGD() {
  }

  @Override
  public SimpleSGD clone() {
    return new SimpleSGD();
  }

  @Override
  public void setup(final int d) {
    // no setup to be done
  }

  @Override
  public void update(final Vec x, final Vec grad, final double eta) {
    x.mutableSubtract(eta, grad);
  }

  @Override
  public double update(final Vec x, final Vec grad, final double eta, final double bias, final double biasGrad) {
    x.mutableSubtract(eta, grad);
    return eta * biasGrad;
  }

}
