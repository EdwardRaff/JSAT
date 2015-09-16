package jsat.math.optimization.stochastic;

import jsat.linear.DenseVector;
import jsat.linear.ScaledVector;
import jsat.linear.Vec;

/**
 * Performs unaltered Stochastic Gradient Decent updates using either standard
 * or Nestrov momentum. <br>
 * <br>
 * See:<br>
 * <ul>
 * <li>Bengio, Y., Boulanger-Lewandowski, N.,&amp;Pascanu, R. (2013).
 * <i>Advances in optimizing recurrent networks</i>. In 2013 IEEE International
 * Conference on Acoustics, Speech and Signal Processing (pp. 8624–8628). IEEE.
 * doi:10.1109/ICASSP.2013.6639349</li>
 * <li>Sutskever, I., Martens, J., Dahl, G.,&amp;Hinton, G. (2013). <i>On the
 * importance of initialization and momentum in deep learning</i>. JMLR
 * W&amp;CP, 28, 1139–1147.</li>
 * </ul>
 *
 * @author Edward Raff
 */
public class SGDMomentum implements GradientUpdater {

  private static final long serialVersionUID = -3837883539010356899L;

  private double momentum;
  private boolean nestrov;
  private Vec velocity;
  private double biasVelocity;

  /**
   * Creates a new SGD with Nestrov Momentum learner
   *
   * @param momentum
   *          the amount of momentum to use
   */
  public SGDMomentum(final double momentum) {
    this(momentum, true);
  }

  /**
   * Creates a new SGD with Momentum learner
   *
   * @param momentum
   *          the amount of momentum to use
   * @param nestrov
   *          {@code true} to use Nestrov momentum, {@code false} for standard.
   */
  public SGDMomentum(final double momentum, final boolean nestrov) {
    setMomentum(momentum);
    this.nestrov = nestrov;
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public SGDMomentum(final SGDMomentum toCopy) {
    momentum = toCopy.momentum;
    if (toCopy.velocity != null) {
      velocity = toCopy.velocity.clone();
    }
    biasVelocity = toCopy.biasVelocity;
  }

  @Override
  public SGDMomentum clone() {
    return new SGDMomentum(this);
  }

  /**
   *
   * @return the momentum buildup term
   */
  public double getMomentum() {
    return momentum;
  }

  /**
   * Sets the momentum for accumulating gradients.
   *
   * @param momentum
   *          the momentum buildup term in (0, 1)
   */
  public void setMomentum(final double momentum) {
    if (momentum <= 0 || momentum >= 1 || Double.isNaN(momentum)) {
      throw new IllegalArgumentException("Momentum must be in (0,1) not " + momentum);
    }
    this.momentum = momentum;
  }

  @Override
  public void setup(final int d) {
    velocity = new ScaledVector(new DenseVector(d));
    biasVelocity = 0;
  }

  @Override
  public void update(final Vec x, final Vec grad, final double eta) {
    update(x, grad, eta, 0.0, 0.0);
  }

  @Override
  public double update(final Vec x, final Vec grad, final double eta, final double bias, final double biasGrad) {
    double biasUpdate;
    if (nestrov) {
      // update
      x.mutableAdd(momentum * momentum, velocity);
      x.mutableSubtract((1 + momentum) * eta, grad);
      biasUpdate = -momentum * momentum * biasVelocity + (1 + momentum) * eta * biasGrad;
    } else// clasic momentum
    {
      // update
      x.mutableAdd(momentum, velocity);
      x.mutableSubtract(eta, grad);
      biasUpdate = -momentum * biasVelocity + eta * biasGrad;
    }

    // velocity
    velocity.mutableMultiply(momentum);
    velocity.mutableSubtract(eta, grad);
    biasVelocity = biasVelocity * momentum - eta * biasGrad;

    return biasUpdate;
  }

}
