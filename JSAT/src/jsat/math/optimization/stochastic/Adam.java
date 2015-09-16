package jsat.math.optimization.stochastic;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.ScaledVector;
import jsat.linear.Vec;

/**
 * Adam is inspired by {@link RMSProp} and {@link AdaGrad}, where the former can
 * be seen as a special case of Adam. Adam has been shown to work well in
 * training neural networks, and still converges well with sparse gradients.<br>
 * NOTE: that while it will converge, Adam dose not support sparse updates. So
 * runtime when in highly sparse environments will be hampered. <br>
 * <br>
 * See: Kingma, D. P.,&amp;Ba, J. L. (2015). <i>Adam: A Method for Stochastic
 * Optimization</i>. In ICLR.
 *
 * @author Edward Raff
 */
public class Adam implements GradientUpdater {

  private static final long serialVersionUID = 5352504067435579553L;
  public static final double DEFAULT_ALPHA = 0.0002;
  public static final double DEFAULT_BETA_1 = 0.1;

  public static final double DEFAULT_BETA_2 = 0.001;

  public static final double DEFAULT_EPS = 1e-8;
  public static final double DEFAULT_LAMBDA = 1e-8;
  // internal state
  /**
   * 1st moment vector
   */
  private Vec m;
  /**
   * 2nd moment vector
   */
  private Vec v;
  /**
   * time step
   */
  private long t;

  // parameters of the algo
  private final double alpha;
  private final double beta_1;

  private final double beta_2;
  private final double eps;
  private final double lambda;
  private double vBias;
  private double mBias;

  public Adam() {
    this(DEFAULT_ALPHA, DEFAULT_BETA_1, DEFAULT_BETA_2, DEFAULT_EPS, DEFAULT_LAMBDA);
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public Adam(final Adam toCopy) {
    alpha = toCopy.alpha;
    beta_1 = toCopy.beta_1;
    beta_2 = toCopy.beta_2;
    eps = toCopy.eps;
    lambda = toCopy.lambda;
    t = toCopy.t;
    mBias = toCopy.mBias;
    vBias = toCopy.vBias;

    if (toCopy.m != null) {
      m = toCopy.m.clone();
      v = toCopy.v.clone();
    }

  }

  public Adam(final double alpha, final double beta_1, final double beta_2, final double eps, final double lambda) {
    if (alpha <= 0 || Double.isInfinite(alpha) || Double.isNaN(alpha)) {
      throw new IllegalArgumentException("alpha must be a positive value, not " + alpha);
    }
    if (beta_1 <= 0 || beta_1 > 1 || Double.isInfinite(beta_1) || Double.isNaN(beta_1)) {
      throw new IllegalArgumentException("beta_1 must be in (0, 1], not " + beta_1);
    }
    if (beta_2 <= 0 || beta_2 > 1 || Double.isInfinite(beta_2) || Double.isNaN(beta_2)) {
      throw new IllegalArgumentException("beta_2 must be in (0, 1], not " + beta_2);
    }
    if (pow(1 - beta_1, 2) / sqrt(1 - beta_2) >= 1) {
      throw new IllegalArgumentException(
          "the required property (1-beta_1)^2 / sqrt(1-beta_2) < 1, is not held by beta_1=" + beta_1 + " and beta_2="
              + beta_2);
    }
    if (lambda <= 0 || lambda >= 1 || Double.isInfinite(lambda) || Double.isNaN(lambda)) {
      throw new IllegalArgumentException("lambda must be in (0, 1), not " + lambda);
    }
    this.alpha = alpha;
    this.beta_1 = beta_1;
    this.beta_2 = beta_2;
    this.eps = eps;
    this.lambda = lambda;
  }

  @Override
  public Adam clone() {
    return new Adam(this);
  }

  @Override
  public void setup(final int d) {
    t = 0;
    m = new ScaledVector(new DenseVector(d));
    v = new ScaledVector(new DenseVector(d));
    vBias = mBias = 0;
  }

  @Override
  public void update(final Vec x, final Vec grad, final double eta) {
    update(x, grad, eta, 0, 0);
  }

  @Override
  public double update(final Vec x, final Vec grad, final double eta, final double bias, final double biasGrad) {
    t++;
    // (Decay the first moment running average coefficient
    final double beta_1t = 1 - (1 - beta_1) * pow(lambda, t - 1);
    // (Get gradients w.r.t. stochastic objective at timestep t)
    // grad is already that value
    // (Update biased first moment estimate)
    m.mutableMultiply(1 - beta_1t);
    m.mutableAdd(beta_1t, grad);
    mBias = 1 - beta_1t + beta_1t * biasGrad;
    // (Update biased second raw moment estimate)
    v.mutableMultiply(1 - beta_2);
    vBias = (1 - beta_2) * vBias + beta_2 * biasGrad * biasGrad;
    for (final IndexValue iv : grad) {
      final double g_i = iv.getValue();
      v.increment(iv.getIndex(), beta_2 * (g_i * g_i));
    }
    /*
     * "Note that the efficiency of algorithm 1 can, at the expense of clarity,
     * be improved upon by changing the order of computation, e.g. by replacing
     * the last three lines in the loop with the following line:" θ_t = θ_{t−1}
     * −[α ·√(1−(1−β_2)^t) · (1−(1−β_1)^t)−1] ·m_t/√v_t
     */
    final double cnst = eta * alpha * sqrt(1 - pow(1 - beta_2, t)) / (1 - pow(1 - beta_1, t));

    // while the algorithm may converge well with sparse data, m and v are
    // likely to all be non-zero after observing lots of data.
    for (int i = 0; i < m.length(); i++) {
      x.increment(i, -cnst * m.get(i) / (sqrt(v.get(i)) + eps));
    }
    return cnst * mBias / (sqrt(vBias) + eps);
  }

}
