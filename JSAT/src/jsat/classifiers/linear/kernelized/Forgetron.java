package jsat.classifiers.linear.kernelized;

import static java.lang.Math.abs;
import static java.lang.Math.log;
import static java.lang.Math.min;
import static java.lang.Math.pow;
import static java.lang.Math.signum;
import static java.lang.Math.sqrt;

import java.util.Arrays;
import java.util.List;

import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.classifiers.neuralnetwork.Perceptron;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.FailedToFitException;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;

/**
 * Implementation of the first two Forgetron algorithms. The Forgetron is a
 * kernelized version of the {@link Perceptron} that maintains a fixed sized
 * buffer of data instances that it uses to form its decision boundary. <br>
 * <br>
 * See:<br>
 * Dekel, O., Shalev-Shwartz, S.,&amp;Singer, Y. (2008). <i>The Forgetron: A
 * kernel-based perceptron on a fixed budget</i>. SIAM Journal on Computing,
 * 37(5), 1342–1372.
 *
 * @author Edward Raff
 */
public class Forgetron extends BaseUpdateableClassifier implements BinaryScoreClassifier, Parameterized {

  private static final long serialVersionUID = -2631315082407427077L;

  @ParameterHolder
  private KernelTrick K;
  private Vec[] I;
  /**
   * Stores the label times the weight. Getting the true weight is an abs
   * operation. Getting the true label is a signum operation.
   */
  private double[] s;
  private int size;
  /**
   * Will always point to current insert position. Either empty, or the last
   * value ever inserted
   */
  private int curPos;
  private int budget;
  private double U;
  private double Bconst;
  private double Q, M;

  private boolean selfTuned = true;

  /**
   * Copy constructor
   *
   * @param toClone
   *          the forgetron to clone
   */
  protected Forgetron(final Forgetron toClone) {
    K = toClone.K.clone();
    budget = toClone.budget;
    U = toClone.U;
    Bconst = toClone.Bconst;
    Q = toClone.Q;
    M = toClone.M;
    curPos = toClone.curPos;
    size = toClone.size;
    if (toClone.I != null) {
      I = new Vec[toClone.I.length];
      for (int i = 0; i < toClone.I.length; i++) {
        if (toClone.I[i] != null) {
          I[i] = toClone.I[i].clone();
        }
      }
    }
    if (toClone.s != null) {
      s = Arrays.copyOf(toClone.s, toClone.s.length);
    }
  }

  /**
   * Creates a new Forgetron
   *
   * @param kernel
   *          the kernel function to use
   * @param budget
   *          the maximum number of data points to use
   */
  public Forgetron(final KernelTrick kernel, final int budget) {
    K = kernel;
    setBudget(budget);
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    final CategoricalResults cr = new CategoricalResults(2);
    final int winner = (int) ((signum(getScore(data)) + 1) / 2);
    cr.setProb(winner, 1);
    return cr;
  }

  private double classify(final Vec x) {
    double r = 0;
    for (int i = 0; i < size; i++) {
      r += s[i] * K.eval(I[i], x);
    }
    return r;
  }

  @Override
  public Forgetron clone() {
    return new Forgetron(this);
  }

  /**
   * Returns the current budget
   *
   * @return the current budget
   */
  public int getBudget() {
    return budget;
  }

  /**
   * Returns the current kernel trick
   *
   * @return the current kernel trick
   */
  public KernelTrick getKernelTrick() {
    return K;
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
  public double getScore(final DataPoint dp) {
    return classify(dp.getNumericalValues());
  }

  /**
   *
   * @return {@code true} if the self-tuned variant is used, {@code false}
   *         otherwise.
   */
  public boolean isSelfTuned() {
    return selfTuned;
  }

  /**
   * See equation 15
   *
   * @param lambda
   * @param mu
   * @return the update for equation 15
   */
  private double psi(final double lambda, final double mu) {
    return lambda * lambda + 2 * lambda - 2 * lambda * mu;
  }

  /**
   * Sets the new budget, which is the maximum number of data points the
   * Forgetron can use to form its decision boundary.
   *
   * @param budget
   *          the maximum number of data points to use
   */
  public void setBudget(final int budget) {
    this.budget = budget;
    final double B = budget;
    U = sqrt((B + 1) / log(B + 1)) / 4;
    Bconst = pow(B + 1, 1.0 / (2 * B + 2));
  }

  /**
   * Sets the kernel trick to use
   *
   * @param K
   *          the kernel trick to use
   */
  public void setKernelTrick(final KernelTrick K) {
    this.K = K;
  }

  /**
   * Sets whether or not the self-tuned variant of the Forgetron is used, the
   * default is {@code true}
   *
   * @param selfTurned
   *          {@code true} to use the self-tuned variance, {@code false}
   *          otherwise.
   */
  public void setSelfTurned(final boolean selfTurned) {
    selfTuned = selfTurned;
  }

  @Override
  public void setUp(final CategoricalData[] categoricalAttributes, final int numericAttributes,
      final CategoricalData predicting) {
    if (predicting.getNumOfCategories() != 2) {
      throw new FailedToFitException("Forgetron only supports binary classification");
    } else if (numericAttributes == 0) {
      throw new FailedToFitException("Forgetron requires numeric attributes");
    }
    I = new Vec[budget];
    s = new double[budget];
    Q = M = 0;
    size = 0;
    curPos = 0;
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void update(final DataPoint dataPoint, final int targetClass) {
    final Vec x = dataPoint.getNumericalValues();

    final double f_t = classify(x);
    final double y_t = targetClass * 2 - 1;

    if (y_t * f_t > 0) {
      // its all cool bro
    } else// not cool bro (error)
    {
      M++;
      if (selfTuned) {
        if (size + 1 <= budget) // in budget, we can add safly
        {
          size++;
          I[curPos] = x;
          s[curPos] = y_t;
        } else// over budget, remove oldest
        {
          final int r = curPos;

          // f'_t equation (27)
          final double fp_t = classify(I[r]) + y_t * K.eval(x, I[r]);

          // equations (44)
          final double s_r = abs(s[r]);
          final double y_r = signum(s[r]);
          final double a = s_r * s_r - 2 * y_r * s_r * fp_t;
          final double b = 2 * s_r;
          final double c = Q - 15.0 / 32.0 * M;
          final double d = b * b - 4 * a * c;

          // equations (43)
          double phi_t;
          if (a > 0 || a < 0 && d > 0 && (-b - sqrt(d)) / (2 * a) > 1) {
            phi_t = min(1, (-b + sqrt(d)) / (2 * a));
          } else if (a == 0) {
            phi_t = min(1, -c / b);
          } else {
            phi_t = 1;
          }

          final double fpp_t_r = phi_t * fp_t;
          Q += psi(s_r, y_r * fpp_t_r);

          I[curPos] = x;
          s[curPos] = y_t;
          if (phi_t != 1) {
            for (int i = 0; i < s.length; i++) {
              s[i] *= phi_t;
            }
          }

        }
      } else// normal version
      {

        double ff = 1;// for the added term that makes us remove one.
        if (size > 0) {
          for (int i = 0; i < size; i++) {
            ff += pow(s[i], 2) * K.eval(I[i], I[i]);
          }
        }
        final double fNorm = sqrt(ff);// obtained from after equation 2
        final double phi = min(Bconst, U / fNorm);

        I[curPos] = x;
        s[curPos] = y_t;
        if (size < budget) {
          size++;
        }
        for (int i = 0; i < size; i++) {
          s[i] *= phi;
        }
      }

      curPos = (curPos + 1) % I.length;

    }

  }

}
