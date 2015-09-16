package jsat.classifiers.linear;

import java.util.Iterator;
import java.util.List;

import jsat.SingleWeightVectorModel;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.Classifier;
import jsat.exceptions.FailedToFitException;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.Regressor;

/**
 * This base class provides shared functionality and variables used by two
 * different training algorithms for L<sub>1</sub> regularized linear models.
 * Both squared and log loss are supported, making the results equivalent to
 * LASSO regression and Logistic regression respectively. <br>
 * <br>
 * These algorithms requires all feature values to be in the range [-1, 1]. The
 * implementation can do implicit rescaling, but rescaling may destroy sparsity.
 * If the data set is sparse and all values are zero or positive use the default
 * [0,1] rescaling to perform efficient rescaling that will not destroy
 * sparsity. <br>
 * <br>
 * Both algorithms are from: <br>
 * <a href="http://eprints.pascal-network.org/archive/00005418/">Shalev-Shwartz,
 * S.,&amp;Tewari, A. (2009). <i>Stochastic Methods for L<sub>1</sub>
 * -regularized Loss Minimization</i>. 26th International Conference on Machine
 * Learning (Vol. 12, pp. 929–936).</a>
 *
 * @author Edward Raff
 */
public abstract class StochasticSTLinearL1 implements Classifier, Regressor, Parameterized, SingleWeightVectorModel {

  public static enum Loss {
    SQUARED {
      @Override
      public double beta() {
        return 1;
      }

      @Override
      public CategoricalResults classify(double a) {
        final CategoricalResults cr = new CategoricalResults(2);

        a = (a + 1) / 2;

        if (a > 1) {
          a = 1;
        } else if (a < 0) {
          a = 0;
        }

        cr.setProb(1, a);
        cr.setProb(0, 1 - a);

        return cr;
      }

      @Override
      public double deriv(final double a, final double y) {
        return a - y;
      }

      @Override
      public double loss(final double a, final double y) {
        return 0.5 * Math.pow(a - y, 2);
      }

      @Override
      public double regress(final double a) {
        return a;
      }

    },
    LOG {
      @Override
      public double beta() {
        return 1.0 / 4.0;
      }

      @Override
      public CategoricalResults classify(final double a) {
        final CategoricalResults cr = new CategoricalResults(2);
        cr.setProb(1, regress(a));
        cr.setProb(0, 1.0 - cr.getProb(1));

        return cr;
      }

      @Override
      public double deriv(final double a, final double y) {
        return -y / (1 + Math.exp(a * y));
      }

      @Override
      public double loss(final double a, final double y) {
        return 1 + Math.exp(-y * a);
      }

      @Override
      public double regress(final double a) {
        return 1 / (1 + Math.exp(-a));
      }
    };

    /**
     * Returns an upper bound on the 2nd derivative for classification
     *
     * @return an upper bound on the 2nd derivative for classification
     */
    abstract public double beta();

    /**
     * The categorical results for a classification problem
     *
     * @param a
     *          the dot product of the weight vector and an input
     * @return the binary problem classification results
     */
    abstract public CategoricalResults classify(double a);

    /**
     * Returns the value of the derivative of the loss function
     *
     * @param a
     *          the predicted value
     * @param y
     *          the target value
     * @return the derivative of the loss
     */
    abstract public double deriv(double a, double y);

    /**
     * Returns the loss on the prediction
     *
     * @param a
     *          the predicted value
     * @param y
     *          the target value
     * @return the loss
     */
    abstract public double loss(double a, double y);

    /**
     * The output value result for a regression problem
     *
     * @param a
     *          the dot product of the weight vector and an input
     * @return the final regression output
     */
    abstract public double regress(double a);
  }

  private static final long serialVersionUID = -6761456665014802608L;
  public static final int DEFAULT_EPOCHS = 1000;
  public static final double DEFAULT_REG = 1e-14;
  public static final Loss DEFAULT_LOSS = Loss.SQUARED;
  /**
   * The number of training iterations
   */
  protected int epochs;

  /**
   * The regularization penalty
   */
  protected double lambda;
  /**
   * The loss function to use
   */
  protected Loss loss;
  /**
   * The final weight vector
   */
  protected Vec w;
  /**
   * The bias term to add
   */
  protected double bias;
  /**
   * The minimum observed value for each feature
   */
  protected double[] obvMin;

  /**
   * The maximum observed value for each feature
   */
  protected double[] obvMax;
  /**
   * Whether or not to perform feature rescaling
   */
  protected boolean reScale;
  /**
   * The scaled minimum
   */
  protected double minScaled = 0;

  /**
   * The scaled maximum
   */
  protected double maxScaled = 1;

  @Override
  abstract public StochasticSTLinearL1 clone();

  @Override
  public double getBias() {
    return bias;
  }

  @Override
  public double getBias(final int index) {
    if (index < 1) {
      return getBias();
    } else {
      throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }
  }

  /**
   * Returns the number of iterations of updating that will be done
   *
   * @return the number of iterations
   */
  public double getEpochs() {
    return epochs;
  }

  /**
   * Returns the amount of regularization to used in training
   *
   * @return the regularization parameter.
   */
  public double getLambda() {
    return lambda;
  }

  /**
   * returns the loss function in use
   *
   * @return the loss function in use
   */
  public Loss getLoss() {
    return loss;
  }

  /**
   * Returns the maximum feature value after scaling
   *
   * @return the maximum feature value after scaling
   */
  public double getMaxScaled() {
    return maxScaled;
  }

  /**
   * Returns the minimum feature value after scaling
   *
   * @return the minimum feature value after scaling
   */
  public double getMinScaled() {
    return minScaled;
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
  public Vec getRawWeight() {
    return w;
  }

  @Override
  public Vec getRawWeight(final int index) {
    if (index < 1) {
      return getRawWeight();
    } else {
      throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }
  }

  /**
   * Returns a copy of the weight vector used to compute results via a dot
   * product.
   *
   * @return a copy of the learned weight vector for prediction
   */
  public Vec getW() {
    if (w == null) {
      return w;
    } else {
      return w.clone();
    }
  }

  /**
   * Returns the weight vector used to compute results via a dot product. <br>
   * Do not modify this value, or you will alter the results returned.
   *
   * @return the learned weight vector for prediction
   */
  public Vec getWRaw() {
    return w;
  }

  /**
   * Returns if scaling is in use
   *
   * @return <tt>true</tt> if feature values are rescaled during training.
   */
  public boolean isReScale() {
    return reScale;
  }

  @Override
  public int numWeightsVecs() {
    return 1;
  }

  /**
   * Sets the number of iterations of training that will be performed.
   *
   * @param epochs
   *          the number of iterations
   */
  public void setEpochs(final int epochs) {
    if (epochs < 1) {
      throw new ArithmeticException("A positive amount of iterations must be performed");
    }
    this.epochs = epochs;
  }

  /**
   * Sets the regularization constant used for learning. The regularization must
   * be positive, and the learning rate is proportional to the regularization
   * value. This means regularizations very near zero will take a long time to
   * converge.
   *
   * @param lambda
   *          the regularization to apply
   */
  public void setLambda(final double lambda) {
    if (Double.isInfinite(lambda) || Double.isNaN(lambda) || lambda <= 0) {
      throw new ArithmeticException("A positive amount of regularization must be performed");
    }
    this.lambda = lambda;
  }

  /**
   * Sets the loss function to use. This should not be altered after training
   * unless the leaner is going to be trained again.
   *
   * @param loss
   *          the loss function to use
   */
  public void setLoss(final Loss loss) {
    this.loss = loss;
  }

  /**
   * Sets the maximum value of any feature after scaling is applied. This value
   * can be no greater than 1.
   *
   * @param maxFeature
   *          the maximum feature value after scaling
   */
  public void setMaxScaled(final double maxFeature) {
    if (Double.isNaN(maxFeature)) {
      throw new ArithmeticException("NaN is not a valid feature value");
    } else if (maxFeature > 1) {
      throw new ArithmeticException("Maximum possible feature value is 1, can not use " + maxFeature);
    } else if (maxFeature <= minScaled) {
      throw new ArithmeticException("Maximum feature value must be learger than the minimum");
    }
    maxScaled = maxFeature;
  }

  /**
   * Sets the minimum value of any feature after scaling is applied. This value
   * can be no smaller than -1
   *
   * @param minFeature
   *          the minimum feature value after scaling
   */
  public void setMinScaled(final double minFeature) {
    if (Double.isNaN(minFeature)) {
      throw new ArithmeticException("NaN is not a valid feature value");
    } else if (minFeature < -1) {
      throw new ArithmeticException("Minimum possible feature value is -1, can not use " + minFeature);
    } else if (minFeature >= maxScaled) {
      throw new ArithmeticException("Minimum feature value must be smaller than the maximum");
    }
    minScaled = minFeature;
  }

  /**
   * Sets whether or not scaling should be applied on th feature values of the
   * training vectors. Scaling should be used intelligently, scaling can destroy
   * sparsity in the data set. If scaling is not applied, and a value is not in
   * the range [-1, 1], a {@link FailedToFitException} could occur. <br>
   * Rescaling does not alter the data points passed in.
   *
   * @param reScale
   *          whether or not to rescale feature values
   */
  public void setReScale(final boolean reScale) {
    this.reScale = reScale;
  }

  /**
   * Computes {@link #w}.{@link Vec#dot(jsat.linear.Vec) }<tt>x</tt> and does so
   * by rescaling <tt>x</tt> as needed automatically and efficiently, even if
   * <tt>x</tt> is sparse.
   *
   * @param x
   *          the value to compute the dot product with
   * @return the dot produce of w and x with the bias term
   */
  protected double wDot(final Vec x) {
    double a;
    if (reScale) {
      a = bias;
      if (!w.isSparse()) // w is dense, jsut iterate over x
      {
        for (final IndexValue iv : x) {
          final int j = iv.getIndex();
          double xV = iv.getValue() - obvMin[j];
          xV *= (maxScaled - minScaled) / (obvMax[j] - obvMin[j]);
          xV += minScaled;
          a += w.get(j) * xV;
        }
        return a;
      }
      // Compute the dot and rescale w/o extra spacein a sprase freindly way
      final Iterator<IndexValue> wIter = w.getNonZeroIterator();
      final Iterator<IndexValue> xIter = x.getNonZeroIterator();

      if (!wIter.hasNext() || !xIter.hasNext()) {
        return a;
      }

      IndexValue wIV = wIter.next();
      IndexValue xIV = xIter.next();
      do {
        if (wIV.getIndex() == xIV.getIndex()) {
          final int j = xIV.getIndex();
          double xV = xIV.getValue() - obvMin[j];
          xV *= (maxScaled - minScaled) / (obvMax[j] - obvMin[j]);
          xV += minScaled;
          // Scaled, now add to result
          a += wIV.getValue() * xV;

          if (!wIter.hasNext() || !xIter.hasNext()) {
            break;
          }
          wIV = wIter.next();
          xIV = xIter.next();
        } else if (wIV.getIndex() < xIV.getIndex()) {
          if (wIter.hasNext()) {
            wIV = wIter.next();
          } else {
            break;
          }
        } else if (wIV.getIndex() > xIV.getIndex()) {
          if (xIter.hasNext()) {
            xIV = xIter.next();
          } else {
            break;
          }
        }
      } while (wIV != null && xIV != null);
    } else {
      a = w.dot(x) + bias;
    }
    return a;
  }
}
