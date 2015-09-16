package jsat.classifiers.linear;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.SingleWeightVectorModel;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.UpdateableClassifier;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.BaseUpdateableRegressor;
import jsat.regression.RegressionDataSet;
import jsat.regression.UpdateableRegressor;

/**
 * An implementations of the 3 versions of the Passive Aggressive algorithm for
 * binary classification and regression. Its a type of online algorithm that
 * performs the minimal update necessary to correct for a mistake. <br>
 * <br>
 * See:<br>
 * Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S.,&amp;Singer, Y.
 * (2006). <a href="http://dl.acm.org/citation.cfm?id=1248566"> <i>Online
 * passive-aggressive algorithms</i></a>. Journal of Machine Learning Research,
 * 7, 551–585.
 *
 * @author Edward Raff
 */
public class PassiveAggressive implements UpdateableClassifier, BinaryScoreClassifier, UpdateableRegressor,
    Parameterized, SingleWeightVectorModel {

  /**
   * Controls which version of the Passive Aggressive update is used
   */
  public static enum Mode {
    /**
     * The default Passive Aggressive algorithm, it performs correction updates
     * that make the minimal change necessary to correct the output for a single
     * input
     */
    PA, /**
         * Limits the aggressiveness by reducing the maximum correction to the
         * {@link #setC(double) aggressiveness parameter}
         */
    PA1, /**
          * Limits the aggressiveness by adding a constant factor to the
          * denominator of the correction.
          */
    PA2
  }

  private static final long serialVersionUID = -7130964391528405832L;

  /**
   * Guess the distribution to use for the regularization term
   * {@link #setC(double) C} in PassiveAggressive.
   *
   * @param d
   *          the data set to get the guess for
   * @return the guess for the C parameter
   */
  public static Distribution guessC(final DataSet d) {
    return new LogUniform(0.001, 100);
  }

  private int epochs;
  private double C = 0.01;
  private double eps = 0.001;

  private Vec w;

  private Mode mode;

  /**
   * Creates a new Passive Aggressive learner that does 10 epochs and uses
   * {@link Mode#PA1}
   */
  public PassiveAggressive() {
    this(10, Mode.PA1);
  }

  /**
   * Creates a new Passive Aggressive learner
   *
   * @param epochs
   *          the number of training epochs to use during batch training
   * @param mode
   *          which version of the update to perform
   */
  public PassiveAggressive(final int epochs, final Mode mode) {
    this.epochs = epochs;
    this.mode = mode;
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    final CategoricalResults cr = new CategoricalResults(2);
    if (getScore(data) > 0) {
      cr.setProb(1, 1);
    } else {
      cr.setProb(0, 1);
    }

    return cr;
  }

  @Override
  public PassiveAggressive clone() {
    final PassiveAggressive clone = new PassiveAggressive(epochs, mode);
    clone.eps = eps;
    clone.C = C;
    if (w != null) {
      clone.w = w;
    }

    return clone;
  }

  @Override
  public double getBias() {
    return 0;
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
   * Returns the aggressiveness parameter
   *
   * @return the aggressiveness parameter
   */
  public double getC() {
    return C;
  }

  private double getCorrection(final double loss, final Vec x) {
    final double xNorm = Math.pow(x.pNorm(2), 2);
    if (mode == Mode.PA1) {
      return Math.min(C, loss / xNorm);
    } else if (mode == Mode.PA2) {
      return loss / (xNorm + 1.0 / (2 * C));
    } else {
      return loss / xNorm;
    }
  }

  /**
   * Returns the number of epochs used for training
   *
   * @return the number of epochs used for training
   */
  public int getEpochs() {
    return epochs;
  }

  /**
   * Returns the maximum acceptable difference in prediction and truth
   *
   * @return the maximum acceptable difference in prediction and truth
   */
  public double getEps() {
    return eps;
  }

  /**
   * Returns which version of the PA update is used
   *
   * @return which PA update style is used
   */
  public Mode getMode() {
    return mode;
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

  @Override
  public double getScore(final DataPoint dp) {
    return dp.getNumericalValues().dot(w);
  }

  @Override
  public int numWeightsVecs() {
    return 1;
  }

  @Override
  public double regress(final DataPoint data) {
    return w.dot(data.getNumericalValues());
  }

  /**
   * Set the aggressiveness parameter. Increasing the value of this parameter
   * increases the aggressiveness of the algorithm. It must be a positive value.
   * This parameter essentially performs a type of regularization on the updates
   * <br>
   * An infinitely large value is equivalent to being completely aggressive, and
   * is performed when the mode is set to {@link Mode#PA}.
   *
   * @param C
   *          the positive aggressiveness parameter
   */
  public void setC(final double C) {
    if (Double.isNaN(C) || Double.isInfinite(C) || C <= 0) {
      throw new ArithmeticException("Aggressiveness must be a positive constant");
    }
    this.C = C;
  }

  /**
   * Sets the number of whole iterations through the training set that will be
   * performed for training
   *
   * @param epochs
   *          the number of whole iterations through the data set
   */
  public void setEpochs(final int epochs) {
    if (epochs < 1) {
      throw new IllegalArgumentException("epochs must be a positive value");
    }
    this.epochs = epochs;
  }

  /**
   * Sets the range for numerical prediction. If it is within range of the given
   * value, no error will be incurred.
   *
   * @param eps
   *          the maximum acceptable difference in prediction and truth
   */
  public void setEps(final double eps) {
    this.eps = eps;
  }

  /**
   * Sets which version of the PA update is used.
   *
   * @param mode
   *          which PA update style to perform
   */
  public void setMode(final Mode mode) {
    this.mode = mode;
  }

  @Override
  public void setUp(final CategoricalData[] categoricalAttributes, final int numericAttributes) {
    if (numericAttributes < 1) {
      throw new FailedToFitException("only suppors learning from numeric attributes");
    }
    w = new DenseVector(numericAttributes);
  }

  @Override
  public void setUp(final CategoricalData[] categoricalAttributes, final int numericAttributes,
      final CategoricalData predicting) {
    if (predicting.getNumOfCategories() != 2) {
      throw new FailedToFitException("Only supports binary classification problems");
    } else if (numericAttributes < 1) {
      throw new FailedToFitException("only suppors learning from numeric attributes");
    }
    w = new DenseVector(numericAttributes);
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void train(final RegressionDataSet dataSet) {
    BaseUpdateableRegressor.trainEpochs(dataSet, this, epochs);
  }

  @Override
  public void train(final RegressionDataSet dataSet, final ExecutorService threadPool) {
    train(dataSet);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    BaseUpdateableClassifier.trainEpochs(dataSet, this, epochs);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    trainC(dataSet);
  }

  @Override
  public void update(final DataPoint dataPoint, final double targetValue) {
    final Vec x = dataPoint.getNumericalValues();
    final double y_t = targetValue;
    final double y_p = x.dot(w);

    final double loss = Math.max(0, Math.abs(y_p - y_t) - eps);
    if (loss == 0) {
      return;
    }

    final double tau = getCorrection(loss, x);

    w.mutableAdd(Math.signum(y_t - y_p) * tau, x);
  }

  @Override
  public void update(final DataPoint dataPoint, final int targetClass) {
    final Vec x = dataPoint.getNumericalValues();
    final int y_t = targetClass * 2 - 1;
    final double dot = x.dot(w);

    final double loss = Math.max(0, 1 - y_t * dot);
    if (loss == 0) {
      return;
    }

    final double tau = getCorrection(loss, x);

    w.mutableAdd(y_t * tau, x);
  }
}
