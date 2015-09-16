package jsat.classifiers.linear;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;

import jsat.DataSet;
import jsat.SimpleWeightVectorModel;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.ScaledVector;
import jsat.linear.Vec;
import jsat.lossfunctions.HingeLoss;
import jsat.lossfunctions.LossC;
import jsat.lossfunctions.LossFunc;
import jsat.lossfunctions.LossMC;
import jsat.lossfunctions.LossR;
import jsat.math.decayrates.DecayRate;
import jsat.math.decayrates.PowerDecay;
import jsat.math.optimization.stochastic.GradientUpdater;
import jsat.math.optimization.stochastic.SimpleSGD;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.BaseUpdateableRegressor;
import jsat.regression.RegressionDataSet;
import jsat.regression.UpdateableRegressor;

/**
 * LinearSGD learns either a classification or regression problem depending on
 * the {@link #setLoss(jsat.lossfunctions.LossFunc) loss function &#x2113;(w,x)}
 * used. The solution attempts to minimize <big>&sum;</big><sub>i</sub>
 * &#x2113;(w,x<sub>i</sub>) + {@link #setLambda0(double) &lambda;<sub>0</sub>}
 * /2 ||w||<sub>2</sub><sup>2</sup> + {@link #setLambda1(double) &lambda;
 * <sub>1</sub>} ||w||<sub>1</sub>, and is trained by Stochastic Gradient
 * Descent. <br>
 * <br>
 * <br>
 * NOTE: To support L<sub>1</sub> regularization with sparse results and online
 * learning at the same time, the normalization of the regularization penalty by
 * the number of data points is not present in the implementation at this time.
 * Setting {@link #setLambda1(double) &lambda;<sub>1</sub>} to the desired value
 * divided by the number of unique data points in the whole set will result in
 * the correct regularization penalty being applied.
 *
 * See:
 * <ul>
 * <li>Tsuruoka, Y., Tsujii, J.,&amp;Ananiadou, S. (2009). <i>Stochastic
 * gradient descent training for L1-regularized log-linear models with
 * cumulative penalty</i>. Proceedings of the Joint Conference of the 47th
 * Annual Meeting of the ACL and the 4th International Joint Conference on
 * Natural Language Processing of the AFNLP, 1, 477. doi:10.3115/1687878.1687946
 * </li>
 * </ul>
 *
 * @author Edward Raff
 */
public class LinearSGD extends BaseUpdateableClassifier
    implements UpdateableRegressor, Parameterized, SimpleWeightVectorModel {

  private static final long serialVersionUID = -59695592724956535L;

  /**
   * Guess the distribution to use for the regularization term
   * {@link #setLambda0(double) &lambda;<sub>0</sub>} .
   *
   * @param d
   *          the data set to get the guess for
   * @return the guess for the &lambda;<sub>0</sub> parameter
   */
  public static Distribution guessLambda0(final DataSet d) {
    return new LogUniform(1e-7, 1e-2);
  }

  /**
   * Guess the distribution to use for the regularization term
   * {@link #setLambda0(double) &lambda;<sub>1</sub>} .
   *
   * @param d
   *          the data set to get the guess for
   * @return the guess for the &lambda;<sub>1</sub> parameter
   */
  public static Distribution guessLambda1(final DataSet d) {
    final int N = d.getSampleSize();
    return new LogUniform(1e-7 / N, 1e-3 / N);
  }

  private LossFunc loss;
  private GradientUpdater gradientUpdater;
  private double eta;
  private DecayRate decay;
  private Vec[] ws;
  private GradientUpdater[] gus;
  private double[] bs;
  private int time;
  private double lambda0;
  private double lambda1;
  private double l1U;

  private double[][] l1Q;

  private boolean useBias = true;

  /**
   * Creates a new LinearSGD learner for multi-class classification problems.
   */
  public LinearSGD() {
    this(new HingeLoss(), 1e-4, 0.0);
  }

  /**
   * Copy constructor
   *
   * @param toClone
   *          the object to copy
   */
  public LinearSGD(final LinearSGD toClone) {
    loss = toClone.loss.clone();
    eta = toClone.eta;
    decay = toClone.decay.clone();
    time = toClone.time;
    lambda0 = toClone.lambda0;
    lambda1 = toClone.lambda1;
    l1U = toClone.l1U;
    useBias = toClone.useBias;
    gradientUpdater = toClone.gradientUpdater;
    if (toClone.l1Q != null) {
      l1Q = new double[toClone.l1Q.length][];
      for (int i = 0; i < toClone.l1Q.length; i++) {
        l1Q[i] = Arrays.copyOf(toClone.l1Q[i], toClone.l1Q[i].length);
      }
    }
    if (toClone.ws != null) {
      ws = new Vec[toClone.ws.length];
      bs = new double[toClone.bs.length];
      gus = new GradientUpdater[toClone.gus.length];
      for (int i = 0; i < ws.length; i++) {
        ws[i] = toClone.ws[i].clone();
        bs[i] = toClone.bs[i];
        gus[i] = toClone.gus[i].clone();
      }
    }
  }

  /**
   * Creates a new LinearSGD learner.
   *
   * @param loss
   *          the loss function to use
   * @param eta
   *          the initial learning rate
   * @param decay
   *          the decay rate for &eta;
   * @param lambda0
   *          the L<sub>2</sub> regularization term
   * @param lambda1
   *          the L<sub>1</sub> regularization term
   */
  public LinearSGD(final LossFunc loss, final double eta, final DecayRate decay, final double lambda0,
      final double lambda1) {
    setLoss(loss);
    setEta(eta);
    setEtaDecay(decay);
    setGradientUpdater(new SimpleSGD());
    setLambda0(lambda0);
    setLambda1(lambda1);
  }

  /**
   * Creates a new LinearSGD learner
   *
   * @param loss
   *          the loss function to use
   * @param lambda0
   *          the L<sub>2</sub> regularization term
   * @param lambda1
   *          the L<sub>1</sub> regularization term
   */
  public LinearSGD(final LossFunc loss, final double lambda0, final double lambda1) {
    this(loss, 0.001, new PowerDecay(1, 0.1), lambda0, lambda1);
  }

  /**
   * Applies L1 regularization to the model
   *
   * @param eta_t
   *          the learning rate in use
   * @param x
   *          the input vector the update is from
   */
  private void applyL1Reg(final double eta_t, final Vec x) {
    // apply l1 regularization
    if (lambda1 > 0) {
      l1U += eta_t * lambda1;// line 6: in Tsuruoka et al paper, figure 2
      for (int k = 0; k < ws.length; k++) {
        final Vec w_k = ws[k];
        final double[] l1Q_k = l1Q[k];
        for (final IndexValue iv : x) {
          final int i = iv.getIndex();
          // see "APPLYPENALTY(i)" on line 15: from Figure 2 in Tsuruoka et al
          // paper
          final double z = w_k.get(i);
          double newW_i;
          if (z > 0) {
            newW_i = Math.max(0, z - (l1U + l1Q_k[i]));
          } else {
            newW_i = Math.min(0, z + (l1U - l1Q_k[i]));
          }
          l1Q_k[i] += newW_i - z;
          w_k.set(i, newW_i);
        }
      }
    }
  }

  /**
   * Applies L2 regularization to the model
   *
   * @param eta_t
   *          the learning rate in use
   */
  private void applyL2Reg(final double eta_t) {
    if (lambda0 > 0) {// apply L2 regularization
      for (final Vec v : ws) {
        v.mutableMultiply(1 - eta_t * lambda0);
      }
    }
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    final Vec x = data.getNumericalValues();
    if (ws.length == 1) {
      return ((LossC) loss).getClassification(ws[0].dot(x) + bs[0]);
    } else {
      final Vec pred = new DenseVector(ws.length);
      for (int i = 0; i < ws.length; i++) {
        pred.set(i, ws[i].dot(x) + bs[i]);
      }
      ((LossMC) loss).process(pred, pred);
      return ((LossMC) loss).getClassification(pred);
    }
  }

  @Override
  public LinearSGD clone() {
    return new LinearSGD(this);
  }

  @Override
  public double getBias(final int index) {
    return bs[index];
  }

  /**
   * Returns the current learning rate in use
   *
   * @return the current learning rate in use
   */
  public double getEta() {
    return eta;
  }

  /**
   * Returns the decay rate in use
   *
   * @return the decay rate in use
   */
  public DecayRate getEtaDecay() {
    return decay;
  }

  /**
   *
   * @return the method to use for updating the weight vectors from the gradient
   */
  public GradientUpdater getGradientUpdater() {
    return gradientUpdater;
  }

  /**
   * Returns the L<sub>2</sub> regularization term in use
   *
   * @return the L<sub>2</sub> regularization term in use
   */
  public double getLambda0() {
    return lambda0;
  }

  /**
   * Returns the L<sub>1</sub> regularization term in use
   *
   * @return the L<sub>1</sub> regularization term in use
   */
  public double getLambda1() {
    return lambda1;
  }

  /**
   * Returns the loss function in use
   *
   * @return the loss function in use
   */
  public LossFunc getLoss() {
    return loss;
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
  public Vec getRawWeight(final int index) {
    return ws[index];
  }

  /**
   * Returns whether or not an implicit bias term is in use
   *
   * @return {@code true} if a bias term is in use
   */
  public boolean isUseBias() {
    return useBias;
  }

  @Override
  public int numWeightsVecs() {
    return ws.length;
  }

  /**
   *
   * @param i
   *          the index of the weight vector array to update
   * @param eta_t
   *          the learning rate to use
   * @param lossD
   *          the loss for the specified weight vector
   * @param x
   *          the input vector the loss was incurred on
   */
  private void performGradientUpdate(final int i, final double eta_t, final double lossD, final Vec x) {
    final Vec grad = new ScaledVector(lossD, x);
    if (useBias) {
      bs[i] -= gus[i].update(ws[i], grad, eta_t, bs[i], lossD);
    } else {
      gus[i].update(ws[i], grad, eta_t);
    }
  }

  @Override
  public double regress(final DataPoint data) {
    final Vec x = data.getNumericalValues();
    return ((LossR) loss).getRegression(ws[0].dot(x) + bs[0]);
  }

  /**
   * Sets the initial learning rate &eta; to use. It should generally be in (0,
   * 1), but any positive value is acceptable.
   *
   * @param eta
   *          the learning rate to use.
   */
  public void setEta(final double eta) {
    if (eta <= 0 || Double.isNaN(eta) || Double.isInfinite(eta)) {
      throw new IllegalArgumentException("eta must be a positive constant, not " + eta);
    }
    this.eta = eta;
  }

  /**
   * Sets the rate at which {@link #setEta(double) &eta;} is decayed at each
   * update.
   *
   * @param decay
   *          the decay rate to use
   */
  public void setEtaDecay(final DecayRate decay) {
    this.decay = decay;
  }

  /**
   * Sets the method that will be used to update the weight vectors given their
   * gradient information.
   *
   * @param gradientUpdater
   *          the method to use for updating the weight vectors from the
   *          gradient
   */
  public void setGradientUpdater(final GradientUpdater gradientUpdater) {
    if (gradientUpdater == null) {
      throw new IllegalArgumentException("Gradient updater must be non-null");
    }
    this.gradientUpdater = gradientUpdater;
  }

  /**
   * &lambda;<sub>0</sub> controls the L<sub>2</sub> regularization penalty.
   *
   * @param lambda0
   *          the L<sub>2</sub> regularization penalty to use
   */
  public void setLambda0(final double lambda0) {
    if (lambda0 < 0 || Double.isNaN(lambda0) || Double.isInfinite(lambda0)) {
      throw new IllegalArgumentException("Lambda0 must be non-negative, not " + lambda0);
    }
    this.lambda0 = lambda0;
  }

  /**
   * &lambda;<sub>1</sub> controls the L<sub>1</sub> regularization penalty.
   *
   * @param lambda1
   *          the L<sub>1</sub> regularization penalty to use
   */
  public void setLambda1(final double lambda1) {
    if (lambda1 < 0 || Double.isNaN(lambda1) || Double.isInfinite(lambda1)) {
      throw new IllegalArgumentException("Lambda1 must be non-negative, not " + lambda1);
    }
    this.lambda1 = lambda1;
  }

  /**
   * Sets the loss function used for the model. The loss function controls
   * whether or not regression, binary classification, or multi-class
   * classification is supported.
   *
   * @param loss
   *          the loss function to use
   */
  public void setLoss(final LossFunc loss) {
    this.loss = loss;
  }

  @Override
  public void setUp(final CategoricalData[] categoricalAttributes, final int numericAttributes) {
    if (!(loss instanceof LossR)) {
      throw new FailedToFitException(
          "Loss function " + loss.getClass().getSimpleName() + "does not support regression");
    }

    ws = new Vec[1];
    bs = new double[1];
    gus = new GradientUpdater[1];
    setUpShared(numericAttributes);
  }

  @Override
  public void setUp(final CategoricalData[] categoricalAttributes, final int numericAttributes,
      final CategoricalData predicting) {
    if (!(loss instanceof LossC)) {
      throw new FailedToFitException("Loss function " + loss.getClass().getSimpleName() + " only supports regression");
    }
    if (predicting.getNumOfCategories() == 2) {
      ws = new Vec[1];
      bs = new double[1];
      gus = new GradientUpdater[1];
    } else {
      if (!(loss instanceof LossMC)) {
        throw new FailedToFitException(
            "Loss function " + loss.getClass().getSimpleName() + " only supports binary classification");
      }
      ws = new Vec[predicting.getNumOfCategories()];
      bs = new double[predicting.getNumOfCategories()];
      gus = new GradientUpdater[predicting.getNumOfCategories()];
    }
    setUpShared(numericAttributes);
  }

  private void setUpShared(final int numericAttributes) {
    if (numericAttributes <= 0) {
      throw new FailedToFitException("LinearSGD requires numeric features to use");
    }
    for (int i = 0; i < ws.length; i++) {
      ws[i] = new ScaledVector(new DenseVector(numericAttributes));
      gus[i] = gradientUpdater.clone();
      gus[i].setup(ws[i].length());
    }
    time = 0;
    l1U = 0;
    if (lambda1 > 0) {
      l1Q = new double[ws.length][ws[0].length()];
    } else {
      l1Q = null;
    }
  }

  /**
   * Sets whether or not an implicit bias term will be added to the data set
   *
   * @param useBias
   *          {@code true} to add an implicit bias term
   */
  public void setUseBias(final boolean useBias) {
    this.useBias = useBias;
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void train(final RegressionDataSet dataSet) {
    BaseUpdateableRegressor.trainEpochs(dataSet, this, getEpochs());
  }

  @Override
  public void train(final RegressionDataSet dataSet, final ExecutorService threadPool) {
    train(dataSet);
  }

  @Override
  public void update(final DataPoint dataPoint, final double targetValue) {
    final double eta_t = decay.rate(time++, eta);
    final Vec x = dataPoint.getNumericalValues();

    applyL2Reg(eta_t);

    final double lossD = loss.getDeriv(ws[0].dot(x) + bs[0], targetValue);

    performGradientUpdate(0, eta_t, lossD, x);

    applyL1Reg(eta_t, x);
  }

  @Override
  public void update(final DataPoint dataPoint, final int targetClass) {
    final double eta_t = decay.rate(time++, eta);
    final Vec x = dataPoint.getNumericalValues();

    applyL2Reg(eta_t);

    // apply gradient updates
    if (ws.length == 1) {
      final double y = targetClass * 2 - 1;
      final double lossD = loss.getDeriv(ws[0].dot(x) + bs[0], y);
      performGradientUpdate(0, eta_t, lossD, x);
    } else {
      final Vec pred = new DenseVector(ws.length);
      for (int i = 0; i < ws.length; i++) {
        pred.set(i, ws[i].dot(x) + bs[i]);
      }
      ((LossMC) loss).process(pred, pred);
      ((LossMC) loss).deriv(pred, pred, targetClass);
      for (final IndexValue iv : pred) {
        final int i = iv.getIndex();
        final double lossD = iv.getValue();
        performGradientUpdate(i, eta_t, lossD, x);
      }
    }

    applyL1Reg(eta_t, x);
  }

}
