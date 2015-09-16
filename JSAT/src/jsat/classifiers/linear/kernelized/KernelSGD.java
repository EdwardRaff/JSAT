package jsat.classifiers.linear.kernelized;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.UpdateableClassifier;
import jsat.classifiers.linear.LinearSGD;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.kernels.KernelPoint;
import jsat.distributions.kernels.KernelPoints;
import jsat.distributions.kernels.KernelTrick;
import jsat.distributions.kernels.RBFKernel;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.lossfunctions.LossC;
import jsat.lossfunctions.LossFunc;
import jsat.lossfunctions.LossMC;
import jsat.lossfunctions.LossR;
import jsat.lossfunctions.SoftmaxLoss;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.regression.BaseUpdateableRegressor;
import jsat.regression.RegressionDataSet;
import jsat.regression.UpdateableRegressor;

/**
 * Kernel SGD is the kernelized counterpart to {@link LinearSGD}, and learns
 * nonlinear functions via the kernel trick. The implementation is built upon
 * {@link KernelPoint} and {@link KernelPoints} to support budgeted learning.
 * Following the LinearSGD implementation, whether or not this algorithm
 * supports regression, binary-classification, or multi-class classification is
 * controlled by the {@link #setLoss(jsat.lossfunctions.LossFunc) loss function}
 * used. <br>
 * <br>
 * The learning rate decay is not configurable for this implementation, and is
 * decayed at a rate of null {@link #setEta(double) &eta;} / (
 * {@link #setLambda(double) &lambda;} * (t + 2 / &lambda;)) , where {@code t}
 * is the time step.
 *
 * @author Edward Raff
 */
public class KernelSGD implements UpdateableClassifier, UpdateableRegressor, Parameterized {

  private static final long serialVersionUID = -4956596506787859023L;

  /**
   * Guess the distribution to use for the regularization term
   * {@link #setLambda(double) &lambda;} .
   *
   * @param d
   *          the data set to get the guess for
   * @return the guess for the &lambda; parameter
   */
  public static Distribution guessLambda(final DataSet d) {
    return new LogUniform(1e-7, 1e-2);
  }

  private LossFunc loss;
  @ParameterHolder
  private KernelTrick kernel;
  private double lambda;
  private double eta;
  private KernelPoint.BudgetStrategy budgetStrategy;
  private int budgetSize;
  private double errorTolerance;
  private int time;
  private KernelPoint kpoint;
  private KernelPoints kpoints;

  private int epochs = 1;

  /**
   * Creates a new Kernel SGD object for classification with the RBF kernel
   */
  public KernelSGD() {
    this(new SoftmaxLoss(), new RBFKernel(), 1e-4, KernelPoint.BudgetStrategy.MERGE_RBF, 300);
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public KernelSGD(final KernelSGD toCopy) {
    loss = toCopy.loss.clone();
    kernel = toCopy.kernel.clone();
    lambda = toCopy.lambda;
    eta = toCopy.eta;
    budgetStrategy = toCopy.budgetStrategy;
    budgetSize = toCopy.budgetSize;
    errorTolerance = toCopy.errorTolerance;
    time = toCopy.time;
    epochs = toCopy.epochs;
    if (toCopy.kpoint != null) {
      kpoint = toCopy.kpoint.clone();
    }
    if (toCopy.kpoints != null) {
      kpoints = toCopy.kpoints.clone();
    }
  }

  /**
   * Creates a new Kernel SGD object
   *
   * @param loss
   *          the loss function to use
   * @param kernel
   *          the kernel trick to use
   * @param lambda
   *          the regularization penalty
   * @param budgetStrategy
   *          the budget maintenance strategy to use
   * @param budgetSize
   *          the maximum support vector budget
   */
  public KernelSGD(final LossFunc loss, final KernelTrick kernel, final double lambda,
      final KernelPoint.BudgetStrategy budgetStrategy, final int budgetSize) {
    this(loss, kernel, lambda, budgetStrategy, budgetSize, 1.0, 0.05);
  }

  /**
   * Creates a new Kernel SGD object
   *
   * @param loss
   *          the loss function to use
   * @param kernel
   *          the kernel trick to use
   * @param lambda
   *          the regularization penalty
   * @param eta
   *          the initial learning rate
   * @param budgetStrategy
   *          the budget maintenance strategy to use
   * @param errorTolerance
   *          the error tolerance used in certain budget maintenance steps
   * @param budgetSize
   *          the maximum support vector budget
   */
  public KernelSGD(final LossFunc loss, final KernelTrick kernel, final double lambda,
      final KernelPoint.BudgetStrategy budgetStrategy, final int budgetSize, final double eta,
      final double errorTolerance) {
    setLoss(loss);
    setKernel(kernel);
    setLambda(lambda);
    setEta(eta);
    setBudgetStrategy(budgetStrategy);
    setErrorTolerance(errorTolerance);
    setBudgetSize(budgetSize);
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    final Vec x = data.getNumericalValues();
    final List<Double> qi = kernel.getQueryInfo(x);
    if (kpoint != null) {
      return ((LossC) loss).getClassification(kpoint.dot(x, qi));
    } else {
      final Vec pred = new DenseVector(kpoints.dot(x, qi));
      ((LossMC) loss).process(pred, pred);
      return ((LossMC) loss).getClassification(pred);
    }
  }

  @Override
  public KernelSGD clone() {
    return new KernelSGD(this);
  }

  /**
   * Returns the budget size, or maximum number of allowed support vectors.
   *
   * @return the maximum number of allowed support vectors
   */
  public int getBudgetSize() {
    return budgetSize;
  }

  /**
   * Returns the method of budget maintenance
   *
   * @return the method of budget maintenance
   */
  public KernelPoint.BudgetStrategy getBudgetStrategy() {
    return budgetStrategy;
  }

  /**
   * Returns the number of epochs to use
   *
   * @return the number of epochs to use
   */
  public int getEpochs() {
    return epochs;
  }

  /**
   * Returns the error tolerance that would be used
   *
   * @return the error tolerance that would be used
   */
  public double getErrorTolerance() {
    return errorTolerance;
  }

  /**
   * Returns the base learning rate
   *
   * @return the base learning rate
   */
  public double getEta() {
    return eta;
  }

  /**
   * Returns the kernel in use
   *
   * @return the kernel in use
   */
  public KernelTrick getKernel() {
    return kernel;
  }

  /**
   * Returns the L<sub>2</sub> regularization parameter
   *
   * @return the L<sub>2</sub> regularization parameter
   */
  public double getLambda() {
    return lambda;
  }

  /**
   * Returns the loss function in use
   *
   * @return the loss function in use
   */
  public LossFunc getLoss() {
    return loss;
  }

  private double getNextEta() {
    return eta / (lambda * (++time + 2 / lambda));
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
  public double regress(final DataPoint data) {
    final Vec x = data.getNumericalValues();
    final List<Double> qi = kernel.getQueryInfo(x);
    return ((LossR) loss).getRegression(kpoint.dot(x, qi));
  }

  /**
   * Sets the maximum budget size, or number of support vectors, to allow during
   * training. Increasing the budget size will increase the accuracy of the
   * model, but will also increase the computational cost
   *
   * @param budgetSize
   *          the maximum allowed number of support vectors
   */
  public void setBudgetSize(final int budgetSize) {
    if (budgetSize < 1) {
      throw new IllegalArgumentException("Budgest size must be a positive constant, not " + budgetSize);
    }
    this.budgetSize = budgetSize;
  }

  /**
   * Sets the budget maintenance strategy.
   *
   * @param budgetStrategy
   *          the method to meet budget size requirements
   */
  public void setBudgetStrategy(final KernelPoint.BudgetStrategy budgetStrategy) {
    if (budgetStrategy == null) {
      throw new NullPointerException("Budget strategy must be non null");
    }
    this.budgetStrategy = budgetStrategy;
  }

  /**
   * Sets the number of iterations of the training set done during batch
   * training
   *
   * @param epochs
   *          the number of iterations in batch training
   */
  public void setEpochs(final int epochs) {
    if (epochs < 1) {
      throw new IllegalArgumentException("Epochs must be a poistive constant, not " + epochs);
    }
    this.epochs = epochs;
  }

  /**
   * Sets the error tolerance used for certain
   * {@link #setBudgetStrategy(jsat.distributions.kernels.KernelPoint.BudgetStrategy)
   * budget strategies}
   *
   * @param errorTolerance
   *          the error tolerance in [0, 1]
   */
  public void setErrorTolerance(final double errorTolerance) {
    if (errorTolerance < 0 || errorTolerance > 1 || Double.isNaN(errorTolerance)) {
      throw new IllegalArgumentException("Error tolerance must be in [0, 1], not " + errorTolerance);
    }
    this.errorTolerance = errorTolerance;
  }

  /**
   * Sets the base learning rate to start from. Because of the decay rate in use
   * a good value for &eta; is 1.0.
   *
   * @param eta
   *          the starting learning rate to use
   */
  public void setEta(final double eta) {
    this.eta = eta;
  }

  /**
   * Sets the kernel to use
   *
   * @param kernel
   *          the kernel to use
   */
  public void setKernel(final KernelTrick kernel) {
    if (kernel == null) {
      throw new NullPointerException("kernel trick must be non null");
    }
    this.kernel = kernel;
  }

  /**
   * Sets the L<sub>2</sub> regularization parameter used during learning.
   *
   * @param lambda
   *          the positive regularization parameter
   */
  public void setLambda(final double lambda) {
    if (lambda <= 0 || Double.isNaN(lambda) || Double.isInfinite(lambda)) {
      throw new IllegalArgumentException("lambda must be a positive constant, not " + lambda);
    }
    this.lambda = lambda;
  }

  /**
   * Sets the loss function to use. The loss function controls whether or not
   * classification or regression is supported.
   *
   * @param loss
   */
  public void setLoss(final LossFunc loss) {
    if (loss == null) {
      throw new NullPointerException("Loss may not be null");
    }
    this.loss = loss;
  }

  @Override
  public void setUp(final CategoricalData[] categoricalAttributes, final int numericAttributes) {
    if (!(loss instanceof LossR)) {
      throw new FailedToFitException(
          "Loss in use (" + loss.getClass().getSimpleName() + ") does not support regession");
    }
    kpoint = new KernelPoint(kernel, errorTolerance);
    kpoint.setBudgetStrategy(budgetStrategy);
    kpoint.setErrorTolerance(errorTolerance);
    kpoint.setMaxBudget(budgetSize);
    kpoints = null;
    time = 0;
  }

  @Override
  public void setUp(final CategoricalData[] categoricalAttributes, final int numericAttributes,
      final CategoricalData predicting) {
    if (!(loss instanceof LossC)) {
      throw new FailedToFitException(
          "Loss in use (" + loss.getClass().getSimpleName() + ") does not support classification");
    }
    if (predicting.getNumOfCategories() == 2) {
      kpoint = new KernelPoint(kernel, errorTolerance);
      kpoint.setBudgetStrategy(budgetStrategy);
      kpoint.setErrorTolerance(errorTolerance);
      kpoint.setMaxBudget(budgetSize);
      kpoints = null;

    } else {
      if (!(loss instanceof LossMC)) {
        throw new FailedToFitException(
            "Loss in use (" + loss.getClass().getSimpleName() + ") does not support multi-class classification");
      }
      kpoint = null;
      kpoints = new KernelPoints(kernel, predicting.getNumOfCategories(), errorTolerance);
      kpoints.setBudgetStrategy(budgetStrategy);
      kpoints.setErrorTolerance(errorTolerance);
      kpoints.setMaxBudget(budgetSize);
    }

    time = 0;
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
    final List<Double> qi = kernel.getQueryInfo(x);

    final double eta_t = getNextEta();

    kpoint.mutableMultiply(1 - eta_t * lambda);
    final double y = targetValue;
    final double dot = kpoint.dot(x, qi);
    final double lossD = loss.getDeriv(dot, y);
    if (lossD != 0) {
      kpoint.mutableAdd(-eta_t * lossD, x, qi);
    }

  }

  @Override
  public void update(final DataPoint dataPoint, final int targetClass) {
    final Vec x = dataPoint.getNumericalValues();
    final List<Double> qi = kernel.getQueryInfo(x);

    final double eta_t = getNextEta();

    if (kpoint != null) {
      kpoint.mutableMultiply(1 - eta_t * lambda);
      final double y = targetClass * 2 - 1;
      final double dot = kpoint.dot(x, qi);
      final double lossD = loss.getDeriv(dot, y);
      if (lossD != 0) {
        kpoint.mutableAdd(-eta_t * lossD, x, qi);
      }
    } else if (kpoints != null) {
      kpoints.mutableMultiply(1 - eta_t * lambda);
      final Vec pred = new DenseVector(kpoints.dot(x, qi));
      ((LossMC) loss).process(pred, pred);
      ((LossMC) loss).deriv(pred, pred, targetClass);
      pred.mutableMultiply(-eta_t);// should we wrap in a scaledVec? Probably
                                   // fine unless someone pulls out a 200 class
                                   // problem
      kpoints.mutableAdd(x, pred, qi);
    }
  }

}
