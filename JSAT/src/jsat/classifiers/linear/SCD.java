package jsat.classifiers.linear;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;

import jsat.SingleWeightVectorModel;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.lossfunctions.LogisticLoss;
import jsat.lossfunctions.LossC;
import jsat.lossfunctions.LossFunc;
import jsat.lossfunctions.LossR;
import jsat.lossfunctions.SquaredLoss;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.random.XORWOW;

/**
 * Implementation of Stochastic Coordinate Descent for L1 regularized
 * classification and regression. Which one is supported is controlled by the
 * {@link LossFunc} used. To be used the loss function must be twice
 * differentiable with a finite maximal second derivative value.
 * {@link LogisticLoss} for classification and {@link SquaredLoss} for
 * regression are the ones used in the original paper. <br>
 * <br>
 * Because the SCD needs column major data for efficient implementation, a
 * second copy of data will be created in memory during training. <br>
 * <br>
 * See: Shalev-Shwartz, S.,&amp;Tewari, A. (2009). <i>Stochastic Methods for L
 * <sub>1</sub>-regularized Loss Minimization</i>. In 26th International
 * Conference on Machine Learning (Vol. 12, pp. 929–936). Retrieved from
 * <a href="http://eprints.pascal-network.org/archive/00005418/">here</a>
 *
 * @author Edward Raff
 */
public class SCD implements Classifier, Regressor, Parameterized, SingleWeightVectorModel {

  private static final long serialVersionUID = 3576901723216525618L;
  private Vec w;
  private final LossFunc loss;
  private double reg;
  private int iterations;

  /**
   * Creates anew SCD learner
   *
   * @param loss
   *          the loss function to use
   * @param regularization
   *          the regularization term to used
   * @param iterations
   *          the number of iterations to perform
   */
  public SCD(final LossFunc loss, final double regularization, final int iterations) {
    final double beta = loss.getDeriv2Max();
    if (Double.isNaN(beta) || Double.isInfinite(beta) || beta <= 0) {
      throw new IllegalArgumentException("SCD needs a loss function with a finite positive maximal second derivative");
    }
    this.loss = loss;
    setRegularization(regularization);
    setIterations(iterations);
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public SCD(final SCD toCopy) {
    this(toCopy.loss.clone(), toCopy.reg, toCopy.iterations);
    if (toCopy.w != null) {
      w = toCopy.w.clone();
    }
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    if (w != null && loss instanceof LossC) {
      return ((LossC) loss).getClassification(w.dot(data.getNumericalValues()));
    } else {
      throw new UntrainedModelException("Model was not trained with a classification function");
    }
  }

  @Override
  public SCD clone() {
    return new SCD(this);
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
   * Returns the number of iterations used
   *
   * @return the number of iterations used
   */
  public int getIterations() {
    return iterations;
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
   * Returns the regularization parameter value used for learning.
   *
   * @return the regularization parameter value used for learning.
   */
  public double getRegularization() {
    return reg;
  }

  @Override
  public int numWeightsVecs() {
    return 1;
  }

  @Override
  public double regress(final DataPoint data) {
    if (w != null && loss instanceof LossR) {
      return ((LossR) loss).getRegression(w.dot(data.getNumericalValues()));
    } else {
      throw new UntrainedModelException("Model was not trained with a classification function");
    }
  }

  /**
   * Sets the number of iterations that will be used.
   *
   * @param iterations
   *          the number of training iterations
   */
  public void setIterations(final int iterations) {
    if (iterations < 1) {
      throw new IllegalArgumentException("The iterations must be a positive value, not " + iterations);
    }
    this.iterations = iterations;
  }

  /**
   * Sets the regularization constant used for learning. The regularization must
   * be positive, and the learning rate is proportional to the regularization
   * value. This means regularizations very near zero will take a long time to
   * converge.
   *
   * @param regularization
   *          the regularization to apply in (0, Infinity)
   */
  public void setRegularization(final double regularization) {
    if (Double.isInfinite(regularization) || Double.isNaN(regularization) || regularization <= 0) {
      throw new IllegalArgumentException("Regularization must be a positive value");
    }
    reg = regularization;
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void train(final RegressionDataSet dataSet) {
    train(dataSet.getNumericColumns(), dataSet.getTargetValues().arrayCopy());
  }

  @Override
  public void train(final RegressionDataSet dataSet, final ExecutorService threadPool) {
    train(dataSet);
  }

  /**
   *
   * @param columns
   *          columns of the training matrix
   * @param y
   *          the target values
   */
  private void train(final Vec[] columns, final double[] y) {
    final double beta = loss.getDeriv2Max();
    final double[] z = new double[y.length];/// stores w.dot(x)
    w = new DenseVector(columns.length);
    final Random rand = new XORWOW();
    for (int iter = 0; iter < iterations; iter++) {
      final int j = rand.nextInt(columns.length);
      double g = 0;
      for (final IndexValue iv : columns[j]) {
        g += loss.getDeriv(z[iv.getIndex()], y[iv.getIndex()]) * iv.getValue();
      }
      g /= y.length;
      final double w_j = w.get(j);
      final double eta;
      if (w_j - g / beta > reg / beta) {
        eta = -g / beta - reg / beta;
      } else if (w_j - g / beta < -reg / beta) {
        eta = -g / beta + reg / beta;
      } else {
        eta = -w_j;
      }
      w.increment(j, eta);

      for (final IndexValue iv : columns[j]) {
        z[iv.getIndex()] += eta * iv.getValue();
      }
    }
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    final double[] targets = new double[dataSet.getSampleSize()];
    for (int i = 0; i < targets.length; i++) {
      targets[i] = dataSet.getDataPointCategory(i) * 2 - 1;
    }
    train(dataSet.getNumericColumns(), targets);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    trainC(dataSet);
  }
}
