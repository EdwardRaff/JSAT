package jsat.regression;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;

import jsat.SingleWeightVectorModel;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.bayesian.NaiveBayes;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.SubVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.optimization.IterativelyReweightedLeastSquares;
import jsat.math.optimization.Optimizer;
import jsat.utils.FakeExecutor;

/**
 * Logistic regression is a common method used to fit a probability between
 * binary outputs. It can also be used to perform regression on a real function.
 * For classification tasks, when all variables are independent, the results
 * converge to the same prediction values as {@link NaiveBayes}. When variables
 * have a high degree of correlation, Logistic Regression should produce better
 * results.
 *
 * @author Edward Raff
 */
public class LogisticRegression implements Classifier, Regressor, SingleWeightVectorModel {

  private static final long serialVersionUID = -5115807516729861730L;

  private static double logit(final double z) {
    return 1 / (1 + Math.exp(-z));
  }

  private Vec coefficents;
  /**
   * Logistic regression needs values on the range [0, 1]. The shift makes sure
   * that values are on the range [0, x], for some x
   */
  private double shift;

  /**
   * Logistic regression needs values on the range [0, 1]. The scale makes sure
   * that values of the form [0, x] are converted to [0, 1]
   */
  private double scale;

  final private Function logitFun = new Function() {

    /**
     *
     */
    private static final long serialVersionUID = -653111120605227341L;

    @Override
    public double f(final double... x) {
      return logitReg(DenseVector.toDenseVec(x));
    }

    @Override
    public double f(final Vec x) {
      return logitReg(x);
    }
  };

  final private Function logitFunD = new Function() {

    /**
     *
     */
    private static final long serialVersionUID = 4844651397674391691L;

    @Override
    public double f(final double... x) {
      return logitReg(DenseVector.toDenseVec(x));
    }

    @Override
    public double f(final Vec x) {
      final double y = logitReg(x);
      return y * (1 - y);
    }
  };

  @Override
  public CategoricalResults classify(final DataPoint data) {
    if (coefficents == null) {
      throw new UntrainedModelException("Model has not yet been trained");
    } else if (shift != 0 || scale != 1) {
      throw new UntrainedModelException("Model was trained for regression, not classifiaction");
    }
    final CategoricalResults results = new CategoricalResults(2);

    // It looks a little backwards. But if the true class is 0, and we are
    // accurate, then we expect regress to return a value near zero.
    results.setProb(1, regress(data));
    results.setProb(0, 1.0 - results.getProb(1));
    return results;
  }

  @Override
  public LogisticRegression clone() {
    final LogisticRegression clone = new LogisticRegression();
    if (coefficents != null) {
      clone.coefficents = coefficents.clone();
    }
    clone.scale = scale;
    clone.shift = shift;
    return clone;
  }

  @Override
  public double getBias() {
    return coefficents.get(0);
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
   * Returns the backing vector that containing the learned coefficients for the
   * logistic regression. Changes to it will alter the results of the model
   *
   * @return the backing vector that containing the learned coefficients for the
   *         logistic regression.
   */
  public Vec getCoefficents() {
    return coefficents;
  }

  @Override
  public Vec getRawWeight() {
    return new SubVector(1, coefficents.length() - 1, coefficents);
  }

  @Override
  public Vec getRawWeight(final int index) {
    if (index < 1) {
      return getRawWeight();
    } else {
      throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }
  }

  private double logitReg(final Vec input) {
    double z = coefficents.get(0);
    for (int i = 1; i < coefficents.length(); i++) {
      z += input.get(i - 1) * coefficents.get(i);
    }
    return logit(z);
  }

  @Override
  public int numWeightsVecs() {
    return 1;
  }

  @Override
  public double regress(final DataPoint data) {
    if (coefficents == null) {
      throw new UntrainedModelException("Model has not been trained");
    }
    return logitReg(data.getNumericalValues()) * scale + shift;
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void train(final RegressionDataSet dataSet) {
    train(dataSet, new FakeExecutor());
  }

  @Override
  public void train(final RegressionDataSet dataSet, final ExecutorService threadPool) {
    final List<Vec> inputs = new ArrayList<Vec>(dataSet.getSampleSize());
    for (int i = 0; i < dataSet.getSampleSize(); i++) {
      inputs.add(dataSet.getDataPoint(i).getNumericalValues());
    }

    coefficents = new DenseVector(dataSet.getNumNumericalVars() + 1);
    final Vec targetValues = dataSet.getTargetValues();
    final double minTarget = targetValues.min();
    final double maxTarget = targetValues.max();
    shift = minTarget;
    scale = maxTarget - minTarget;

    // Now all values are in the range [0, 1]
    targetValues.subtract(shift);
    targetValues.mutableDivide(scale);

    final Optimizer optimizer = new IterativelyReweightedLeastSquares();

    coefficents = optimizer.optimize(1e-5, 100, logitFun, logitFunD, coefficents, inputs, targetValues, threadPool);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    trainC(dataSet, new FakeExecutor());
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    if (dataSet.getClassSize() != 2) {
      throw new FailedToFitException("Logistic Regression works only in the case of two classes, and can not handle "
          + dataSet.getClassSize() + " classes");
    }
    final RegressionDataSet rds = new RegressionDataSet(dataSet.getNumNumericalVars(), dataSet.getCategories());
    for (int i = 0; i < dataSet.getSampleSize(); i++) {
      // getDataPointCategory will return either 0 or 1, so it works perfectly
      rds.addDataPoint(dataSet.getDataPoint(i), dataSet.getDataPointCategory(i));
    }

    train(rds, threadPool);

  }

}
