package jsat.classifiers.linear;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;

import jsat.DataSet;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;

/**
 * Implements an iterative and single threaded form of fast Stochastic
 * Coordinate Decent for optimizing L<sub>1</sub> regularized linear regression
 * problems. It performs very well when the number of data points is very large,
 * especially when the feature count is smaller in comparison. It also works
 * well on sparse data sets. <br>
 * Using the squared loss is equivalent to LASSO regression, and the LOG loss is
 * equivalent to logistic regression. <br>
 * <br>
 * Note: This implementation requires all feature values to be in the range [-1,
 * 1]. By default scaling is performed to [0,1]. If your data is dense or has
 * negative and positive feature values, scaling to [-1, 1] may perform better.
 * See {@link #setReScale(boolean) }<br>
 * <br>
 * See:<br>
 * <a href="http://eprints.pascal-network.org/archive/00005418/">Shalev-Shwartz,
 * S.,&amp;Tewari, A. (2009). <i>Stochastic Methods for L<sub>1</sub>
 * -regularized Loss Minimization</i>. 26th International Conference on Machine
 * Learning (Vol. 12, pp. 929–936).</a>
 *
 * @author Edward Raff
 */
public class LinearL1SCD extends StochasticSTLinearL1 {

  private static final long serialVersionUID = 3135562347568407186L;

  /**
   * Creates a new SCD L<sub>1</sub> learner using default settings.
   */
  public LinearL1SCD() {
    this(DEFAULT_EPOCHS, DEFAULT_REG, DEFAULT_LOSS);
  }

  /**
   * Creates a new SCD L<sub>1</sub> learner.
   *
   * @param epochs
   *          the number of learning iterations
   * @param lambda
   *          the regularization penalty
   * @param loss
   *          the loss function to use
   */
  public LinearL1SCD(final int epochs, final double lambda, final Loss loss) {
    this(epochs, lambda, loss, true);
  }

  /**
   * Creates a new SCD L<sub>1</sub> learner.
   *
   * @param epochs
   *          the number of learning iterations
   * @param lambda
   *          the regularization penalty
   * @param loss
   *          the loss function to use
   * @param reScale
   *          whether or not to rescale the feature values
   */
  public LinearL1SCD(final int epochs, final double lambda, final Loss loss, final boolean reScale) {
    setEpochs(epochs);
    setLambda(lambda);
    setLoss(loss);
    setReScale(reScale);
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    if (w == null) {
      throw new UntrainedModelException("Model has not been trained");
    }
    final Vec x = data.getNumericalValues();
    return loss.classify(wDot(x));
  }

  @Override
  public LinearL1SCD clone() {
    final LinearL1SCD clone = new LinearL1SCD(epochs, lambda, loss, reScale);

    if (w != null) {
      clone.w = w.clone();
    }
    clone.bias = bias;
    clone.minScaled = minScaled;
    clone.maxScaled = maxScaled;
    if (obvMin != null) {
      clone.obvMin = Arrays.copyOf(obvMin, obvMin.length);
    }
    if (obvMax != null) {
      clone.obvMax = Arrays.copyOf(obvMax, obvMax.length);
    }

    return clone;
  }

  /**
   * Performs rescaling as requested or throws an exception if a violation was
   * encountered
   *
   * @param featureVals
   *          the array of feature values
   * @param m
   *          the number of data points
   * @throws FailedToFitException
   */
  private void featureScaleCheck(final Vec[] featureVals, final int m) throws FailedToFitException {
    if (reScale) {
      for (int j = 0; j < featureVals.length; j++) {
        if (obvMin[j] == 0 && minScaled == 0) // We can skip 1st and last step
        {
          featureVals[j].mutableMultiply(maxScaled / obvMax[j]);
        } else// do all steps
        {
          featureVals[j].mutableSubtract(obvMin[j]);
          featureVals[j].mutableMultiply((maxScaled - minScaled) / (obvMax[j] - obvMin[j]));
          featureVals[j].mutableAdd(minScaled);
        }
        if (featureVals[j].isSparse() && featureVals[j].nnz() > m * 0.75) {
          featureVals[j] = new DenseVector(featureVals[j]);
        }
      }
    } else {
      for (int j = 0; j < obvMin.length; j++) {
        if (obvMax[j] > 1 || obvMin[j] < -1) {
          throw new FailedToFitException("All feature values must be in the range [-1,1]");
        }
      }
    }
  }

  @Override
  public double regress(final DataPoint data) {
    if (w == null) {
      throw new UntrainedModelException("Model has not been trained");
    }
    final Vec x = data.getNumericalValues();
    return loss.regress(wDot(x));
  }

  private void setUpFeatureVals(final Vec[] featureVals, final boolean sparse, final int m, final DataSet dataSet) {
    // All feature values need to be scaled into -1, 1
    obvMin = new double[featureVals.length];
    Arrays.fill(obvMin, Double.POSITIVE_INFINITY);
    obvMax = new double[featureVals.length];
    Arrays.fill(obvMax, Double.NEGATIVE_INFINITY);
    for (int i = 0; i < featureVals.length; i++) {
      featureVals[i] = sparse ? new SparseVector(m) : new DenseVector(m);
    }
    if (sparse) {
      Arrays.fill(obvMin, 0.0);
    }

    for (int i = 0; i < dataSet.getSampleSize(); i++) {
      final Vec x = dataSet.getDataPoint(i).getNumericalValues();
      for (final IndexValue iv : x) {
        final int j = iv.getIndex();
        final double v = iv.getValue();
        featureVals[j].set(i, v);
        obvMax[j] = Math.max(obvMax[j], v);
        obvMin[j] = Math.min(obvMin[j], v);
      }
    }
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void train(final RegressionDataSet dataSet) {
    final boolean sparse = dataSet.getDataPoint(0).getNumericalValues().isSparse();
    final int m = dataSet.getSampleSize();

    final Vec[] featureVals = new Vec[dataSet.getNumNumericalVars()];
    for (int i = 0; i < featureVals.length; i++) {
      featureVals[i] = sparse ? new SparseVector(m) : new DenseVector(m);
    }

    setUpFeatureVals(featureVals, sparse, m, dataSet);

    featureScaleCheck(featureVals, m);

    final double[] target = new double[m];
    for (int i = 0; i < dataSet.getSampleSize(); i++) {
      target[i] = dataSet.getTargetValue(i);
    }

    train(featureVals, target);
  }

  @Override
  public void train(final RegressionDataSet dataSet, final ExecutorService threadPool) {
    train(dataSet);
  }

  /**
   *
   * @param featureVals
   *          a vector for each feature, where each vector contains all values
   *          for the feature in dataset order
   * @param target
   *          target values
   */
  private void train(final Vec[] featureVals, final double[] target) {
    final int d = featureVals.length;
    final int m = target.length;
    w = new DenseVector(d);
    final double[] z = new double[m];
    final double beta = loss.beta();

    final Random rand = new Random();
    for (int t = 1; t <= epochs; t++) {
      final int j = rand.nextInt(d + 1);// +1 for the bias term

      double g = 0.0;
      if (j < d) {
        final Vec xj = featureVals[j];
        for (final IndexValue iv : xj) {
          final int i = iv.getIndex();
          g += loss.deriv(z[i], target[i]) * iv.getValue();
        }
      } else// Bias term update, all x[i]_j = 1
      {
        for (int i = 0; i < target.length; i++) {
          g += loss.deriv(z[i], target[i]);
        }
      }
      g /= m;

      double eta;
      final double w_j = j == d ? bias : w.get(j);
      if (w_j - g / beta > lambda / beta) {
        eta = -g / beta - lambda / beta;
      } else if (w_j - g / beta < -lambda / beta) {
        eta = -g / beta + lambda / beta;
      } else {
        eta = -w_j;
      }

      if (j < d) {
        w.increment(j, eta);
      } else {
        bias += eta;
      }

      if (j < d) {
        for (final IndexValue iv : featureVals[j]) {
          z[iv.getIndex()] += eta * iv.getValue();
        }
      } else {
        for (int i = 0; i < target.length; i++) {
          z[i] += eta;
        }
      }
    }
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    if (dataSet.getClassSize() != 2) {
      throw new FailedToFitException("Only binary classification problems are supported");
    }
    final boolean sparse = dataSet.getDataPoint(0).getNumericalValues().isSparse();
    final int m = dataSet.getSampleSize();
    final Vec[] featureVals = new Vec[dataSet.getNumNumericalVars()];

    setUpFeatureVals(featureVals, sparse, m, dataSet);

    featureScaleCheck(featureVals, m);

    final double[] target = new double[m];
    for (int i = 0; i < dataSet.getSampleSize(); i++) {
      target[i] = dataSet.getDataPointCategory(i) * 2 - 1;
    }

    train(featureVals, target);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    trainC(dataSet);
  }

}
