package jsat.classifiers.boosting;

import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.distributions.ContinuousDistribution;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;

/**
 * Wagging is a meta-classifier that is related to {@link Bagging}. Instead
 * training on re-sampled data sets, it trains on randomly re-weighted data
 * sets. The weight of each point is selected at random from a specified
 * distribution, and set to zero if negative. <br>
 * <br>
 * See: <a href="http://www.springerlink.com/index/L006M1614W023752.pdf"> Bauer,
 * E.,&amp;Kohavi, R. (1999). <i>An empirical comparison of voting
 * classification algorithms</i>: Bagging, boosting, and variants. Machine
 * learning, 38(1998), 1–38.</a>
 *
 * @author Edward Raff
 */
public class Wagging implements Classifier, Regressor, Parameterized {

  /**
   * Fills a subset of the array
   */
  private class WagFill implements Runnable {

    int start;
    int end;
    DataSet ds;
    Random rand;
    CountDownLatch latch;

    public WagFill(final int start, final int end, final DataSet ds, final Random rand, final CountDownLatch latch) {
      this.start = start;
      this.end = end;
      this.ds = ds.shallowClone();
      this.rand = rand;
      this.latch = latch;

      // point at different objects so we can adjsut weights independently
      for (int i = 0; i < this.ds.getSampleSize(); i++) {
        final DataPoint dp = this.ds.getDataPoint(i);
        this.ds.setDataPoint(i,
            new DataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), dp.getCategoricalData()));
      }
    }

    @Override
    public void run() {
      if (ds instanceof ClassificationDataSet) {
        final ClassificationDataSet cds = (ClassificationDataSet) ds;
        for (int i = start; i < end; i++) {
          for (int j = 0; j < ds.getSampleSize(); j++) {
            final double newWeight = Math.max(1e-6, dist.invCdf(rand.nextDouble()));
            ds.getDataPoint(j).setWeight(newWeight);
          }
          final Classifier hypot = weakL.clone();
          hypot.trainC(cds);
          hypotsL[i] = hypot;
        }
      } else if (ds instanceof RegressionDataSet) {
        final RegressionDataSet rds = (RegressionDataSet) ds;
        for (int i = start; i < end; i++) {
          for (int j = 0; j < ds.getSampleSize(); j++) {
            ds.getDataPoint(i).setWeight(Math.max(1e-6, dist.invCdf(rand.nextDouble())));
          }
          final Regressor hypot = weakR.clone();
          hypot.train(rds);
          hypotsR[i] = hypot;
        }
      } else {
        throw new RuntimeException("BUG: please report");
      }

      latch.countDown();
    }
  }

  private static final long serialVersionUID = 4999034730848794619L;
  private ContinuousDistribution dist;
  private int iterations;
  private Classifier weakL;

  private Regressor weakR;

  private CategoricalData predicting;
  private Classifier[] hypotsL;

  private Regressor[] hypotsR;

  /**
   * Creates a new Wagging classifier
   *
   * @param dist
   *          the distribution to select weights from
   * @param weakL
   *          the weak learner to use
   * @param iterations
   *          the number of iterations to perform
   */
  public Wagging(final ContinuousDistribution dist, final Classifier weakL, final int iterations) {
    setDistribution(dist);
    setIterations(iterations);
    setWeakLearner(weakL);
  }

  /**
   * Creates a new Wagging regressor
   *
   * @param dist
   *          the distribution to select weights from
   * @param weakR
   *          the weak learner to use
   * @param iterations
   *          the number of iterations to perform
   */
  public Wagging(final ContinuousDistribution dist, final Regressor weakR, final int iterations) {
    setDistribution(dist);
    setIterations(iterations);
    setWeakLearner(weakR);
  }

  /**
   * Copy constructor
   *
   * @param clone
   *          the one to clone
   */
  protected Wagging(final Wagging clone) {
    dist = clone.dist.clone();
    iterations = clone.iterations;
    if (clone.weakL != null) {
      setWeakLearner(clone.weakL.clone());
    }
    if (clone.weakR != null) {
      setWeakLearner(clone.weakR.clone());
    }
    if (clone.predicting != null) {
      predicting = clone.predicting.clone();
    }

    if (clone.hypotsL != null) {
      hypotsL = new Classifier[clone.hypotsL.length];
      for (int i = 0; i < hypotsL.length; i++) {
        hypotsL[i] = clone.hypotsL[i].clone();
      }
    }
    if (clone.hypotsR != null) {
      hypotsR = new Regressor[clone.hypotsR.length];
      for (int i = 0; i < hypotsR.length; i++) {
        hypotsR[i] = clone.hypotsR[i].clone();
      }
    }
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    if (hypotsL == null) {
      throw new UntrainedModelException("Model has not been trained for classification");
    }

    final CategoricalResults results = new CategoricalResults(predicting.getNumOfCategories());

    for (final Classifier hypot : hypotsL) {
      results.incProb(hypot.classify(data).mostLikely(), 1);
    }
    results.normalize();
    return results;
  }

  @Override
  public Wagging clone() {
    return new Wagging(this);
  }

  /**
   * Returns the distribution used for weight sampling
   *
   * @return the distribution used
   */
  public ContinuousDistribution getDistribution() {
    return dist;
  }

  /**
   * Returns the number of iterations to create weak learners
   *
   * @return the number of iterations to perform
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

  /**
   * Returns the weak learner used for classification.
   *
   * @return the weak learner used for classification.
   */
  public Classifier getWeakClassifier() {
    return weakL;
  }

  /**
   * Returns the weak learner used for regression
   *
   * @return the weak learner used for regression
   */
  public Regressor getWeakRegressor() {
    return weakR;
  }

  private void performTraining(final ExecutorService threadPool, final DataSet dataSet) {
    final int chunkSize = iterations / SystemInfo.LogicalCores;
    int extra = iterations % SystemInfo.LogicalCores;

    int used = 0;
    final Random rand = new Random();
    final CountDownLatch latch = new CountDownLatch(chunkSize > 0 ? SystemInfo.LogicalCores : extra);
    while (used < iterations) {
      final int start = used;
      int end = start + chunkSize;
      if (extra-- > 0) {
        end++;
      }
      used = end;
      threadPool.submit(new WagFill(start, end, dataSet, new Random(rand.nextInt()), latch));
    }
    try {
      latch.await();
    } catch (final InterruptedException ex) {
      throw new FailedToFitException(ex);
    }
  }

  @Override
  public double regress(final DataPoint data) {
    if (hypotsR == null) {
      throw new UntrainedModelException("Model has not been trained for regression");
    }

    double avg = 0.0;
    for (final Regressor hypot : hypotsR) {
      avg += hypot.regress(data);
    }
    avg /= hypotsR.length;
    return avg;
  }

  /**
   * Sets the distribution to select the random weights from
   *
   * @param dist
   *          the distribution to use
   */
  public void setDistribution(final ContinuousDistribution dist) {
    if (dist == null) {
      throw new NullPointerException();
    }
    this.dist = dist;
  }

  /**
   * Sets the number of iterations to create weak learners
   *
   * @param iterations
   *          the number of iterations to perform
   */
  public void setIterations(final int iterations) {
    if (iterations < 1) {
      throw new ArithmeticException("The number of iterations must be positive");
    }
    this.iterations = iterations;
  }

  /**
   * Sets the weak learner used for classification. If it also supports
   * regressions that will be set as well.
   *
   * @param weakL
   *          the weak learner to use
   */
  public void setWeakLearner(final Classifier weakL) {
    if (weakL == null) {
      throw new NullPointerException();
    }
    this.weakL = weakL;
    if (weakL instanceof Regressor) {
      weakR = (Regressor) weakL;
    }
  }

  /**
   * Sets the weak learner used for regressions . If it also supports
   * classification that will be set as well.
   *
   * @param weakR
   *          the weak learner to use
   */
  public void setWeakLearner(final Regressor weakR) {
    if (weakR == null) {
      throw new NullPointerException();
    }
    this.weakR = weakR;
    if (weakR instanceof Classifier) {
      weakL = (Classifier) weakR;
    }
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
    if (weakR == null) {
      throw new FailedToFitException("No regression weak learner was provided");
    }
    hypotsL = null;
    hypotsR = new Regressor[iterations];

    performTraining(threadPool, dataSet);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    trainC(dataSet, new FakeExecutor());
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    if (weakL == null) {
      throw new FailedToFitException("No classification weak learner was provided");
    }
    predicting = dataSet.getPredicting();
    hypotsL = new Classifier[iterations];
    hypotsR = null;

    performTraining(threadPool, dataSet);
  }
}
