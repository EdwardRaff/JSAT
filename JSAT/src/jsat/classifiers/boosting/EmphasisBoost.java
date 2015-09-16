package jsat.classifiers.boosting;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;

import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.classifiers.trees.DecisionTree;
import jsat.distributions.Distribution;
import jsat.distributions.Uniform;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;

/**
 * Emphasis Boost is a generalization of the Real AdaBoost algorithm, expanding
 * the update term and providing the {@link #setLambda(double) &lambda; } term
 * to control the trade off. With &lambda; = 1/2, it becomes equivalent to Real
 * AdaBoost. If the weak learner does not support confidence outputs (non-hard
 * decisions), then it further becomes equivalent to Discrete Ada Boost. <br>
 * Emphasis Boost only supports binary classification problems, the learner used
 * should support weighted predictions. <br>
 * <br>
 * NOTE: In the face of extreme outliers, it is possible for numerical
 * instability to occur. This implementation attempts to reset weights when
 * numerical issues occur. <br>
 * <br>
 * See: <br>
 * Gómez-Verdejo, V., Ortega-Moral, M., Arenas-García, J.,&amp;Figueiras-Vidal,
 * A. R. (2006). <i>Boosting by weighting critical and erroneous samples</i>.
 * Neurocomputing, 69(7-9), 679–685. doi:10.1016/j.neucom.2005.12.011
 *
 * @author Edward Raff
 */
public class EmphasisBoost implements Classifier, Parameterized, BinaryScoreClassifier {

  private static final long serialVersionUID = -6372897830449685891L;

  /**
   * Guesses the distribution to use for the &lambda; parameter
   *
   * @param d
   *          the dataset to get the guess for
   * @return the guess for the &lambda; parameter
   * @see #setLambda(double)
   */
  public static Distribution guessLambda(final DataSet d) {
    return new Uniform(0.25, 0.45);
  }

  @ParameterHolder
  private Classifier weakLearner;
  private int maxIterations;
  /**
   * The list of weak hypothesis
   */
  protected List<Classifier> hypoths;
  /**
   * The weights for each weak learner
   */
  protected List<Double> hypWeights;
  protected CategoricalData predicting;

  private double lambda;

  /**
   * Creates a new EmphasisBooster with shallow decision trees and &lambda; =
   * 0.35
   */
  public EmphasisBoost() {
    this(new DecisionTree(6), 200, 0.35);
  }

  /**
   * Creates a new EmphasisBoost learner
   *
   * @param weakLearner
   *          the weak learner to use
   * @param maxIterations
   *          the maximum number of boosting iterations
   * @param lambda
   *          the trade off parameter in [0, 1]
   */
  public EmphasisBoost(final Classifier weakLearner, final int maxIterations, final double lambda) {
    setWeakLearner(weakLearner);
    setMaxIterations(maxIterations);
    setLambda(lambda);
  }

  /**
   * Copy constructor
   *
   * @param toClone
   *          the object to clone
   */
  protected EmphasisBoost(final EmphasisBoost toClone) {
    this(toClone.weakLearner.clone(), toClone.maxIterations, toClone.lambda);
    if (toClone.hypWeights != null) {
      hypWeights = new DoubleList(toClone.hypWeights);
      hypoths = new ArrayList<Classifier>(toClone.maxIterations);
      for (final Classifier weak : toClone.hypoths) {
        hypoths.add(weak.clone());
      }
      predicting = toClone.predicting.clone();
    }
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    if (predicting == null) {
      throw new RuntimeException("Classifier has not been trained yet");
    }

    final CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());

    final double score = getScore(data);
    if (score < 0) {
      cr.setProb(0, 1.0);
    } else {
      cr.setProb(1, 1.0);
    }
    return cr;
  }

  @Override
  public EmphasisBoost clone() {
    return new EmphasisBoost(this);
  }

  /**
   * Returns the value of the &lambda; trade off parameter
   *
   * @return the value of the &lambda; trade off parameter
   */
  public double getLambda() {
    return lambda;
  }

  /**
   * Returns the maximum number of iterations used
   *
   * @return the maximum number of iterations used
   */
  public int getMaxIterations() {
    return maxIterations;
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
    double score = 0;
    for (int i = 0; i < hypoths.size(); i++) {
      score += H(hypoths.get(i), dp) * hypWeights.get(i);
    }
    return score;
  }

  /**
   * Returns the weak learner currently being used by this method.
   *
   * @return the weak learner currently being used by this method.
   */
  public Classifier getWeakLearner() {
    return weakLearner;
  }

  private double H(final Classifier weak, final DataPoint dp) {
    final CategoricalResults catResult = weak.classify(dp);

    return catResult.getProb(1) * 2 - 1;
  }

  /**
   * &lambda; controls the trade off between weighting the errors based on their
   * distance to the margin and the quadratic error of the output. The three
   * extreme values are: <br>
   * <ul>
   * <li>&lambda; = 0 , in this case all the weight is placed on points based on
   * their distance to the margin of the classification boundary.</li>
   * <li>&lambda; = 1/2, in this case weight is balanced between the margin
   * distance and the quadratic error. This is equivalent to Real Ada Boost</li>
   * <li>&lambda; = 1, in this case the weight is placed purely based on the
   * quadratic error of the output</li>
   * </ul>
   * <br>
   * <br>
   * According to the original paper, values in the range [0.3, 0.4] often
   * perform well.
   *
   * @param lambda
   *          the trade off parameter in [0, 1]
   */
  public void setLambda(final double lambda) {
    this.lambda = lambda;
  }

  /**
   * Sets the maximal number of boosting iterations that may be performed
   *
   * @param maxIterations
   *          the maximum number of iterations
   */
  public void setMaxIterations(final int maxIterations) {
    if (maxIterations < 1) {
      throw new IllegalArgumentException("Iterations must be positive, not " + maxIterations);
    }
    this.maxIterations = maxIterations;
  }

  /**
   * Sets the weak learner used during training.
   *
   * @param weakLearner
   *          the weak learner to use
   */
  public void setWeakLearner(final Classifier weakLearner) {
    if (!weakLearner.supportsWeightedData()) {
      throw new IllegalArgumentException("WeakLearner must support weighted data to be boosted");
    }
    this.weakLearner = weakLearner;
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    trainC(dataSet, null);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    predicting = dataSet.getPredicting();
    hypWeights = new DoubleList(maxIterations);
    hypoths = new ArrayList<Classifier>(maxIterations);
    final int N = dataSet.getSampleSize();

    final List<DataPointPair<Integer>> dataPoints = dataSet.getTwiceShallowClone().getAsDPPList();
    // Initialization step, set up the weights so they are all 1 / size of
    // dataset
    for (final DataPointPair<Integer> dpp : dataPoints) {
      dpp.getDataPoint().setWeight(1.0 / N);// Scaled, they are all 1
    }
    double weightSum = 1;

    // Keep track of the cumaltive score for everything
    final double[] H_cur = new double[N];
    final double[] curH_Result = new double[N];

    for (int t = 0; t < maxIterations; t++) {
      final Classifier weak = weakLearner.clone();
      if (threadPool != null && !(threadPool instanceof FakeExecutor)) {
        weak.trainC(new ClassificationDataSet(dataPoints, predicting), threadPool);
      } else {
        weak.trainC(new ClassificationDataSet(dataPoints, predicting));
      }

      double error = 0.0;
      for (int i = 0; i < dataPoints.size(); i++) {
        final DataPointPair<Integer> dpp = dataPoints.get(i);
        final double y_hat = H_cur[i] = H(weak, dpp.getDataPoint());
        final double y_true = dpp.getPair() * 2 - 1;// {-1 or 1}
        error += dpp.getDataPoint().getWeight() * y_hat * y_true;
      }

      if (error < 0) {
        return;
      }

      final double alpha_m = Math.log((1 + error) / (1 - error)) / 2;

      weightSum = 0;

      for (int i = 0; i < dataPoints.size(); i++) {
        curH_Result[i] += alpha_m * H_cur[i];
        final double f_t = curH_Result[i];

        final DataPointPair<Integer> dpp = dataPoints.get(i);
        final DataPoint dp = dpp.getDataPoint();
        final double y_true = dpp.getPair() * 2 - 1;

        double w_i = Math.exp(lambda * Math.pow(f_t - y_true, 2) - (1 - lambda) * f_t * f_t);
        if (Double.isInfinite(w_i)) {
          w_i = 50;// Let it grow back isntead of bizaro huge values
        }
        weightSum += w_i;
        dp.setWeight(w_i);
      }

      for (int i = 0; i < dataPoints.size(); i++) {
        final DataPointPair<Integer> dpp = dataPoints.get(i);
        final DataPoint dp = dpp.getDataPoint();
        dp.setWeight(dp.getWeight() / weightSum);
      }

      hypoths.add(weak);
      hypWeights.add(alpha_m);
    }
  }
}
