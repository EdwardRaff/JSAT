package jsat.classifiers.trees;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;

import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.boosting.Bagging;
import jsat.classifiers.trees.ImpurityScore.ImpurityMeasure;
import jsat.math.OnLineStatistics;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.FakeExecutor;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.AtomicDoubleArray;

/**
 * Random Forest is an extension of {@link Bagging} that is applied only to
 * {@link DecisionTree DecisionTrees}. It works in a similar manner, but also
 * only uses a random sub set of the features for each tree trained. This
 * provides increased performance in accuracy of predictions, and reduced
 * training time over just Bagging.
 *
 * @author Edward Raff
 * @see Bagging
 */
public class RandomForest implements Classifier, Regressor, Parameterized {
  // TODO implement Out of Bag estimates of proximity, importance, and outlier
  // detection

  private class LearningWorker implements Callable<LearningWorker> {

    int toLearn;
    List<DecisionTree> learned;
    DataSet dataSet;
    Random random;
    /**
     * For regression: sum of predictions
     */
    private AtomicDoubleArray votes;

    private int[][] counts;

    public LearningWorker(final DataSet dataSet, final int toLearn, final Random random, final int[][] counts,
        final AtomicDoubleArray pred) {
      this.dataSet = dataSet;
      this.toLearn = toLearn;
      this.random = random;
      learned = new ArrayList<DecisionTree>(toLearn);
      if (useOutOfBagError) {
        votes = pred;
        this.counts = counts;
      }
    }

    @Override
    public LearningWorker call() throws Exception {
      final Set<Integer> features = new IntSet(baseLearner.getRandomFeatureCount());
      final int[] sampleCounts = new int[dataSet.getSampleSize()];
      for (int i = 0; i < toLearn; i++) {
        // Sample to get the training points
        Bagging.sampleWithReplacement(sampleCounts, sampleCounts.length + extraSamples, random);
        // Sample to select the feature subset
        features.clear();
        while (features.size() < Math.min(baseLearner.getRandomFeatureCount(), dataSet.getNumFeatures())) {
          features.add(random.nextInt(dataSet.getNumFeatures()));
        }

        final DecisionTree learner = baseLearner.clone();

        if (dataSet instanceof ClassificationDataSet) {
          learner.trainC(Bagging.getWeightSampledDataSet((ClassificationDataSet) dataSet, sampleCounts), features);
        } else {
          // It must be regression!
          learner.train(Bagging.getWeightSampledDataSet((RegressionDataSet) dataSet, sampleCounts), features);
        }
        learned.add(learner);
        if (useOutOfBagError) {
          for (int j = 0; j < sampleCounts.length; j++) {
            if (sampleCounts[j] != 0) {
              continue;
            }

            final DataPoint dp = dataSet.getDataPoint(j);
            if (dataSet instanceof ClassificationDataSet) {
              final int pred = learner.classify(dp).mostLikely();
              synchronized (counts[j]) {
                counts[j][pred]++;
              }
            } else {
              votes.getAndAdd(j, learner.regress(dp));
              synchronized (counts[j]) {
                counts[j][0]++;
              }
            }
          }
        }
      }
      return this;
    }

  }

  private static final long serialVersionUID = 2725020584282958141L;
  /**
   * Only used when training for a classification problem
   */
  private CategoricalData predicting;
  private int extraSamples;
  /**
   * Setting the number of features to use. Default value is -1, indicating the
   * heuristic of sqrt(N) or N/3 should be used for classification and
   * regression respectively. This value should be set away from -1 before
   * training work begins, and set back if it was not set explicitly by the used
   */
  private int featureSamples;
  private int maxForestSize;
  private boolean useOutOfBagError = false;
  private double outOfBagError;
  private RandomDecisionTree baseLearner;

  private List<DecisionTree> forest;

  public RandomForest() {
    this(100);
  }

  public RandomForest(final int maxForestSize) {
    setExtraSamples(0);
    setMaxForestSize(maxForestSize);
    autoFeatureSample();
    baseLearner = new RandomDecisionTree(1, Integer.MAX_VALUE, 3, TreePruner.PruningMethod.NONE, 1e-15);
    baseLearner.setNumericHandling(DecisionStump.NumericHandlingC.BINARY_BEST_GAIN);
    baseLearner.setGainMethod(ImpurityMeasure.GINI);
  }

  /**
   * Tells the class to automatically select the number of features to use. For
   * classification problems, this is the square root of the number of features.
   * For regression, the number of features divided by 3 is used.
   */
  public void autoFeatureSample() {
    featureSamples = -1;
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    if (forest == null || forest.isEmpty()) {
      throw new RuntimeException("Classifier has not yet been trained");
    } else if (predicting == null) {
      throw new RuntimeException("Classifier has been trained for regression");
    }
    final CategoricalResults totalResult = new CategoricalResults(predicting.getNumOfCategories());
    for (final DecisionTree tree : forest) {
      totalResult.incProb(tree.classify(data).mostLikely(), 1.0);
    }

    totalResult.normalize();
    return totalResult;
  }

  @Override
  public RandomForest clone() {
    final RandomForest clone = new RandomForest(maxForestSize);
    clone.extraSamples = extraSamples;
    clone.featureSamples = featureSamples;
    if (predicting != null) {
      clone.predicting = predicting.clone();
    }
    if (forest != null) {
      clone.forest = new ArrayList<DecisionTree>(forest.size());
      for (final DecisionTree tree : forest) {
        clone.forest.add(tree.clone());
      }
    }
    clone.baseLearner = baseLearner.clone();

    return clone;
  }

  public int getExtraSamples() {
    return extraSamples;
  }

  /**
   * Returns the number of rounds of boosting that will be done, which is also
   * the number of base learners that will be trained
   *
   * @return the number of rounds of boosting that will be done, which is also
   *         the number of base learners that will be trained
   */
  public int getMaxForestSize() {
    return maxForestSize;
  }

  /**
   * If {@link #isUseOutOfBagError() } is false, then this method will return 0
   * after training. Otherwise, it will return the out of bag error estimate
   * after training has completed. For classification problems, this is the 0/1
   * loss error rate. Regression problems return the mean squared error.
   *
   * @return the out of bag error estimate for this predictor
   */
  public double getOutOfBagError() {
    return outOfBagError;
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
   * Returns true if heuristics are currently in use for the number of features,
   * or false if the number has been specified.
   *
   * @return true if heuristics are currently in use for the number of features,
   *         or false if the number has been specified.
   */
  public boolean isAutoFeatureSample() {
    return featureSamples == -1;
  }

  /**
   * Indicates if the out of bag error rate will be computed during training
   *
   * @return <tt>true</tt> if the out of bag error will be computed,
   *         <tt>false</tt> otherwise
   */
  public boolean isUseOutOfBagError() {
    return useOutOfBagError;
  }

  @Override
  public double regress(final DataPoint data) {
    if (forest == null || forest.isEmpty()) {
      throw new RuntimeException("Classifier has not yet been trained");
    } else if (predicting != null) {
      throw new RuntimeException("Classifier has been trained for classification");
    }
    final OnLineStatistics stats = new OnLineStatistics();
    for (final DecisionTree tree : forest) {
      stats.add(tree.regress(data));
    }
    return stats.getMean();
  }

  /**
   * RandomForest performs Bagging. Bagging samples from the training set with
   * replacement, and draws a sampleWithReplacement at least as large as the
   * training set. This controls how many extra samples are taken. If negative,
   * fewer samples will be taken. Using negative values is not recommended.
   *
   * @param i
   *          how many extra samples to take
   */
  public void setExtraSamples(final int i) {
    extraSamples = i;
  }

  /**
   * Instead of using a heuristic, the exact number of features to sample is
   * provided. If equal to or larger then the number of features in one of the
   * training data sets, RandomForest degrades to {@link Bagging} performed on
   * {@link DecisionTree}.<br>
   * <br>
   * To re-enable the heuristic mode, call {@link #autoFeatureSample() }
   *
   * @param featureSamples
   *          the number of features to randomly select for each tree in the
   *          forest.
   * @throws ArithmeticException
   *           if the number given is less then or equal to zero
   * @see #autoFeatureSample()
   * @see Bagging
   */
  public void setFeatureSamples(final int featureSamples) {
    if (featureSamples <= 0) {
      throw new ArithmeticException("A positive number of features must be given");
    }
    this.featureSamples = featureSamples;
  }

  /**
   * Sets the maximum number of trees to create for the forest.
   *
   * @param maxForestSize
   *          the number of base learners to train
   * @throws ArithmeticException
   *           if the number specified is not a positive value
   */
  public void setMaxForestSize(final int maxForestSize) {
    if (maxForestSize <= 0) {
      throw new ArithmeticException("Must train a positive number of learners");
    }
    this.maxForestSize = maxForestSize;
  }

  /**
   * Sets whether or not to compute the out of bag error during training
   *
   * @param useOutOfBagError
   *          <tt>true</tt> to compute the out of bag error, <tt>false</tt> to
   *          skip it
   */
  public void setUseOutOfBagError(final boolean useOutOfBagError) {
    this.useOutOfBagError = useOutOfBagError;
  }

  @Override
  public boolean supportsWeightedData() {
    return true;
  }

  @Override
  public void train(final RegressionDataSet dataSet) {
    train(dataSet, new FakeExecutor());
  }

  @Override
  public void train(final RegressionDataSet dataSet, final ExecutorService threadPool) {
    predicting = null;
    forest = new ArrayList<DecisionTree>(maxForestSize);
    trainStep(dataSet, threadPool);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    trainC(dataSet, new FakeExecutor());
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    predicting = dataSet.getPredicting();
    forest = new ArrayList<DecisionTree>(maxForestSize);
    trainStep(dataSet, threadPool);
  }

  /**
   * Does the actual set up and training. {@link #predicting } and
   * {@link #forest} should be set up appropriately first. Everything else is
   * handled by this and {@link LearningWorker}
   *
   * @param dataSet
   *          the data set, classification or regression
   * @param threadPool
   *          the source of threads
   */
  private void trainStep(final DataSet dataSet, final ExecutorService threadPool) {
    final boolean autoLearners = isAutoFeatureSample();// We will need to set it
                                                       // back after, so
                                                       // remember if we need to
    if (autoLearners) {
      baseLearner.setRandomFeatureCount(Math.max((int) Math.sqrt(dataSet.getNumFeatures()), 1));
    } else {
      baseLearner.setRandomFeatureCount(featureSamples);
    }

    int roundsToDistribut = maxForestSize;
    int roundShare = roundsToDistribut / SystemInfo.LogicalCores;// The number
                                                                 // of rounds
                                                                 // each thread
                                                                 // gets
    int extraRounds = roundsToDistribut % SystemInfo.LogicalCores;// The number
                                                                  // of extra
                                                                  // rounds that
                                                                  // need to get
                                                                  // distributed

    if (threadPool == null || threadPool instanceof FakeExecutor) {// No point
                                                                   // in
                                                                   // duplicatin
                                                                   // recources
      roundShare = roundsToDistribut;// All the rounds get shoved onto one
                                     // thread
    }

    // Random used for creating more random objects, faster to duplicate such a
    // small recourse then share and lock
    final Random rand = new Random();
    final List<Future<LearningWorker>> futures = new ArrayList<Future<LearningWorker>>(SystemInfo.LogicalCores);

    int[][] counts = null;
    AtomicDoubleArray pred = null;
    if (dataSet instanceof RegressionDataSet) {
      pred = new AtomicDoubleArray(dataSet.getSampleSize());
      counts = new int[pred.length()][1];// how many predictions are in this?
    } else {
      counts = new int[dataSet.getSampleSize()][((ClassificationDataSet) dataSet).getClassSize()];
    }

    while (roundsToDistribut > 0) {
      final int extra = extraRounds-- > 0 ? 1 : 0;
      final Future<LearningWorker> future = threadPool
          .submit(new LearningWorker(dataSet, roundShare + extra, new Random(rand.nextInt()), counts, pred));
      roundsToDistribut -= roundShare + extra;
      futures.add(future);
    }

    outOfBagError = 0;
    try {
      for (final LearningWorker worker : ListUtils.collectFutures(futures)) {
        forest.addAll(worker.learned);
      }
    } catch (final Exception ex) {
      Logger.getLogger(RandomForest.class.getName()).log(Level.SEVERE, null, ex);
    }

    if (useOutOfBagError) {
      if (dataSet instanceof ClassificationDataSet) {
        final ClassificationDataSet cds = (ClassificationDataSet) dataSet;
        for (int i = 0; i < counts.length; i++) {
          int max = 0;
          for (int j = 1; j < counts[i].length; j++) {
            if (counts[i][j] > counts[i][max]) {
              max = j;
            }
          }
          if (max != cds.getDataPointCategory(i)) {
            outOfBagError++;
          }
        }
      } else {
        final RegressionDataSet rds = (RegressionDataSet) dataSet;
        for (int i = 0; i < counts.length; i++) {
          outOfBagError += Math.pow(pred.get(i) / counts[i][0] - rds.getTargetValue(i), 2);
        }
      }
      outOfBagError /= dataSet.getSampleSize();
    }

  }
}
