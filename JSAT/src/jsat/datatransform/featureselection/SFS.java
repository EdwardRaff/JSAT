package jsat.datatransform.featureselection;

import java.util.Random;
import java.util.Set;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.datatransform.DataTransformFactoryParm;
import jsat.datatransform.RemoveAttributeTransform;
import jsat.regression.RegressionDataSet;
import jsat.regression.RegressionModelEvaluation;
import jsat.regression.Regressor;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;

/**
 * Sequential Forward Selection (SFS) is a greedy method of selecting a subset
 * of features to use for prediction. It starts from the set of no features and
 * attempts to add the next best feature to the set at each iteration.
 *
 * @author Edward Raff
 */
public class SFS implements DataTransform {

  /**
   * Factory for producing new {@link SFS} transforms
   */
  static public class SFSFactory extends DataTransformFactoryParm {

    private double maxDecrease;
    private Classifier classifier;
    private Regressor regressor;
    private int minFeatures, maxFeatures;

    /**
     * Creates a new SFS transform factory
     *
     * @param maxDecrease
     *          the maximum allowable increase in the error rate compared to the
     *          previous set of features
     * @param evaluater
     *          the classifier to use to evaluate accuracy
     * @param minFeatures
     *          the minimum number of features to learn
     * @param maxFeatures
     *          the maximum number of features to learn
     */
    public SFSFactory(final double maxDecrease, final Classifier evaluater, final int minFeatures,
        final int maxFeatures) {
      setMaxDecrease(maxDecrease);
      classifier = evaluater;
      if (evaluater instanceof Regressor) {
        regressor = (Regressor) evaluater;
      }
      setMinFeatures(minFeatures);
      setMaxFeatures(maxFeatures);
    }

    /**
     * Creates a new SFS transform factory
     *
     * @param maxDecrease
     *          the maximum allowable increase in the error rate compared to the
     *          previous set of features
     * @param evaluater
     *          the regressor to use to evaluate accuracy
     * @param minFeatures
     *          the minimum number of features to learn
     * @param maxFeatures
     *          the maximum number of features to learn
     */
    public SFSFactory(final double maxDecrease, final Regressor evaluater, final int minFeatures,
        final int maxFeatures) {
      setMaxDecrease(maxDecrease);
      regressor = evaluater;
      if (evaluater instanceof Classifier) {
        classifier = (Classifier) evaluater;
      }
      setMinFeatures(minFeatures);
      setMaxFeatures(maxFeatures);
    }

    /**
     * Copy constructor
     *
     * @param toCopy
     *          the object to copy
     */
    public SFSFactory(final SFSFactory toCopy) {
      if (toCopy.classifier == toCopy.regressor) {
        classifier = toCopy.classifier.clone();
        regressor = (Regressor) classifier;
      } else if (toCopy.classifier != null) {
        classifier = toCopy.classifier.clone();
      } else if (toCopy.regressor != null) {
        regressor = toCopy.regressor.clone();
      } else {
        throw new RuntimeException("BUG: Please report");
      }
      maxDecrease = toCopy.maxDecrease;
      minFeatures = toCopy.minFeatures;
      maxFeatures = toCopy.maxFeatures;
    }

    @Override
    public SFSFactory clone() {
      return new SFSFactory(this);
    }

    /**
     * Returns the maximum allowable decrease in accuracy from one set of
     * features to the next
     *
     * @return the maximum allowable decrease in accuracy from one set of
     *         features to the next
     */
    public double getMaxDecrease() {
      return maxDecrease;
    }

    /**
     * Returns the maximum number of features to find
     *
     * @return the maximum number of features to find
     */
    public int getMaxFeatures() {
      return maxFeatures;
    }

    /**
     * Returns the minimum number of features to find
     *
     * @return the minimum number of features to find
     */
    public int getMinFeatures() {
      return minFeatures;
    }

    @Override
    public SFS getTransform(final DataSet dataset) {
      if (dataset instanceof ClassificationDataSet) {
        return new SFS(minFeatures, maxFeatures, (ClassificationDataSet) dataset, classifier, 5, maxDecrease);
      } else {
        return new SFS(minFeatures, maxFeatures, (RegressionDataSet) dataset, regressor, 5, maxDecrease);
      }
    }

    /**
     * Sets the maximum allowable decrease in accuracy (increase in error) from
     * the previous set of features to the new current set.
     *
     * @param maxDecrease
     *          the maximum allowable decrease in the accuracy from removing a
     *          feature
     */
    public void setMaxDecrease(final double maxDecrease) {
      if (maxDecrease < 0) {
        throw new IllegalArgumentException("Decarese must be a positive value, not " + maxDecrease);
      }
      this.maxDecrease = maxDecrease;
    }

    /**
     * Sets the maximum number of features that must be selected
     *
     * @param maxFeatures
     *          the maximum number of features to find
     */
    public void setMaxFeatures(final int maxFeatures) {
      this.maxFeatures = maxFeatures;
    }

    /**
     * Sets the minimum number of features that must be selected
     *
     * @param minFeatures
     *          the minimum number of features to learn
     */
    public void setMinFeatures(final int minFeatures) {
      this.minFeatures = minFeatures;
    }

  }

  private static final long serialVersionUID = 140187978708131002L;

  /**
   *
   * @param curBest
   *          the value of curBest
   * @param nCat
   *          the value of nCat
   * @param catF
   *          the value of catF
   * @param numF
   *          the value of numF
   */
  static protected void addFeature(final int curBest, final int nCat, final Set<Integer> catF,
      final Set<Integer> numF) {
    if (curBest >= nCat) {
      numF.add(curBest - nCat);
    } else {
      catF.add(curBest);
    }
  }

  /**
   * The score function for a data set and a learner by cross validation of a
   * classifier
   *
   * @param workOn
   *          the transformed data set to test from with cross validation
   * @param evaluater
   *          the learning algorithm to use
   * @param folds
   *          the number of cross validation folds to perform
   * @param rand
   *          the source of randomness
   * @return the score value in terms of cross validated error
   */
  protected static double getScore(final DataSet workOn, final Object evaluater, final int folds, final Random rand) {
    if (workOn instanceof ClassificationDataSet) {
      final ClassificationModelEvaluation cme = new ClassificationModelEvaluation((Classifier) evaluater,
          (ClassificationDataSet) workOn);
      cme.evaluateCrossValidation(folds, rand);

      return cme.getErrorRate();
    } else if (workOn instanceof RegressionDataSet) {
      final RegressionModelEvaluation rme = new RegressionModelEvaluation((Regressor) evaluater,
          (RegressionDataSet) workOn);
      rme.evaluateCrossValidation(folds, rand);

      return rme.getMeanError();
    }
    return Double.POSITIVE_INFINITY;
  }

  /**
   *
   * @param feature
   *          the value of feature
   * @param nCat
   *          the value of nCat
   * @param catF
   *          the value of catF
   * @param numF
   *          the value of numF
   */
  static protected void removeFeature(final int feature, final int nCat, final Set<Integer> catF,
      final Set<Integer> numF) {
    if (feature >= nCat) {
      numF.remove(feature - nCat);
    } else {
      catF.remove(feature);
    }
  }

  /**
   * Attempts to add one feature to the list of features while increasing or
   * maintaining the current accuracy
   *
   * @param available
   *          the set of available features from [0, n) to consider for adding
   * @param dataSet
   *          the original data set to perform feature selection from
   * @param catToRemove
   *          the current set of categorical features to remove
   * @param numToRemove
   *          the current set of numerical features to remove
   * @param catSelecteed
   *          the current set of categorical features we are keeping
   * @param numSelected
   *          the current set of numerical features we are keeping
   * @param evaluater
   *          the classifier or regressor to perform evaluations with
   * @param folds
   *          the number of cross validation folds to determine performance
   * @param rand
   *          the source of randomness
   * @param PbestScore
   *          an array to behave as a pointer to the best score seen so far
   * @param minFeatures
   *          the minimum number of features needed
   * @return the feature that was selected to add, or -1 if none were added.
   */
  static protected int SFSSelectFeature(final Set<Integer> available, final DataSet dataSet,
      final Set<Integer> catToRemove, final Set<Integer> numToRemove, final Set<Integer> catSelecteed,
      final Set<Integer> numSelected, final Object evaluater, final int folds, final Random rand,
      final double[] PbestScore, final int minFeatures) {
    final int nCat = dataSet.getNumCategoricalVars();
    int curBest = -1;
    double curBestScore = Double.POSITIVE_INFINITY;
    for (final int feature : available) {
      removeFeature(feature, nCat, catToRemove, numToRemove);

      final DataSet workOn = dataSet.shallowClone();
      final RemoveAttributeTransform remove = new RemoveAttributeTransform(workOn, catToRemove, numToRemove);
      workOn.applyTransform(remove);

      final double score = getScore(workOn, evaluater, folds, rand);

      if (score < curBestScore) {
        curBestScore = score;
        curBest = feature;
      }
      addFeature(feature, nCat, catToRemove, numToRemove);
    }
    if (curBestScore <= 1e-14 && PbestScore[0] <= 1e-14 && catSelecteed.size() + numSelected.size() >= minFeatures) {
      return -1;
    }
    if (curBestScore < PbestScore[0] || catSelecteed.size() + numSelected.size() < minFeatures
        || Math.abs(PbestScore[0] - curBestScore) < 1e-3) {
      PbestScore[0] = curBestScore;
      addFeature(curBest, nCat, catSelecteed, numSelected);
      removeFeature(curBest, nCat, catToRemove, numToRemove);
      available.remove(curBest);
      return curBest;
    } else {
      return -1; // No possible improvment & weve got enough
    }
  }

  private RemoveAttributeTransform finalTransform;

  private Set<Integer> catSelected;

  private Set<Integer> numSelected;

  private final double maxIncrease;

  private Classifier classifier;

  private Regressor regressor;

  /**
   * Performs SFS feature selection for a classification problem
   *
   * @param minFeatures
   *          the minimum number of features to find
   * @param maxFeatures
   *          the maximum number of features to find
   * @param dataSet
   *          the data set to perform feature selection on
   * @param evaluater
   *          the classifier to use in determining accuracy given a feature
   *          subset
   * @param folds
   *          the number of cross validation folds to use in selection
   * @param maxIncrease
   *          the maximum tolerable increase in error when a feature is added
   */
  public SFS(final int minFeatures, final int maxFeatures, final ClassificationDataSet dataSet,
      final Classifier evaluater, final int folds, final double maxIncrease) {
    classifier = evaluater.clone();
    this.maxIncrease = maxIncrease;
    search(minFeatures, maxFeatures, dataSet, folds);
  }

  /**
   * Performs SFS feature selection for a regression problem
   *
   * @param minFeatures
   *          the minimum number of features to find
   * @param maxFeatures
   *          the maximum number of features to find
   * @param dataSet
   *          the data set to perform feature selection on
   * @param regressor
   *          the regressor to use in determining accuracy given a feature
   *          subset
   * @param folds
   *          the number of cross validation folds to use in selection
   * @param maxIncrease
   *          the maximum tolerable increase in error when a feature is added
   */
  public SFS(final int minFeatures, final int maxFeatures, final RegressionDataSet dataSet, final Regressor regressor,
      final int folds, final double maxIncrease) {
    this.regressor = regressor.clone();
    this.maxIncrease = maxIncrease;
    search(minFeatures, maxFeatures, dataSet, folds);
  }

  /**
   * Copy constructor
   *
   * @param toClone
   *          the SFS to copy
   */
  private SFS(final SFS toClone) {
    if (toClone.catSelected != null) {
      finalTransform = toClone.finalTransform.clone();
      catSelected = new IntSet(toClone.catSelected);
      numSelected = new IntSet(toClone.numSelected);
    }

    maxIncrease = toClone.maxIncrease;
    if (toClone.classifier != null) {
      classifier = toClone.classifier.clone();
    }
    if (toClone.regressor != null) {
      regressor = toClone.regressor.clone();
    }
  }

  @Override
  public SFS clone() {
    return new SFS(this);
  }

  /**
   * Returns a copy of the set of categorical features selected by the search
   * algorithm
   *
   * @return the set of categorical features to use
   */
  public Set<Integer> getSelectedCategorical() {
    return new IntSet(catSelected);
  }

  /**
   * Returns a copy of the set of numerical features selected by the search
   * algorithm.
   *
   * @return the set of numeric features to use
   */
  public Set<Integer> getSelectedNumerical() {
    return new IntSet(numSelected);
  }

  private void search(final int minFeatures, final int maxFeatures, final DataSet dataSet, final int folds) {
    final Random rand = new Random();
    final int nF = dataSet.getNumFeatures();
    final int nCat = dataSet.getNumCategoricalVars();

    final Set<Integer> available = new IntSet();
    ListUtils.addRange(available, 0, nF, 1);
    catSelected = new IntSet(dataSet.getNumCategoricalVars());
    numSelected = new IntSet(dataSet.getNumNumericalVars());

    final Set<Integer> catToRemove = new IntSet(dataSet.getNumCategoricalVars());
    final Set<Integer> numToRemove = new IntSet(dataSet.getNumNumericalVars());
    ListUtils.addRange(catToRemove, 0, nCat, 1);
    ListUtils.addRange(numToRemove, 0, nF - nCat, 1);

    final double[] bestScore = new double[] { Double.POSITIVE_INFINITY };

    Object learner = regressor;
    if (dataSet instanceof ClassificationDataSet) {
      learner = classifier;
    }

    while (catSelected.size() + numSelected.size() < maxFeatures) {
      if (SFSSelectFeature(available, dataSet, catToRemove, numToRemove, catSelected, numSelected, learner, folds, rand,
          bestScore, minFeatures) < 0) {
        break;
      }

    }

    finalTransform = new RemoveAttributeTransform(dataSet, catToRemove, numToRemove);
  }

  @Override
  public DataPoint transform(final DataPoint dp) {
    return finalTransform.transform(dp);
  }
}
