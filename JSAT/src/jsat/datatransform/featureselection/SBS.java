package jsat.datatransform.featureselection;

import static jsat.datatransform.featureselection.SFS.addFeature;
import static jsat.datatransform.featureselection.SFS.removeFeature;

import java.util.Arrays;
import java.util.Random;
import java.util.Set;

import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.datatransform.DataTransformFactoryParm;
import jsat.datatransform.RemoveAttributeTransform;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.IntList;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;

/**
 * Sequential Backward Selection (SBS) is a greedy method of selecting a subset
 * of features to use for prediction. It starts from the set of all features and
 * attempts to remove the least informative feature from the set at each
 * iteration
 *
 * @author Edward Raff
 */
public class SBS extends RemoveAttributeTransform {

  /**
   * Factory for producing new {@link SBS} transforms
   */
  static public class SBSFactory extends DataTransformFactoryParm {

    private double maxDecrease;
    private Classifier classifier;
    private Regressor regressor;
    private int minFeatures, maxFeatures;

    /**
     * Creates a new SBS transform factory
     *
     * @param maxDecrease
     *          the maximum allowable decrease in the accuracy rate compared to
     *          the previous set of features
     * @param evaluater
     *          the classifier to use to evaluate accuracy
     * @param minFeatures
     *          the minimum number of features to learn
     * @param maxFeatures
     *          the maximum number of features to learn
     */
    public SBSFactory(final double maxDecrease, final Classifier evaluater, final int minFeatures,
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
     * Creates a new SBS transform factory
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
    public SBSFactory(final double maxDecrease, final Regressor evaluater, final int minFeatures,
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
    public SBSFactory(final SBSFactory toCopy) {
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
    public SBSFactory clone() {
      return new SBSFactory(this);
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
    public SBS getTransform(final DataSet dataset) {
      if (dataset instanceof ClassificationDataSet) {
        return new SBS(minFeatures, maxFeatures, (ClassificationDataSet) dataset, classifier, 5, maxDecrease);
      } else {
        return new SBS(minFeatures, maxFeatures, (RegressionDataSet) dataset, regressor, 5, maxDecrease);
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

  private static final long serialVersionUID = -2516121100148559742L;

  /**
   * Attempts to remove one feature from the list while maintaining its accuracy
   *
   * @param available
   *          the set of available features from [0, n) to consider for removal
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
   * @param maxFeatures
   *          the maximum allowable number of features
   * @param PbestScore
   *          an array to behave as a pointer to the best score seen so far
   * @param maxDecrease
   *          the maximum allowable decrease in accuracy from the best accuracy
   *          we see
   * @return the feature that was selected to be removed, or -1 if none were
   *         removed
   */
  protected static int SBSRemoveFeature(final Set<Integer> available, final DataSet dataSet,
      final Set<Integer> catToRemove, final Set<Integer> numToRemove, final Set<Integer> catSelecteed,
      final Set<Integer> numSelected, final Object evaluater, final int folds, final Random rand, final int maxFeatures,
      final double[] PbestScore, final double maxDecrease) {
    int curBest = -1;
    final int nCat = dataSet.getNumCategoricalVars();
    double curBestScore = Double.POSITIVE_INFINITY;
    for (final int feature : available) {
      final DataSet workOn = dataSet.shallowClone();
      addFeature(feature, nCat, catToRemove, numToRemove);

      final RemoveAttributeTransform remove = new RemoveAttributeTransform(workOn, catToRemove, numToRemove);
      workOn.applyTransform(remove);

      final double score = SFS.getScore(workOn, evaluater, folds, rand);

      if (score < curBestScore) {
        curBestScore = score;
        curBest = feature;
      }
      removeFeature(feature, nCat, catToRemove, numToRemove);
    }
    if (catSelecteed.size() + numSelected.size() > maxFeatures || PbestScore[0] - curBestScore > -maxDecrease) {
      PbestScore[0] = curBestScore;
      removeFeature(curBest, nCat, catSelecteed, numSelected);
      addFeature(curBest, nCat, catToRemove, numToRemove);
      available.remove(curBest);
      return curBest;
    } else {
      return -1; // No possible improvment & weve got enough
    }
  }

  private final double maxDecrease;

  /**
   * Performs SBS feature selection for a classification problem
   *
   * @param minFeatures
   *          the minimum number of features to find
   * @param maxFeatures
   *          the maximum number of features to find
   * @param cds
   *          the data set to perform feature selection on
   * @param evaluater
   *          the classifier to use in determining accuracy given a feature
   *          subset
   * @param folds
   *          the number of cross validation folds to use in selection
   * @param maxDecrease
   *          the maximum tolerable decrease in accuracy in accuracy when a
   *          feature is removed
   */
  public SBS(final int minFeatures, final int maxFeatures, final ClassificationDataSet cds, final Classifier evaluater,
      final int folds, final double maxDecrease) {
    this.maxDecrease = maxDecrease;
    search(cds, evaluater, minFeatures, maxFeatures, folds);
  }

  /**
   * Performs SBS feature selection for a regression problem
   *
   * @param minFeatures
   *          the minimum number of features to find
   * @param maxFeatures
   *          the maximum number of features to find
   * @param rds
   *          the data set to perform feature selection on
   * @param folds
   *          the number of cross validation folds to use in selection
   * @param maxDecrease
   *          the maximum tolerable increase in the error rate when a feature is
   *          removed
   */
  public SBS(final int minFeatures, final int maxFeatures, final RegressionDataSet rds, final Regressor evaluater,
      final int folds, final double maxDecrease) {
    this.maxDecrease = maxDecrease;
    search(rds, evaluater, minFeatures, maxFeatures, folds);
  }

  /**
   * Copy constructor
   *
   * @param toClone
   *          the version to copy
   */
  private SBS(final SBS toClone) {
    super(toClone);
    maxDecrease = toClone.maxDecrease;
  }

  @Override
  public SBS clone() {
    return new SBS(this);
  }

  /**
   * Returns a copy of the set of categorical features selected by the search
   * algorithm
   *
   * @return the set of categorical features to use
   */
  public Set<Integer> getSelectedCategorical() {
    return new IntSet(IntList.view(catIndexMap, catIndexMap.length));
  }

  /**
   * Returns a copy of the set of numerical features selected by the search
   * algorithm.
   *
   * @return the set of numeric features to use
   */
  public Set<Integer> getSelectedNumerical() {
    return new IntSet(IntList.view(numIndexMap, numIndexMap.length));
  }

  private void search(final DataSet dataSet, final Object learner, final int minFeatures, final int maxFeatures,
      final int folds) {
    final Random rand = new Random();
    final int nF = dataSet.getNumFeatures();
    final int nCat = dataSet.getNumCategoricalVars();

    final Set<Integer> available = new IntSet();
    ListUtils.addRange(available, 0, nF, 1);
    final Set<Integer> catSelected = new IntSet(dataSet.getNumCategoricalVars());
    final Set<Integer> numSelected = new IntSet(dataSet.getNumNumericalVars());

    final Set<Integer> catToRemove = new IntSet(dataSet.getNumCategoricalVars());
    final Set<Integer> numToRemove = new IntSet(dataSet.getNumNumericalVars());

    // Start will all selected, and prune them out
    ListUtils.addRange(catSelected, 0, nCat, 1);
    ListUtils.addRange(numSelected, 0, nF - nCat, 1);

    final double[] bestScore = new double[] { Double.POSITIVE_INFINITY };

    while (catSelected.size() + numSelected.size() > minFeatures) {

      if (SBSRemoveFeature(available, dataSet, catToRemove, numToRemove, catSelected, numSelected, learner, folds, rand,
          maxFeatures, bestScore, maxDecrease) < 0) {
        break;
      }

    }

    int pos = 0;
    catIndexMap = new int[catSelected.size()];
    for (final int i : catSelected) {
      catIndexMap[pos++] = i;
    }
    Arrays.sort(catIndexMap);

    pos = 0;
    numIndexMap = new int[numSelected.size()];
    for (final int i : numSelected) {
      numIndexMap[pos++] = i;
    }
    Arrays.sort(numIndexMap);
  }
}
