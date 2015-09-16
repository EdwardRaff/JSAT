package jsat.datatransform.featureselection;

import java.util.Random;
import java.util.Set;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.datatransform.DataTransformFactoryParm;
import jsat.datatransform.RemoveAttributeTransform;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;

/**
 * Bidirectional Search (BDS) is a greedy method of selecting a subset of
 * features to use for prediction. It performs both {@link SFS} and {@link SBS}
 * search at the same time. At each step, a feature is greedily added to one
 * set, and then a feature greedily removed from another set. Once a feature is
 * added / removed in one set, it is unavailable for selection in the other.
 * This can be used to select up to half of the original features.
 *
 * @author Edward Raff
 */
public class BDS implements DataTransform {

  /**
   * Factory for producing new {@link BDS} transforms.
   */
  static public class BDSFactory extends DataTransformFactoryParm {

    private Classifier classifier;
    private Regressor regressor;
    private int featureCount;

    /**
     * Copy constructor
     *
     * @param toCopy
     *          the object to copy
     */
    public BDSFactory(final BDSFactory toCopy) {
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
      featureCount = toCopy.featureCount;
    }

    /**
     * Creates a new BDS factory
     *
     * @param evaluater
     *          the classifier to use in determining accuracy given a feature
     *          subset
     * @param featureCount
     *          the number of features to select
     */
    public BDSFactory(final Classifier evaluater, final int featureCount) {
      classifier = evaluater;
      if (evaluater instanceof Regressor) {
        regressor = (Regressor) evaluater;
      }
      setFeatureCount(featureCount);
    }

    /**
     * Creates a new BDS factory
     *
     * @param evaluater
     *          the regressor to use in determining accuracy given a feature
     *          subset
     * @param featureCount
     *          the number of features to select
     */
    public BDSFactory(final Regressor evaluater, final int featureCount) {
      regressor = evaluater;
      if (evaluater instanceof Classifier) {
        classifier = (Classifier) evaluater;
      }
      setFeatureCount(featureCount);
    }

    @Override
    public BDSFactory clone() {
      return new BDSFactory(this);
    }

    /**
     * Returns the number of features to sue
     *
     * @return the number of features to sue
     */
    public int getFeatureCount() {
      return featureCount;
    }

    @Override
    public BDS getTransform(final DataSet dataset) {
      if (dataset instanceof ClassificationDataSet) {
        return new BDS(featureCount, (ClassificationDataSet) dataset, classifier, 5);
      } else {
        return new BDS(featureCount, (RegressionDataSet) dataset, regressor, featureCount);
      }
    }

    /**
     * Sets the number of features to select for use from the set of all input
     * features
     *
     * @param featureCount
     *          the number of features to use
     */
    public void setFeatureCount(final int featureCount) {
      if (featureCount < 1) {
        throw new IllegalArgumentException("Number of features to select must be positive, not " + featureCount);
      }
      this.featureCount = featureCount;
    }

  }

  private static final long serialVersionUID = 8633823674617843754L;
  private RemoveAttributeTransform finalTransform;
  private Set<Integer> catSelected;

  private Set<Integer> numSelected;

  /**
   * Copy constructor
   *
   * @param toClone
   */
  public BDS(final BDS toClone) {
    if (toClone.finalTransform != null) {
      finalTransform = toClone.finalTransform.clone();
      catSelected = new IntSet(toClone.catSelected);
      numSelected = new IntSet(toClone.numSelected);
    }
  }

  /**
   * Performs BDS feature selection for a classification problem
   *
   * @param featureCount
   *          the number of features to select
   * @param dataSet
   *          the data set to perform feature selection on
   * @param evaluator
   *          the classifier to use in determining accuracy given a feature
   *          subset
   * @param folds
   *          the number of cross validation folds to use in selection
   */
  public BDS(final int featureCount, final ClassificationDataSet dataSet, final Classifier evaluator, final int folds) {
    search(dataSet, featureCount, folds, evaluator);
  }

  /**
   * Performs BDS feature selection for a regression problem
   *
   * @param featureCount
   *          the number of features to select
   * @param dataSet
   *          the data set to perform feature selection on
   * @param evaluator
   *          the regressor to use in determining accuracy given a feature
   *          subset
   * @param folds
   *          the number of cross validation folds to use in selection
   */
  public BDS(final int featureCount, final RegressionDataSet dataSet, final Regressor evaluator, final int folds) {
    search(dataSet, featureCount, folds, evaluator);
  }

  @Override
  public BDS clone() {
    return new BDS(this);
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

  private void search(final DataSet dataSet, final int maxFeatures, final int folds, final Object evaluator) {
    final Random rand = new Random();
    final int nF = dataSet.getNumFeatures();
    final int nCat = dataSet.getNumCategoricalVars();

    // True selected, also used for SFS
    catSelected = new IntSet(dataSet.getNumCategoricalVars());
    numSelected = new IntSet(dataSet.getNumNumericalVars());

    // Structs for SFS side
    final Set<Integer> availableSFS = new IntSet();
    ListUtils.addRange(availableSFS, 0, nF, 1);

    final Set<Integer> catToRemoveSFS = new IntSet(dataSet.getNumCategoricalVars());
    final Set<Integer> numToRemoveSFS = new IntSet(dataSet.getNumNumericalVars());
    ListUtils.addRange(catToRemoveSFS, 0, nCat, 1);
    ListUtils.addRange(numToRemoveSFS, 0, nF - nCat, 1);

    /// Structes fro SBS side
    final Set<Integer> availableSBS = new IntSet();
    ListUtils.addRange(availableSBS, 0, nF, 1);
    final Set<Integer> catSelecteedSBS = new IntSet(dataSet.getNumCategoricalVars());
    final Set<Integer> numSelectedSBS = new IntSet(dataSet.getNumNumericalVars());

    final Set<Integer> catToRemoveSBS = new IntSet(dataSet.getNumCategoricalVars());
    final Set<Integer> numToRemoveSBS = new IntSet(dataSet.getNumNumericalVars());

    // Start will all selected, and prune them out
    ListUtils.addRange(catSelecteedSBS, 0, nCat, 1);
    ListUtils.addRange(numSelectedSBS, 0, nF - nCat, 1);

    final double[] pBestScore0 = new double[] { Double.POSITIVE_INFINITY };
    final double[] pBestScore1 = new double[] { Double.POSITIVE_INFINITY };
    final int max = Math.min(maxFeatures, nF / 2);
    for (int i = 0; i < max; i++) {
      // Find and keep one good one
      final int mustKeep = SFS.SFSSelectFeature(availableSFS, dataSet, catToRemoveSFS, numToRemoveSFS, catSelected,
          numSelected, evaluator, folds, rand, pBestScore0, max);
      availableSBS.remove(mustKeep);
      SFS.removeFeature(mustKeep, nCat, catToRemoveSBS, numToRemoveSBS);

      // Find and remove one bad one
      final int mustRemove = SBS.SBSRemoveFeature(availableSBS, dataSet, catToRemoveSBS, numToRemoveSBS,
          catSelecteedSBS, numSelectedSBS, evaluator, folds, rand, max, pBestScore1, 0.0);
      availableSFS.remove(mustRemove);
      SFS.addFeature(mustRemove, nCat, catToRemoveSFS, numToRemoveSFS);
    }

    catSelecteedSBS.clear();
    numToRemoveSBS.clear();
    ListUtils.addRange(catSelecteedSBS, 0, nCat, 1);
    ListUtils.addRange(numSelectedSBS, 0, nF - nCat, 1);

    catSelecteedSBS.removeAll(catSelected);
    numSelectedSBS.removeAll(numSelected);

    finalTransform = new RemoveAttributeTransform(dataSet, catSelecteedSBS, numSelectedSBS);
  }

  @Override
  public DataPoint transform(final DataPoint dp) {
    return finalTransform.transform(dp);
  }
}
