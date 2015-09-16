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
 * plus-L minus-R Selection (LRS) is a greedy method of selecting a subset of
 * features to use for prediction. Its behavior is dependent upon whether L or R
 * is the larger value. No mater what, L features will be greedily added to the
 * set to decrease the error rate, and R features will be greedily removed while
 * trying to maintain the error rate. <br>
 * If L &gt; R, then L-R features will be selected, the L step running first
 * followed by R performing pruning on the found set. <br>
 * If L &lt; R, then D-R+L features will be selected, where D is the original
 * number of features. First R features will be removed, and then L of the
 * removed features will be added back to the final set. <br>
 * L = R is not allowed.
 *
 * @author Edward Raff
 */
public class LRS implements DataTransform {

  /**
   * Factory for producing new {@link LRS} transforms.
   */
  static public class LRSFactory extends DataTransformFactoryParm {

    private Classifier classifier;
    private Regressor regressor;
    private int featuresToAdd, featuresToRemove;

    /**
     * Creates a new LRS transform factory
     *
     * @param evaluater
     *          the classifier to use to perform evaluation
     * @param toAdd
     *          the number of features to add
     * @param toRemove
     *          the number of features to remove
     */
    public LRSFactory(final Classifier evaluater, final int toAdd, final int toRemove) {
      if (toAdd == toRemove) {
        throw new RuntimeException("L and R must be different");
      }
      classifier = evaluater;
      if (evaluater instanceof Regressor) {
        regressor = (Regressor) evaluater;
      }
      setFeaturesToAdd(toAdd);
      setFeaturesToRemove(toRemove);
    }

    /**
     * Copy constructor
     *
     * @param toCopy
     *          the object to copy
     */
    public LRSFactory(final LRSFactory toCopy) {
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
      featuresToAdd = toCopy.featuresToAdd;
      featuresToRemove = toCopy.featuresToRemove;
    }

    /**
     * Creates a new LRS transform factory
     *
     * @param evaluater
     *          the regressor to use to perform evaluation
     * @param toAdd
     *          the number of features to add
     * @param toRemove
     *          the number of features to remove
     */
    public LRSFactory(final Regressor evaluater, final int toAdd, final int toRemove) {
      if (toAdd == toRemove) {
        throw new RuntimeException("L and R must be different");
      }
      regressor = evaluater;
      if (evaluater instanceof Classifier) {
        classifier = (Classifier) evaluater;
      }
      setFeaturesToAdd(toAdd);
      setFeaturesToRemove(toRemove);
    }

    @Override
    public LRSFactory clone() {
      return new LRSFactory(this);
    }

    /**
     * Returns the number of features to add
     *
     * @return the number of features to add
     */
    public int getFeaturesToAdd() {
      return featuresToAdd;
    }

    /**
     * Returns the number of features to remove
     *
     * @return the number of features to remove
     */
    public int getFeaturesToRemove() {
      return featuresToRemove;
    }

    @Override
    public LRS getTransform(final DataSet dataset) {
      if (dataset instanceof ClassificationDataSet) {
        return new LRS(featuresToAdd, featuresToRemove, (ClassificationDataSet) dataset, classifier, 5);
      } else {
        return new LRS(featuresToAdd, featuresToRemove, (RegressionDataSet) dataset, regressor, 5);
      }
    }

    /**
     * Sets the number of features to add (the L parameter).<br>
     * <b>NOTE:</b> setting this and {@link #setFeaturesToRemove(int) } is
     * allowed for the Factory, but is is assumed that it is occurring because
     * you are about to change the value of the other. Attempting to obtain a
     * {@link LRS} transform will result in a runtime exception until one of the
     * values is changed.
     *
     * @param featuresToAdd
     *          the number of features to greedily add
     */
    public void setFeaturesToAdd(final int featuresToAdd) {
      if (featuresToAdd < 1) {
        throw new IllegalArgumentException("Number of features to add must be positive, not " + featuresToAdd);
      }
      this.featuresToAdd = featuresToAdd;
    }

    /**
     * Sets the number of features to remove (the R parameter).<br>
     * <b>NOTE:</b> setting this and {@link #setFeaturesToAdd(int) } is allowed
     * for the Factory, but is is assumed that it is occurring because you are
     * about to change the value of the other. Attempting to obtain a
     * {@link LRS} transform will result in a runtime exception until one of the
     * values is changed.
     *
     * @param featuresToRemove
     *          the number of features to greedily remove
     */
    public void setFeaturesToRemove(final int featuresToRemove) {
      if (featuresToRemove < 1) {
        throw new IllegalArgumentException("Number of features to remove must be positive, not " + featuresToRemove);
      }
      this.featuresToRemove = featuresToRemove;
    }

  }

  private static final long serialVersionUID = 3065300352046535656L;
  private RemoveAttributeTransform finalTransform;
  private Set<Integer> catSelected;

  private Set<Integer> numSelected;

  /**
   * Performs LRS feature selection for a classification problem
   *
   * @param L
   *          the number of features to greedily add
   * @param R
   *          the number of features to greedily remove
   * @param cds
   *          the data set to perform feature selection on
   * @param evaluater
   *          the classifier to use in determining accuracy given a feature
   *          subset
   * @param folds
   *          the number of cross validation folds to use in selection
   */
  public LRS(final int L, final int R, final ClassificationDataSet cds, final Classifier evaluater, final int folds) {
    search(cds, L, R, evaluater, folds);
  }

  /**
   * Performs LRS feature selection for a regression problem
   *
   * @param L
   *          the number of features to greedily add
   * @param R
   *          the number of features to greedily remove
   * @param rds
   *          the data set to perform feature selection on
   * @param evaluater
   *          the regressor to use in determining accuracy given a feature
   *          subset
   * @param folds
   *          the number of cross validation folds to use in selection
   */
  public LRS(final int L, final int R, final RegressionDataSet rds, final Regressor evaluater, final int folds) {
    search(rds, L, R, evaluater, folds);
  }

  /**
   * Copy constructor
   *
   * @param toClone
   *          the version to copy
   */
  private LRS(final LRS toClone) {
    if (toClone.catSelected != null) {
      finalTransform = toClone.finalTransform.clone();
      catSelected = new IntSet(toClone.catSelected);
      numSelected = new IntSet(toClone.numSelected);
    }
  }

  @Override
  public LRS clone() {
    return new LRS(this);
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

  private void search(final DataSet cds, final int L, final int R, final Object evaluater, final int folds) {
    final int nF = cds.getNumFeatures();
    final int nCat = cds.getNumCategoricalVars();

    catSelected = new IntSet(nCat);
    numSelected = new IntSet(nF - nCat);
    final Set<Integer> catToRemove = new IntSet(nCat);
    final Set<Integer> numToRemove = new IntSet(nF - nCat);

    final Set<Integer> available = new IntSet(nF);
    ListUtils.addRange(available, 0, nF, 1);

    final Random rand = new Random();
    final double[] pBestScore = new double[] { Double.POSITIVE_INFINITY };

    if (L > R) {
      ListUtils.addRange(catToRemove, 0, nCat, 1);
      ListUtils.addRange(numToRemove, 0, nF - nCat, 1);

      // Select L features
      for (int i = 0; i < L; i++) {
        SFS.SFSSelectFeature(available, cds, catToRemove, numToRemove, catSelected, numSelected, evaluater, folds, rand,
            pBestScore, L);
      }
      // We now restrict ourselves to the L features
      available.clear();
      available.addAll(catSelected);
      for (final int i : numSelected) {
        available.add(i + nCat);
      }
      // Now remove R features from the L selected
      for (int i = 0; i < R; i++) {
        SBS.SBSRemoveFeature(available, cds, catToRemove, numToRemove, catSelected, numSelected, evaluater, folds, rand,
            L - R, pBestScore, 0.0);
      }
    } else if (L < R) {
      ListUtils.addRange(catSelected, 0, nCat, 1);
      ListUtils.addRange(numSelected, 0, nF - nCat, 1);

      // Remove R features
      for (int i = 0; i < R; i++) {
        SBS.SBSRemoveFeature(available, cds, catToRemove, numToRemove, catSelected, numSelected, evaluater, folds, rand,
            nF - R, pBestScore, 0.0);
      }

      // Now we restrict out selves to adding back the features that were
      // removed
      available.clear();
      available.addAll(catToRemove);
      for (final int i : numToRemove) {
        available.add(i + nCat);
      }

      // Now add L features back
      for (int i = 0; i < L; i++) {
        SFS.SFSSelectFeature(available, cds, catToRemove, numToRemove, catSelected, numSelected, evaluater, folds, rand,
            pBestScore, R - L);
      }
    }

    finalTransform = new RemoveAttributeTransform(cds, catToRemove, numToRemove);
  }

  @Override
  public DataPoint transform(final DataPoint dp) {
    return finalTransform.transform(dp);
  }
}
