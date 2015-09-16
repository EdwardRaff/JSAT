package jsat.datatransform.featureselection;

import static java.lang.Math.log;
import java.util.Set;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransformFactoryParm;
import jsat.datatransform.RemoveAttributeTransform;
import jsat.exceptions.FailedToFitException;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.utils.IndexTable;
import jsat.utils.IntSet;

/**
 * Performs greedy feature selection based on Mutual Information of the features
 * with respect to the class values. This is an attempt to select features that
 * are discriminative for classification tasks. <br>
 * The method of performing Mutual Information on numeric attributes is
 * controlled by {@link NumericalHandeling}
 *
 * @author Edward Raff
 */
public class MutualInfoFS extends RemoveAttributeTransform {

  /**
   * Factory for producing {@link MutualInfoFS} transforms
   */
  public static class MutualInfoFSFactory extends DataTransformFactoryParm {

    private int featureCount;
    private NumericalHandeling handling;

    /**
     * Creates a new MutalInfoFS factory that uses
     * {@link NumericalHandeling#BINARY} handling for numeric attributes.
     *
     * @param featureCount
     *          the number of features to select
     */
    public MutualInfoFSFactory(final int featureCount) {
      this(featureCount, NumericalHandeling.BINARY);
    }

    /**
     * Creates a new MutualInfoFS factory
     *
     * @param featureCount
     *          the number of features to select
     * @param handling
     *          the way to handle numeric attributes
     */
    public MutualInfoFSFactory(final int featureCount, final NumericalHandeling handling) {
      setFeatureCount(featureCount);
      setHandling(handling);
    }

    /**
     * Copy constructor
     *
     * @param toCopy
     *          the object to copy
     */
    public MutualInfoFSFactory(final MutualInfoFSFactory toCopy) {
      this(toCopy.featureCount, toCopy.handling);
    }

    @Override
    public MutualInfoFSFactory clone() {
      return new MutualInfoFSFactory(this);
    }

    /**
     * Returns the number of features to select
     *
     * @return the number of features to select
     */
    public int getFeatureCount() {
      return featureCount;
    }

    /**
     * Returns the method of handling numeric features
     *
     * @return the method of handling numeric features
     */
    public NumericalHandeling getHandling() {
      return handling;
    }

    @Override
    public MutualInfoFS getTransform(final DataSet dataset) {
      if (!(dataset instanceof ClassificationDataSet)) {
        throw new FailedToFitException("The given data set was not a classification data set");
      }
      final ClassificationDataSet cds = (ClassificationDataSet) dataset;
      return new MutualInfoFS(cds, featureCount, handling);
    }

    /**
     * Sets the number of features to select
     *
     * @param featureCount
     *          the number of features to select
     */
    public void setFeatureCount(final int featureCount) {
      if (featureCount < 1) {
        throw new IllegalArgumentException("Number of features must be positive, not " + featureCount);
      }
      this.featureCount = featureCount;
    }

    /**
     * Sets the method of handling numeric features
     *
     * @param handling
     *          the numeric handling
     */
    public void setHandling(final NumericalHandeling handling) {
      this.handling = handling;
    }
  }

  /**
   * The definition for mutual information for continuous attributes requires an
   * integration of an unknown function, as such requires some form of
   * approximation. This controls how the approximation is done
   */
  public enum NumericalHandeling {
    /**
     * Mutual information for numeric attributes is not computed, so no numeric
     * attributes will be removed - and are ignored completely. The number of
     * features to select does not include the numeric attributes in this case.
     */
    NONE, /**
           * Numeric attributes are treated as nominal features with binary
           * values. The false value is if the value is zero, and the true value
           * is any non zero value.
           */
    BINARY,
  }

  private static final long serialVersionUID = -4394620220403363542L;

  /**
   * Creates a new Mutual Information feature selection object. Numeric
   * attributes are handled by {@link NumericalHandeling#BINARY}
   *
   * @param dataSet
   *          the classification data set to perform feature selection from
   * @param featureCount
   *          the number of features to select
   */
  public MutualInfoFS(final ClassificationDataSet dataSet, final int featureCount) {
    this(dataSet, featureCount, NumericalHandeling.BINARY);
  }

  /**
   * Creates a new Mutual Information feature selection object.
   *
   * @param dataSet
   *          the classification data set to perform feature selection from
   * @param featureCount
   *          the number of features to select
   * @param numericHandling
   *          the way to handle the computation of mutual information for
   *          numeric attributes
   */
  public MutualInfoFS(final ClassificationDataSet dataSet, final int featureCount,
      final NumericalHandeling numericHandling) {
    super();
    if (featureCount <= 0) {
      throw new RuntimeException("Number of features to select must be positive");
    }
    final int N = dataSet.getSampleSize();
    final double[] classPriors = dataSet.getPriors();
    final double[] logClassPriors = new double[classPriors.length];
    for (int i = 0; i < logClassPriors.length; i++) {
      logClassPriors[i] = log(classPriors[i]);
    }

    int numCatVars;
    int consideredCount = numCatVars = dataSet.getNumCategoricalVars();
    if (numericHandling != NumericalHandeling.NONE) {
      consideredCount = dataSet.getNumFeatures();
    }

    /**
     * 1st index is the feature 2nd index is the option #
     */
    final double[][] featPriors = new double[consideredCount][];

    final CategoricalData[] catInfo = dataSet.getCategories();

    /**
     * 1st index is the feature 2nd index is the option # 3rd index is the class
     */
    final double[][][] jointProb = new double[consideredCount][][];
    for (int i = 0; i < jointProb.length; i++) {
      if (i < dataSet.getNumCategoricalVars()) // Cat value
      {
        final int options = catInfo[i].getNumOfCategories();
        jointProb[i] = new double[options][logClassPriors.length];
        featPriors[i] = new double[options];
      } else// Numeric value
      {
        // Yes/No, but only keep track of the yes values
        jointProb[i] = new double[2][logClassPriors.length];
        featPriors[i] = new double[1];// feature for numeric is just 1.0-other
      }
    }

    double weightSum = 0.0;
    for (int i = 0; i < dataSet.getSampleSize(); i++) {
      final DataPoint dp = dataSet.getDataPoint(i);
      final int trueClass = dataSet.getDataPointCategory(i);
      final double weight = dp.getWeight();
      weightSum += weight;

      final int[] catVals = dp.getCategoricalValues();
      for (int j = 0; j < catVals.length; j++) {
        featPriors[j][catVals[j]] += weight;
        jointProb[j][catVals[j]][trueClass] += weight;
      }

      if (numericHandling == NumericalHandeling.BINARY) {
        final Vec numeric = dp.getNumericalValues();

        for (final IndexValue iv : numeric) {
          featPriors[iv.getIndex() + numCatVars][0] += weight;
          jointProb[iv.getIndex() + numCatVars][0][trueClass] += weight;
        }
      }
    }

    /**
     * Mutual Information for each index
     */
    final double[] mis = new double[consideredCount];

    for (int i = 0; i < consideredCount; i++) {
      double mi = 0.0;
      if (i < dataSet.getNumCategoricalVars()) // Cat attribute
      {
        for (int tVal = 0; tVal < jointProb[i].length; tVal++) {
          final double featPrior = featPriors[i][tVal] / weightSum;
          if (featPrior == 0.0) {
            continue;
          }
          final double logFeatPrior = log(featPrior);
          for (int tClass = 0; tClass < logClassPriors.length; tClass++) {
            final double jp = jointProb[i][tVal][tClass] / weightSum;
            if (jp == 0) {
              continue;
            }
            mi += jp * (log(jp) - logFeatPrior - logClassPriors[tClass]);
          }
        }
      } else// Numeric attribute & it is binary
      {
        for (int tClass = 0; tClass < classPriors.length; tClass++) {
          final double jpNeg = jointProb[i][0][tClass] / weightSum;
          final double jpPos = (classPriors[tClass] * N - jointProb[i][0][tClass]) / weightSum;
          final double posPrio = featPriors[i][0] / weightSum;
          final double negPrio = 1.0 - posPrio;

          if (jpNeg != 0 && negPrio != 0) {
            mi += jpNeg * (log(jpNeg) - log(negPrio) - logClassPriors[tClass]);
          }
          if (jpPos != 0 && posPrio != 0) {
            mi += jpPos * (log(jpPos) - log(posPrio) - logClassPriors[tClass]);
          }
        }
      }
      mis[i] = mi;
    }

    final IndexTable sortedOrder = new IndexTable(mis);

    final Set<Integer> catToRemove = new IntSet();
    final Set<Integer> numToRemove = new IntSet();

    for (int i = 0; i < consideredCount - featureCount; i++) {
      final int removingIndex = sortedOrder.index(i);
      if (removingIndex < numCatVars) {
        catToRemove.add(removingIndex);
      } else {
        numToRemove.add(removingIndex - numCatVars);
      }
    }

    setUp(dataSet, catToRemove, numToRemove);
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  protected MutualInfoFS(final MutualInfoFS toCopy) {
    super(toCopy);
  }

  @Override
  public MutualInfoFS clone() {
    return new MutualInfoFS(this);
  }
}
