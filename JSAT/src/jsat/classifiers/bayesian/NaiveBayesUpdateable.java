package jsat.classifiers.bayesian;

import static java.lang.Math.exp;

import java.util.Arrays;

import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import jsat.distributions.Normal;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.math.MathTricks;
import jsat.math.OnLineStatistics;

/**
 * An implementation of Gaussian Naive Bayes that can be updated in an online
 * fashion. The ability to be updated comes at the cost to slower training and
 * classification time. However NB is already very fast, so the difference is
 * not significant. <br>
 * The more advanced distribution detection of {@link NaiveBayes} is not
 * possible in online form.
 *
 * @author Edward Raff
 */
public class NaiveBayesUpdateable extends BaseUpdateableClassifier {

  private static final long serialVersionUID = 1835073945715343486L;
  /**
   * Counts for each option
   */
  private double[][][] apriori;
  /**
   * Online stats about the variable values
   */
  private OnLineStatistics[][] valueStats;

  private double priorSum = 0;
  private double[] priors;

  /**
   * Handles how vectors are handled. If true, it is assumed vectors are sparce
   * - and zero values will be ignored when training and classifying.
   */
  private boolean sparseInput = true;

  /**
   * Creates a new Naive Bayes classifier that assumes sparce input vectors
   */
  public NaiveBayesUpdateable() {
    this(true);
  }

  /**
   * Creates a new Naive Bayes classifier
   *
   * @param sparse
   *          whether or not to assume input vectors are sparce
   */
  public NaiveBayesUpdateable(final boolean sparse) {
    setSparse(sparse);
  }

  /**
   * Copy Constructor
   *
   * @param other
   *          the classifier to make a copy of
   */
  protected NaiveBayesUpdateable(final NaiveBayesUpdateable other) {
    this(other.sparseInput);
    if (other.apriori != null) {
      apriori = new double[other.apriori.length][][];
      valueStats = new OnLineStatistics[other.valueStats.length][];

      for (int i = 0; i < other.apriori.length; i++) {
        apriori[i] = new double[other.apriori[i].length][];
        for (int j = 0; j < other.apriori[i].length; j++) {
          apriori[i][j] = Arrays.copyOf(other.apriori[i][j], other.apriori[i][j].length);
        }
        valueStats[i] = new OnLineStatistics[other.valueStats[i].length];

        for (int j = 0; j < valueStats[i].length; j++) {
          valueStats[i][j] = new OnLineStatistics(other.valueStats[i][j]);
        }
      }

      priorSum = other.priorSum;
      priors = Arrays.copyOf(other.priors, other.priors.length);

    }
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    if (apriori == null) {
      throw new UntrainedModelException("Model has not been intialized");
    }
    final CategoricalResults results = new CategoricalResults(apriori.length);
    final double[] logProbs = new double[apriori.length];
    double maxLogProg = Double.NEGATIVE_INFINITY;

    final Vec numVals = data.getNumericalValues();
    for (int i = 0; i < valueStats.length; i++) {
      double logProb = 0;
      if (sparseInput) {
        for (final IndexValue iv : numVals) {
          final int indx = iv.getIndex();
          final double mean = valueStats[i][indx].getMean();
          final double stndDev = valueStats[i][indx].getStandardDeviation();
          final double logPDF = Normal.logPdf(iv.getValue(), mean, stndDev);
          if (Double.isNaN(logPDF)) {
            logProb += Math.log(1e-16);
          } else if (Double.isInfinite(logPDF)) {// Avoid propigation -infinty
                                                 // when the probability is zero
            logProb += Math.log(1e-16);
          } else {
            logProb += logPDF;
          }
        }
      } else {
        for (int j = 0; j < valueStats[i].length; j++) {
          final double mean = valueStats[i][j].getMean();
          final double stdDev = valueStats[i][j].getStandardDeviation();
          final double logPDF = Normal.logPdf(numVals.get(j), mean, stdDev);
          if (Double.isInfinite(logPDF)) {// Avoid propigation -infinty when the
                                          // probability is zero
            logProb += Math.log(1e-16);//
          } else {
            logProb += logPDF;
          }
        }
      }

      // the i goes up to the number of categories, same for aprioror
      for (int j = 0; j < apriori[i].length; j++) {
        double sum = 0;
        for (int z = 0; z < apriori[i][j].length; z++) {
          sum += apriori[i][j][z];
        }
        final double p = apriori[i][j][data.getCategoricalValue(j)];
        logProb += Math.log(p / sum);
      }
      logProb += Math.log(priors[i] / priorSum);
      logProbs[i] = logProb;
      maxLogProg = Math.max(maxLogProg, logProb);
    }

    final double denom = MathTricks.logSumExp(logProbs, maxLogProg);

    for (int i = 0; i < results.size(); i++) {
      results.setProb(i, exp(logProbs[i] - denom));
    }
    results.normalize();
    return results;
  }

  @Override
  public NaiveBayesUpdateable clone() {
    return new NaiveBayesUpdateable(this);
  }

  /**
   * Returns <tt>true</tt> if the input is assume sparse
   *
   * @return <tt>true</tt> if the input is assume sparse
   */
  public boolean isSparseInput() {
    return sparseInput;
  }

  /**
   * Sets whether or not that classifier should behave as if the input vectors
   * are sparse. This means zero values in the input will be ignored when
   * performing classification.
   *
   * @param sparseInput
   *          <tt>true</tt> to use a sparse model
   */
  public void setSparse(final boolean sparseInput) {
    this.sparseInput = sparseInput;
  }

  @Override
  public void setUp(final CategoricalData[] categoricalAttributes, final int numericAttributes,
      final CategoricalData predicting) {
    final int nCat = predicting.getNumOfCategories();
    apriori = new double[nCat][categoricalAttributes.length][];
    valueStats = new OnLineStatistics[nCat][numericAttributes];
    priors = new double[nCat];
    priorSum = nCat;
    Arrays.fill(priors, 1.0);

    for (int i = 0; i < nCat; i++) {
      // Iterate through the categorical variables
      for (int j = 0; j < categoricalAttributes.length; j++) {
        apriori[i][j] = new double[categoricalAttributes[j].getNumOfCategories()];

        // Laplace correction, put in an extra occurance for each variable
        for (int z = 0; z < apriori[i][j].length; z++) {
          apriori[i][j][z] = 1;
        }
      }
      for (int j = 0; j < numericAttributes; j++) {
        valueStats[i][j] = new OnLineStatistics();
      }
    }
  }

  @Override
  public boolean supportsWeightedData() {
    return true;
  }

  @Override
  public void update(final DataPoint dataPoint, final int targetClass) {
    final double weight = dataPoint.getWeight();
    final Vec x = dataPoint.getNumericalValues();
    if (sparseInput) {
      for (final IndexValue iv : x) {
        valueStats[targetClass][iv.getIndex()].add(iv.getValue(), weight);
      }
    } else {
      for (int j = 0; j < x.length(); j++) {
        valueStats[targetClass][j].add(x.get(j), weight);
      }
    }

    // Categorical value updates
    final int[] catValues = dataPoint.getCategoricalValues();
    for (int j = 0; j < apriori[targetClass].length; j++) {
      apriori[targetClass][j][catValues[j]]++;
    }
    priorSum++;
    priors[targetClass]++;
  }

}
