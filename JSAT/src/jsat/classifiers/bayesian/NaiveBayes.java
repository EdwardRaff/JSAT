package jsat.classifiers.bayesian;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static jsat.distributions.DistributionSearch.getBestDistribution;

import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;

import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.distributions.ContinuousDistribution;
import jsat.distributions.DistributionSearch;
import jsat.distributions.Normal;
import jsat.distributions.empirical.KernelDensityEstimator;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.math.MathTricks;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;

/**
 *
 * Provides an implementation of the Naive Bayes classifier that assumes numeric
 * features come from some continuous probability distribution. By default this
 * implementation restricts itself to only the {@link Normal Gaussian}
 * distribution, and becomes Gaussian Naive Bayes. Other distributions are
 * supported, and a {@link KernelDensityEstimator} can be used as well. <br>
 * <br>
 * By default, this implementation assumes that the input vectors are sparse and
 * the distribution will only be estimated by the non-zero values, and features
 * that are zero will be ignored during prediction time. This should be turned
 * off when using dense data by calling {@link #setSparceInput(boolean) } <br>
 * <br>
 * Naive Bayes assumes that all attributes are perfectly independent.
 *
 * @author Edward Raff
 */
public class NaiveBayes implements Classifier, Parameterized {

  private class AprioriCounterRunable implements Runnable {

    int i;
    int j;
    List<DataPoint> dataSamples;
    CountDownLatch latch;

    public AprioriCounterRunable(final int i, final int j, final List<DataPoint> dataSamples,
        final CountDownLatch latch) {
      this.i = i;
      this.j = j;
      this.dataSamples = dataSamples;
      this.latch = latch;
    }

    @Override
    public void run() {
      for (final DataPoint point : dataSamples)// Count each occurance
      {
        apriori[i][j][point.getCategoricalValue(j)]++;
      }

      // Convert the coutns to apriori probablities by dividing the count by the
      // total occurances
      double sum = 0;
      for (int z = 0; z < apriori[i][j].length; z++) {
        sum += apriori[i][j][z];
      }
      for (int z = 0; z < apriori[i][j].length; z++) {
        apriori[i][j][z] /= sum;
      }
      latch.countDown();
    }

  }

  /**
   * Runnable task for selecting the right distribution for each task
   */
  private class DistributionSelectRunable implements Runnable {

    int i;
    int j;
    Vec v;
    CountDownLatch countDown;

    public DistributionSelectRunable(final int i, final int j, final Vec v, final CountDownLatch countDown) {
      this.i = i;
      this.j = j;
      this.v = v;
      this.countDown = countDown;
    }

    @Override
    public void run() {
      try {
        distributions[i][j] = numericalHandling.fit(v);
      } catch (final ArithmeticException e) {
        distributions[i][j] = null;
      }
      countDown.countDown();
    }

  }

  /**
   * There are multiple ways of handling numerical attributes. These provide the
   * different ways that NaiveBayes can deal with them.
   */
  public enum NumericalHandeling {
    /**
     * All numerical attributes are fit to a {@link Normal} distribution.
     */
    /**
     * All numerical attributes are fit to a {@link Normal} distribution.
     */
    NORMAL {
      @Override
      protected ContinuousDistribution fit(final Vec v) {
        return getBestDistribution(v, new Normal(0, 1));
      }
    },
    /**
     * The best fitting {@link ContinuousDistribution} is selected by
     * {@link DistributionSearch#getBestDistribution(jsat.linear.Vec) }
     */
    BEST_FIT {

      @Override
      protected ContinuousDistribution fit(final Vec v) {
        return getBestDistribution(v);
      }
    },
    /**
     * The best fitting {@link ContinuousDistribution} is selected by
     * {@link DistributionSearch#getBestDistribution(jsat.linear.Vec, double) },
     * and provides a cut off value to use the {@link KernelDensityEstimator}
     * instead
     */
    BEST_FIT_KDE {

      private final double cutOff = 0.9;
      // XXX these methods are never and cannot be used
      // /**
      // * Sets the cut off value used before fitting an empirical distribution
      // * @param c the cut off value, should be between (0, 1).
      // */
      // public void setCutOff(double c)
      // {
      // cutOff = c;
      // }
      //
      // /**
      // * Returns the cut off value used before fitting an empirical
      // distribution
      // * @return the cut off value used before fitting an empirical
      // distribution
      // */
      // public double getCutOff()
      // {
      // return cutOff;
      // }

      @Override
      protected ContinuousDistribution fit(final Vec v) {
        return getBestDistribution(v, cutOff);
      }
    };

    abstract protected ContinuousDistribution fit(Vec y);
  }

  private static final long serialVersionUID = -2437775653277531182L;
  /**
   * The default method of handling numeric attributes is
   * {@link NumericalHandeling#NORMAL}.
   */
  public static final NumericalHandeling defaultHandling = NumericalHandeling.NORMAL;

  /**
   *
   */
  private double[][][] apriori;

  private ContinuousDistribution[][] distributions;

  private NumericalHandeling numericalHandling;

  private double[] priors;

  /**
   * Handles how vectors are handled. If true, it is assumed vectors are sparce
   * - and zero values will be ignored when training and classifying.
   */
  private boolean sparceInput = true;

  /**
   * Creates a new Gaussian Naive Bayes classifier
   */
  public NaiveBayes() {
    this(defaultHandling);
  }

  /**
   * Creates a new Naive Bayes classifier that uses the specific method for
   * handling numeric features.
   *
   * @param numericalHandling
   *          the method to use for numeric features
   */
  public NaiveBayes(final NumericalHandeling numericalHandling) {
    this.numericalHandling = numericalHandling;
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {

    final CategoricalResults results = new CategoricalResults(distributions.length);
    final double[] logProbs = new double[distributions.length];
    final Vec numVals = data.getNumericalValues();
    double maxLogProg = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < distributions.length; i++) {
      double logProb = 0;
      if (sparceInput) {
        final Iterator<IndexValue> iter = numVals.getNonZeroIterator();
        while (iter.hasNext()) {
          final IndexValue indexValue = iter.next();
          final int j = indexValue.getIndex();
          double logPDF;
          if (distributions[i][j] == null) {
            logPDF = Double.NEGATIVE_INFINITY;// Should not occur
          } else {
            logPDF = distributions[i][j].logPdf(indexValue.getValue());
          }
          if (Double.isInfinite(logPDF)) {// Avoid propigation -infinty when the
                                          // probability is zero
            logProb += log(1e-16);//
          } else {
            logProb += logPDF;
          }
        }
      } else {
        for (int j = 0; j < distributions[i].length; j++) {
          double logPDF;
          if (distributions[i][j] == null) {
            logPDF = Double.NEGATIVE_INFINITY;// Should not occur
          } else {
            logPDF = distributions[i][j].logPdf(numVals.get(j));
          }
          if (Double.isInfinite(logPDF)) {// Avoid propigation -infinty when the
                                          // probability is zero
            logProb += log(1e-16);//
          } else {
            logProb += logPDF;
          }
        }
      }

      // the i goes up to the number of categories, same for aprioror
      for (int j = 0; j < apriori[i].length; j++) {
        final double p = apriori[i][j][data.getCategoricalValue(j)];
        logProb += log(p);
      }

      logProb += log(priors[i]);
      logProbs[i] = logProb;
      maxLogProg = Math.max(maxLogProg, logProb);
    }

    if (maxLogProg == Double.NEGATIVE_INFINITY) // Everything reported no!
    {
      for (int i = 0; i < results.size(); i++) {
        results.setProb(i, 1.0 / results.size());
      }
      return results;
    }

    final double denom = MathTricks.logSumExp(logProbs, maxLogProg);

    for (int i = 0; i < results.size(); i++) {
      results.setProb(i, exp(logProbs[i] - denom));
    }
    results.normalize();
    return results;
  }

  @Override
  public Classifier clone() {
    final NaiveBayes newBayes = new NaiveBayes(numericalHandling);

    if (apriori != null) {
      newBayes.apriori = new double[apriori.length][][];
      for (int i = 0; i < apriori.length; i++) {
        newBayes.apriori[i] = new double[apriori[i].length][];
        for (int j = 0; apriori[i].length > 0 && j < apriori[i][j].length; j++) {
          newBayes.apriori[i][j] = Arrays.copyOf(apriori[i][j], apriori[i][j].length);
        }
      }
    }

    if (distributions != null) {
      newBayes.distributions = new ContinuousDistribution[distributions.length][];
      for (int i = 0; i < distributions.length; i++) {
        newBayes.distributions[i] = new ContinuousDistribution[distributions[i].length];
        for (int j = 0; j < distributions[i].length; j++) {
          newBayes.distributions[i][j] = distributions[i][j].clone();
        }
      }
    }

    if (priors != null) {
      newBayes.priors = Arrays.copyOf(priors, priors.length);
    }

    return newBayes;
  }

  /**
   * Returns the method used to handle numerical attributes
   *
   * @return the method used to handle numerical attributes
   */
  public NumericalHandeling getNumericalHandling() {
    return numericalHandling;
  }

  @Override
  public Parameter getParameter(final String paramName) {
    return Parameter.toParameterMap(getParameters()).get(paramName);
  }

  @Override
  public List<Parameter> getParameters() {
    return Collections.unmodifiableList(Parameter.getParamsFromMethods(this));
  }

  private Vec getSampleVariableVector(final ClassificationDataSet dataSet, final int category, final int j) {
    Vec vals = dataSet.getSampleVariableVector(category, j);

    if (sparceInput) {
      final List<Double> nonZeroVals = new DoubleList();
      for (int i = 0; i < vals.length(); i++) {
        if (vals.get(i) != 0) {
          nonZeroVals.add(vals.get(i));
        }
      }
      vals = new DenseVector(nonZeroVals);
    }

    return vals;
  }

  /**
   * Returns <tt>true</tt> if the Classifier assumes that data points are
   * sparce.
   *
   * @return <tt>true</tt> if the Classifier assumes that data points are
   *         sparce.
   * @see #setSparceInput(boolean)
   */
  public boolean isSparceInput() {
    return sparceInput;
  }

  /**
   * Sets the method used by this instance for handling numerical attributes.
   * This has no effect on an already trained classifier, but will change the
   * result if trained again.
   *
   * @param numericalHandling
   *          the method to use for numerical attributes
   */
  public void setNumericalHandling(final NumericalHandeling numericalHandling) {
    this.numericalHandling = numericalHandling;
  }

  /**
   * Tells the Naive Bayes classifier to assume the importance of sparseness in
   * the numerical values. This means that values of zero will be ignored in
   * computation and classification.<br>
   * This allows faster, more efficient computation of results if the data
   * points are indeed sparce. This will also produce different results. This
   * value should not be changed after training and before classification.
   *
   * @param sparceInput
   *          <tt>true</tt> to assume sparseness in the data, <tt>false</tt> to
   *          ignore it and assume zeros are meaningful values.
   * @see #isSparceInput()
   */
  public void setSparceInput(final boolean sparceInput) {
    this.sparceInput = sparceInput;
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    trainC(dataSet, new FakeExecutor());
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    final int nCat = dataSet.getPredicting().getNumOfCategories();
    apriori = new double[nCat][dataSet.getNumCategoricalVars()][];
    distributions = new ContinuousDistribution[nCat][dataSet.getNumNumericalVars()];
    priors = dataSet.getPriors();

    final int totalWorkers = nCat * (dataSet.getNumNumericalVars() + dataSet.getNumCategoricalVars());
    final CountDownLatch latch = new CountDownLatch(totalWorkers);

    // Go through each classification
    for (int i = 0; i < nCat; i++) {
      // Set ditribution for the numerical values
      for (int j = 0; j < dataSet.getNumNumericalVars(); j++) {
        final Runnable rn = new DistributionSelectRunable(i, j, getSampleVariableVector(dataSet, i, j), latch);
        threadPool.submit(rn);
      }

      final List<DataPoint> dataSamples = dataSet.getSamples(i);

      // Iterate through the categorical variables
      for (int j = 0; j < dataSet.getNumCategoricalVars(); j++) {
        apriori[i][j] = new double[dataSet.getCategories()[j].getNumOfCategories()];

        // Laplace correction, put in an extra occurance for each variable
        for (int z = 0; z < apriori[i][j].length; z++) {
          apriori[i][j][z] = 1;
        }

        final Runnable rn = new AprioriCounterRunable(i, j, dataSamples, latch);
        threadPool.submit(rn);
      }
    }

    // Wait for all the threads to finish
    try {
      latch.await();
    } catch (final InterruptedException ex) {
      ex.printStackTrace();
    }
  }

}
