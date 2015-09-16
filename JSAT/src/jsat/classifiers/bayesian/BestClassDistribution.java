package jsat.classifiers.bayesian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;

import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.distributions.multivariate.MultivariateDistribution;
import jsat.exceptions.FailedToFitException;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;

/**
 * BestClassDistribution is a generic class for performing classification by
 * fitting a {@link MultivariateDistribution} to each class. The distribution is
 * supplied by the user, and each class if fit to the same type of distribution.
 * Classification is then performed by returning the class of the most likely
 * distribution given the data point.
 *
 * @author Edward Raff
 */
public class BestClassDistribution implements Classifier, Parameterized {

  private static final long serialVersionUID = -1746145372146154228L;
  /**
   * The default value for whether or not to use the prior probability of a
   * class when making classification decisions is {@value #USE_PRIORS}.
   */
  public static final boolean USE_PRIORS = true;
  private final MultivariateDistribution baseDist;
  private List<MultivariateDistribution> dists;

  /**
   * The prior probabilities of each class
   */
  private double priors[];
  /**
   * Controls whether or no the prior probability will be used when computing
   * probabilities
   */
  private boolean usePriors;

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public BestClassDistribution(final BestClassDistribution toCopy) {
    if (toCopy.priors != null) {
      priors = Arrays.copyOf(toCopy.priors, toCopy.priors.length);
    }
    baseDist = toCopy.baseDist.clone();
    if (toCopy.dists != null) {
      dists = new ArrayList<MultivariateDistribution>(toCopy.dists.size());
      for (final MultivariateDistribution md : toCopy.dists) {
        dists.add(md == null ? null : md.clone());
      }
    }
  }

  public BestClassDistribution(final MultivariateDistribution baseDist) {
    this(baseDist, USE_PRIORS);
  }

  public BestClassDistribution(final MultivariateDistribution baseDist, final boolean usePriors) {
    this.baseDist = baseDist;
    this.usePriors = usePriors;
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    final CategoricalResults cr = new CategoricalResults(dists.size());

    for (int i = 0; i < dists.size(); i++) {
      if (dists.get(i) == null) {
        continue;
      }
      double p = 0;
      try {
        p = dists.get(i).pdf(data.getNumericalValues());
      } catch (final ArithmeticException ex) {

      }
      if (usePriors) {
        p *= priors[i];
      }
      cr.setProb(i, p);
    }
    cr.normalize();
    return cr;
  }

  @Override
  public Classifier clone() {
    return new BestClassDistribution(this);
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
   * Returns whether or not this object uses the prior probabilities for
   * classification.
   *
   * @return {@code true} if the prior probabilities are being used,
   *         {@code false} if not.
   */
  public boolean isUsePriors() {
    return usePriors;
  }

  /**
   * Controls whether or not the priors will be used for classification. This
   * value can be changed at any time, before or after training has occurred.
   *
   * @param usePriors
   *          <tt>true</tt> to use the prior probabilities for each class,
   *          <tt>false</tt> to ignore them.
   */
  public void setUsePriors(final boolean usePriors) {
    this.usePriors = usePriors;
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    priors = dataSet.getPriors();
    dists = new ArrayList<MultivariateDistribution>(dataSet.getClassSize());

    for (int i = 0; i < dataSet.getClassSize(); i++) {
      final MultivariateDistribution dist = baseDist.clone();
      final List<DataPoint> samp = dataSet.getSamples(i);
      if (samp.isEmpty()) {
        dists.add(null);
        continue;
      }
      dist.setUsingDataList(samp);
      dists.add(dist);
    }
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    try {
      dists = new ArrayList<MultivariateDistribution>();
      priors = dataSet.getPriors();
      final List<Future<MultivariateDistribution>> newDists = new ArrayList<Future<MultivariateDistribution>>();
      final MultivariateDistribution sourceDist = baseDist;
      for (int i = 0; i < dataSet.getPredicting().getNumOfCategories(); i++)// Calculate
                                                                            // the
                                                                            // Multivariate
                                                                            // normal
                                                                            // for
                                                                            // each
                                                                            // category
      {
        final List<DataPoint> class_i = dataSet.getSamples(i);
        final Future<MultivariateDistribution> tmp = threadPool.submit(new Callable<MultivariateDistribution>() {

          @Override
          public MultivariateDistribution call() throws Exception {
            if (class_i.isEmpty()) {// Nowthing we can do
              return null;
            }
            final MultivariateDistribution dist = sourceDist.clone();
            dist.setUsingDataList(class_i);
            return dist;
          }
        });

        newDists.add(tmp);
      }
      for (final Future<MultivariateDistribution> future : newDists) {
        dists.add(future.get());
      }
    } catch (final Exception ex) {
      Logger.getLogger(MultivariateNormals.class.getName()).log(Level.SEVERE, null, ex);
      throw new FailedToFitException(ex);
    }
  }

}
