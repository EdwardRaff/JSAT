package jsat.classifiers.boosting;

import jsat.classifiers.Classifier;
import jsat.distributions.ContinuousDistribution;
import jsat.distributions.Normal;
import jsat.regression.Regressor;

/**
 * Wagging using the {@link Normal} distribution.
 *
 * @author Edward Raff
 */
public class WaggingNormal extends Wagging {

  private static final long serialVersionUID = -4149453672311329863L;

  /**
   * Creates a new Wagging classifier
   *
   * @param weakLearner
   *          the weak learner to use
   * @param interations
   *          the number of iterations to perform
   */
  public WaggingNormal(final Classifier weakLearner, final int interations) {
    super(new Normal(1, 2), weakLearner, interations);
  }

  /**
   * Creates a new Wagging regressor
   *
   * @param weakLearner
   *          the weak learner to use
   * @param interations
   *          the number of iterations to perform
   */
  public WaggingNormal(final Regressor weakLearner, final int interations) {
    super(new Normal(1, 2), weakLearner, interations);
  }

  /**
   * Copy constructor
   *
   * @param clone
   *          to copy
   */
  protected WaggingNormal(final Wagging clone) {
    super(clone);
  }

  @Override
  public WaggingNormal clone() {
    return new WaggingNormal(this);
  }

  @Override
  public ContinuousDistribution getDistribution() {
    return super.getDistribution();
  }

  /**
   * Returns the mean value used for the normal distribution
   *
   * @return the mean value used
   */
  public double getMean() {
    return getDistribution().mean();
  }

  /**
   * Returns the standard deviation used for the normal distribution
   *
   * @return the standard deviation used
   */
  public double getStandardDeviations() {
    return getDistribution().standardDeviation();
  }

  @Override
  public void setDistribution(final ContinuousDistribution dist) {
    if (dist instanceof Normal) {
      super.setDistribution(dist);
    } else {
      throw new RuntimeException("Only the Normal distribution is valid");
    }
  }

  /**
   * Sets the mean value used for the normal distribution
   *
   * @param mean
   *          the new mean value
   */
  public void setMean(final double mean) {
    if (Double.isInfinite(mean) || Double.isNaN(mean)) {
      throw new ArithmeticException("Mean must be a real number, not " + mean);
    }
    ((Normal) getDistribution()).setMean(mean);
  }

  /**
   * Sets the standard deviations used for the normal distribution
   *
   * @param devs
   *          the standard deviations to set
   */
  public void setStandardDeviations(final double devs) {
    if (devs <= 0 || Double.isInfinite(devs) || Double.isNaN(devs)) {
      throw new ArithmeticException("The stnd devs must be a positive value");
    }
    ((Normal) getDistribution()).setStndDev(devs);
  }
}
