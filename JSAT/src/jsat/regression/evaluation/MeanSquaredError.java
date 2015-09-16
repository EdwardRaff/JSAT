package jsat.regression.evaluation;

import jsat.math.OnLineStatistics;

/**
 * Uses the Mean of the Squared Errors between the predictions and the true
 * values.
 *
 * @author Edward Raff
 */
public class MeanSquaredError implements RegressionScore {

  private static final long serialVersionUID = 3655567184376550126L;
  private OnLineStatistics meanError;
  private boolean rmse;

  public MeanSquaredError() {
    this(false);
  }

  public MeanSquaredError(final boolean rmse) {
    setRMSE(rmse);
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public MeanSquaredError(final MeanSquaredError toCopy) {
    if (toCopy.meanError != null) {
      meanError = toCopy.meanError.clone();
    }
    rmse = toCopy.rmse;
  }

  @Override
  public void addResult(final double prediction, final double trueValue, final double weight) {
    if (meanError == null) {
      throw new RuntimeException("regression score has not been initialized");
    }
    meanError.add(Math.pow(prediction - trueValue, 2), weight);
  }

  @Override
  public void addResults(final RegressionScore other) {
    final MeanSquaredError otherObj = (MeanSquaredError) other;
    if (otherObj.meanError != null) {
      meanError.add(otherObj.meanError);
    }
  }

  @Override
  public MeanSquaredError clone() {
    return new MeanSquaredError(this);
  }

  @Override
  public boolean equals(final Object obj) {// XXX check for equality of fields
                                           // and obj == null
    if (this.getClass().isAssignableFrom(obj.getClass()) && obj.getClass().isAssignableFrom(this.getClass())) {
      return rmse == ((MeanSquaredError) obj).rmse;
    }
    return false;
  }

  @Override
  public String getName() {
    final String prefix = rmse ? "Root " : "";
    return prefix + "Mean Squared Error";
  }

  @Override
  public double getScore() {
    if (rmse) {
      return Math.sqrt(meanError.getMean());
    } else {
      return meanError.getMean();
    }
  }

  @Override
  public int hashCode() {// XXX this is a strange hashcode method
    return getName().hashCode();
  }

  public boolean isRMSE() {
    return rmse;
  }

  @Override
  public boolean lowerIsBetter() {
    return true;
  }

  @Override
  public void prepare() {
    meanError = new OnLineStatistics();
  }

  public void setRMSE(final boolean rmse) {
    this.rmse = rmse;
  }

}
