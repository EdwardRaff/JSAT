package jsat.classifiers.evaluation;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;

/**
 * Evaluates a classifier based on the Kappa statistic.
 *
 * @author Edward Raff
 */
public class Kappa implements ClassificationScore {

  private static final long serialVersionUID = -1684937057234736715L;
  private Matrix errorMatrix;

  public Kappa() {
  }

  public Kappa(final Kappa toClone) {
    if (toClone.errorMatrix != null) {
      errorMatrix = toClone.errorMatrix.clone();
    }
  }

  @Override
  public void addResult(final CategoricalResults prediction, final int trueLabel, final double weight) {
    errorMatrix.increment(prediction.mostLikely(), trueLabel, weight);
  }

  @Override
  public void addResults(final ClassificationScore other) {
    final Kappa otherObj = (Kappa) other;
    if (otherObj.errorMatrix == null) {
      return;
    }
    if (errorMatrix == null) {
      throw new RuntimeException("KappaScore has not been prepared");
    }
    errorMatrix.mutableAdd(otherObj.errorMatrix);
  }

  @Override
  public Kappa clone() {
    return new Kappa(this);
  }

  @Override
  public boolean equals(final Object obj) {
    return this.getClass().isAssignableFrom(obj.getClass()) && obj.getClass().isAssignableFrom(this.getClass());
  }

  @Override
  public String getName() {
    return "Kappa";
  }

  @Override
  public double getScore() {
    final double[] rowTotals = new double[errorMatrix.rows()];
    final double[] colTotals = new double[errorMatrix.rows()];
    for (int i = 0; i < errorMatrix.rows(); i++) {
      rowTotals[i] = errorMatrix.getRowView(i).sum();
      colTotals[i] = errorMatrix.getColumnView(i).sum();
    }

    double chanceAgreement = 0;
    double accuracy = 0;
    double totalCount = 0;
    for (int i = 0; i < rowTotals.length; i++) {
      chanceAgreement += rowTotals[i] * colTotals[i];
      totalCount += rowTotals[i];
      accuracy += errorMatrix.get(i, i);
    }
    chanceAgreement /= totalCount * totalCount;
    accuracy /= totalCount;

    return (accuracy - chanceAgreement) / (1 - chanceAgreement);
  }

  @Override
  public int hashCode() {
    return getName().hashCode();
  }

  @Override
  public boolean lowerIsBetter() {
    return false;
  }

  @Override
  public void prepare(final CategoricalData toPredict) {
    final int N = toPredict.getNumOfCategories();
    errorMatrix = new DenseMatrix(N, N);
  }

}
