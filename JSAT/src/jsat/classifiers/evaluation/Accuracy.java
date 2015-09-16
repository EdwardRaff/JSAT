package jsat.classifiers.evaluation;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;

/**
 * Evaluates a classifier based on its accuracy in predicting the correct class.
 *
 * @author Edward Raff
 */
public class Accuracy implements ClassificationScore {

  private static final long serialVersionUID = 397690693205481128L;
  private double correct, total;

  public Accuracy() {
  }

  public Accuracy(final Accuracy toClone) {
    correct = toClone.correct;
    total = toClone.total;
  }

  @Override
  public void addResult(final CategoricalResults prediction, final int trueLabel, final double weight) {
    if (prediction.mostLikely() == trueLabel) {
      correct += weight;
    }
    total += weight;
  }

  @Override
  public void addResults(final ClassificationScore other) {
    final Accuracy otherObj = (Accuracy) other;
    correct += otherObj.correct;
    total += otherObj.total;
  }

  @Override
  public Accuracy clone() {
    return new Accuracy(this);
  }

  @Override
  public boolean equals(final Object obj) {
    return obj instanceof Accuracy;
  }

  @Override
  public String getName() {
    return "Accuracy";
  }

  @Override
  public double getScore() {
    return correct / total;
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
    correct = 0;
    total = 0;
  }

}
