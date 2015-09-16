package jsat.classifiers;

import java.util.concurrent.ExecutorService;

import jsat.exceptions.UntrainedModelException;

/**
 * A Naive classifier that simply returns the prior probabilities as the
 * classification decision.
 *
 * @author Edward Raff
 */
public class PriorClassifier implements Classifier {

  private static final long serialVersionUID = 7763388716880766538L;
  private CategoricalResults cr;

  /**
   * Creates a new PriorClassifeir
   */
  public PriorClassifier() {
  }

  /**
   * Creates a new Prior Classifier that is given the results it should be
   * returning
   *
   * @param cr
   *          the prior probabilities for classification
   */
  public PriorClassifier(final CategoricalResults cr) {
    this.cr = cr;
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    if (cr == null) {
      throw new UntrainedModelException("PriorClassifier has not been trained");
    }
    return cr;
  }

  @Override
  public Classifier clone() {
    final PriorClassifier clone = new PriorClassifier();
    if (cr != null) {
      clone.cr = cr.clone();
    }
    return clone;
  }

  @Override
  public boolean supportsWeightedData() {
    return true;
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    cr = new CategoricalResults(dataSet.getPredicting().getNumOfCategories());
    for (int i = 0; i < dataSet.getSampleSize(); i++) {
      cr.incProb(dataSet.getDataPointCategory(i), dataSet.getDataPoint(i).getWeight());
    }
    cr.normalize();
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    trainC(dataSet);
  }
}
