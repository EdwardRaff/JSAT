package jsat.lossfunctions;

import jsat.classifiers.CategoricalResults;

/**
 * Specifies a loss function for binary classification problems.
 *
 * @author Edward Raff
 */
public interface LossC extends LossFunc {

  @Override
  public LossC clone();

  /**
   * Given the score value of a data point, this returns the classification
   * results.
   *
   * @param score
   *          the score for a data point
   * @return the categorical results with the correct probability values for
   *         this loss function.
   */
  public CategoricalResults getClassification(double score);

  /**
   * Computes the first derivative of the getLoss function.
   *
   * @param pred
   *          the predicted score in (-Infinity, Infinity)
   * @param y
   *          the true class label in {-1, 1}
   * @return the first derivative of the getLoss
   */
  @Override
  public double getDeriv(double pred, double y);

  /**
   * Computes the second derivative of the getLoss function.
   *
   * @param pred
   *          the predicted score in (-Infinity, Infinity)
   * @param y
   *          the true class label in {-1, 1}
   * @return the second derivative of the getLoss function
   */
  @Override
  public double getDeriv2(double pred, double y);

  /**
   * Computes the getLoss for a classification problem.
   *
   * @param pred
   *          the predicted score in (-Infinity, Infinity)
   * @param y
   *          the true class label in {-1, 1}
   * @return the getLoss in [0, Inf)
   */
  @Override
  public double getLoss(double pred, double y);
}
