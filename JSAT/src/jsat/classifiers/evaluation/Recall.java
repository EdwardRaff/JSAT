package jsat.classifiers.evaluation;

/**
 * Evaluates a classifier based on the Recall rate, where the class of index 0
 * is considered the positive class. This score is only valid for binary
 * classification problems.
 *
 * @author Edward Raff
 */
public class Recall extends SimpleBinaryClassMetric {

  private static final long serialVersionUID = 4832185425203972017L;

  /**
   * Creates a new Recall evaluator
   */
  public Recall() {
    super();
  }

  /**
   * Copy constructor
   *
   * @param toClone
   *          the object to copy
   */
  public Recall(final Recall toClone) {
    super(toClone);
  }

  @Override
  public Recall clone() {
    return new Recall(this);
  }

  @Override
  public boolean equals(final Object obj) {
    return this.getClass().isAssignableFrom(obj.getClass()) && obj.getClass().isAssignableFrom(this.getClass());
  }

  @Override
  public String getName() {
    return "Recall";
  }

  @Override
  public double getScore() {
    return tp / (tp + fn);
  }

  @Override
  public int hashCode() {
    return getName().hashCode();
  }

}
