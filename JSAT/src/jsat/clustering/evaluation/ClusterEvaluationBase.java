package jsat.clustering.evaluation;

import jsat.DataSet;
import jsat.clustering.ClustererBase;

/**
 * Base implementation for one of the methods in {@link ClusterEvaluation} to
 * make life easier.
 *
 * @author Edward Raff
 */
abstract public class ClusterEvaluationBase implements ClusterEvaluation {

  @Override
  public abstract ClusterEvaluation clone();

  @Override
  public double evaluate(final int[] designations, final DataSet dataSet) {
    return evaluate(ClustererBase.createClusterListFromAssignmentArray(designations, dataSet));
  }
}
