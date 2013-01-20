package jsat.clustering.evaluation;

import java.util.List;
import jsat.classifiers.DataPoint;
import jsat.clustering.evaluation.intra.IntraClusterEvaluation;

/**
 * Evaluates a cluster based on the sum of scores for some 
 * {@link IntraClusterEvaluation} applied to each cluster. 
 * 
 * @author Edward Raff
 */
public class IntraClusterSumEvaluation extends ClusterEvaluationBase
{
    private IntraClusterEvaluation ice;

    /**
     * Creates a new cluster evaluation that returns the sum of the intra 
     * cluster evaluations
     * @param ice the intra cluster evaluation to use
     */
    public IntraClusterSumEvaluation(IntraClusterEvaluation ice)
    {
        this.ice = ice;
    }

    @Override
    public double evaluate(List<List<DataPoint>> dataSets)
    {
        double score = 0;
        for(List<DataPoint> list : dataSets)
            score += ice.evaluate(list);
        return score;
    }
    
}
