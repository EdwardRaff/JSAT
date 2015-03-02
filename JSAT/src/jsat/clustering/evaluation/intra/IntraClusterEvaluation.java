
package jsat.clustering.evaluation.intra;

import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;

/**
 * This interface defines the contract for a method to evaluate the 
 * intra-cluster distance. This means an evaluation of a single cluster, 
 * where a higher value indicates a poorly formed cluster. This evaluation does 
 * not take into account any other neighboring clusters
 * 
 * @author Edward Raff
 */
public interface IntraClusterEvaluation
{
    /**
     * Evaluates the cluster represented by the given list of data points. 
     * @param designations the array of cluster designations for the data set
     * @param dataSet the full data set of all clusters
     * @param clusterID the cluster id in the <tt>designations</tt> array to 
     * return the evaluation of
     * @return the value in the range [0, Inf) that indicates how well formed
     * the cluster is. 
     */
    public double evaluate(int[] designations, DataSet dataSet, int clusterID);
    /**
     * Evaluates the cluster represented by the given list of data points. 
     * @param dataPoints the data points that make up this cluster
     * @return the value in the range [0, Inf) that indicates how well formed
     * the cluster is. 
     */
    public double evaluate(List<DataPoint> dataPoints);
    
    public IntraClusterEvaluation clone();
}
