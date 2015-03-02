package jsat.clustering.evaluation;

import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.dissimilarity.ClusterDissimilarity;

/**
 * Provides the contract for evaluating the quality of a hard assignment of
 * clustering a dataset. The value returned indicates the quality of the 
 * clustering, with smaller values indicating a good clustering, and larger 
 * values indicating a poor clustering. <br>
 * This differs from {@link ClusterDissimilarity} in that it evaluates all 
 * clusters, instead of just measuring the dissimilarity of two specific clusters. 
 * 
 * @author Edward Raff
 */
public interface ClusterEvaluation 
{
    /**
     * Evaluates the clustering of the given clustering. 
     * @param designations the array that stores the cluster assignments for 
     * each data point in the data set
     * @param dataSet the data set that contains all data points
     * @return a value in [0, Inf) that indicates the quality of the clustering. 
     */
    public double evaluate(int[] designations, DataSet dataSet);
    
    /**
     * Evaluates the clustering of the given set of clusters. 
     * @param dataSets a list of lists, where the size of the first index 
     * indicates the the number of clusters, and the list at each index is the 
     * data points that make up each cluster. 
     * @return a value in [0, Inf) that indicates the quality of the clustering. 
     */
    public double evaluate(List<List<DataPoint>> dataSets);
    
    public ClusterEvaluation clone();
}
