
package jsat.clustering;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import static jsat.clustering.ClustererBase.createClusterListFromAssignmentArray;

/**
 * Defines a clustering method that requires the number of clusters in the data set to be known before hand. 
 * 
 * @author Edward Raff
 */
public interface KClusterer extends Clusterer
{
    /**
     * Performs clustering on the given data set. 
     * 
     * @param dataSet the data points to perform clustering on
     * @param clusters the number of clusters to assume
     * @param parallel a source of threads to run tasks
     * @return the java.util.List<java.util.List<jsat.classifiers.DataPoint>>
     */
    default public List<List<DataPoint>> cluster(DataSet dataSet, int clusters, boolean parallel)
    {
        int[] assignments = cluster(dataSet, clusters, parallel, (int[]) null);
        return createClusterListFromAssignmentArray(assignments, dataSet);
    }
    
    public int[] cluster(DataSet dataSet, int clusters, boolean parallel, int[] designations);
    
    /**
     * Performs clustering on the given data set. 
     * 
     * @param dataSet the data points to perform clustering on
     * @param clusters the number of clusters to assume
     * @return A list of DataSets, where each DataSet contains the data 
     * points for one cluster in the group
     */
    default public List<List<DataPoint>> cluster(DataSet dataSet, int clusters)
    {
        return cluster(dataSet, clusters, false);
    }
    
    default public int[] cluster(DataSet dataSet, int clusters, int[] designations)
    {
        return cluster(dataSet, clusters, false, designations);
    }
    
    /**
     * Performs clustering on the given data set. The implementation will 
     * attempt to determine the best number of clusters for the given data. 
     * 
     * @param dataSet the data points to perform clustering on
     * @param lowK the lower bound, inclusive, of the range to search
     * @param highK the upper bound, inclusive, of the range to search
     * @param parallel a source of threads to run tasks
     * @return  the java.util.List<java.util.List<jsat.classifiers.DataPoint>>
     */
    default public List<List<DataPoint>> cluster(DataSet dataSet, int lowK, int highK, boolean parallel)
    {
        int[] assignments = cluster(dataSet, lowK, highK, parallel, (int[]) null);
        return createClusterListFromAssignmentArray(assignments, dataSet);
    }
    
    public int[] cluster(DataSet dataSet, int lowK, int highK, boolean parallel, int[] designations);
    
    /**
     * Performs clustering on the given data set. The implementation will 
     * attempt to determine the best number of clusters for the given data. 
     * 
     * @param dataSet the data points to perform clustering on
     * @param lowK the lower bound, inclusive, of the range to search
     * @param highK the upper bound, inclusive, of the range to search
     * @return  A list of DataSets, where each DataSet contains the data 
     * points for one cluster in the group
     */
    default public List<List<DataPoint>> cluster(DataSet dataSet, int lowK, int highK)
    {
        return cluster(dataSet, lowK, highK, false);
    }
    
    default public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        return cluster(dataSet, lowK, highK, false, designations);
    }
    
    @Override
    public KClusterer clone();
}
