
package jsat.clustering;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;

/**
 *
 * @author Edward Raff
 */
public interface Clusterer
{
    /**
     * Performs clustering on the given data set. 
     * 
     * @param dataSet the data points to perform clustering on
     * @param clusters the number of clusters to assume
     * @param threadpool a source of threads to run tasks
     * @return A list of DataSets, where each DataSet contains the data 
     * points for one cluster in the group
     */
    public List<List<DataPoint>> cluster(DataSet dataSet, int clusters, ExecutorService threadpool);
    
    public List<List<DataPoint>> cluster(DataSet dataSet, int clusters);
    
    /**
     * Performs clustering on the given data set. The implementation will 
     * attempt to determine the best number of clusters for the given data. 
     * 
     * @param dataSet the data points to perform clustering on
     * @param the lower bound, inclusive, of the range to search
     * @param the uper bound, inclusive, of the range to search
     * @param threadpool a source of threads to run tasks
     * @return  A list of DataSets, where each DataSet contains the data 
     * points for one cluster in the group
     */
    public List<List<DataPoint>> cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool);
    
    public List<List<DataPoint>> cluster(DataSet dataSet, int lowK, int highK);
}
