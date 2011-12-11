
package jsat.clustering;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;

/**
 * Defines the interface for a generic clustering algorithm. 
 * 
 * @author Edward Raff
 */
public interface Clusterer
{
   
    /**
     * Performs clustering on the given data set. Parameters may be estimated by the method, or other heuristics performed. 
     * 
     * @param dataSet the data set to perform clustering on 
     * @return A list of clusters found by this method. 
     */
    public List<List<DataPoint>> cluster(DataSet dataSet);
    
    /**
     * Performs clustering on the given data set. Parameters may be estimated by the method, or other heuristics performed. 
     * 
     * @param dataSet the data set to perform clustering on 
     * @param threadpool a source of threads to run tasks
     * @return list of clusters found by this method. 
     */
    public List<List<DataPoint>> cluster(DataSet dataSet, ExecutorService threadpool);
}
