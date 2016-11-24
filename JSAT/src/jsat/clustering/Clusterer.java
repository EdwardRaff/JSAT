
package jsat.clustering;

import java.io.Serializable;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;

/**
 * Defines the interface for a generic clustering algorithm. 
 * 
 * @author Edward Raff
 */
public interface Clusterer extends Serializable
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
     * @param designations the array which will contain the designated values. The array will be altered and returned by 
     * the function. If <tt>null</tt> is given, a new array will be created and returned.
     * @return an array indicating for each value indicating the cluster designation. This is the same array as 
     * <tt>designations</tt>, or a new one if the input array was <tt>null</tt>
     */
    public int[] cluster(DataSet dataSet, int[] designations);
    
    /**
     * Performs clustering on the given data set. Parameters may be estimated by the method, or other heuristics performed. 
     * 
     * @param dataSet the data set to perform clustering on 
     * @param threadpool a source of threads to run tasks
     * @return list of clusters found by this method. 
     */
    public List<List<DataPoint>> cluster(DataSet dataSet, ExecutorService threadpool);
    
    /**
     * Performs clustering on the given data set. Parameters may be estimated by the method, or other heuristics performed. 
     * 
     * @param dataSet the data set to perform clustering on 
     * @param threadpool a source of threads to run tasks
     * @param designations the array which will contain the designated values. The array will be altered and returned by 
     * the function. If <tt>null</tt> is given, a new array will be created and returned.
     * @return an array indicating for each value indicating the cluster designation. This is the same array as 
     * <tt>designations</tt>, or a new one if the input array was <tt>null</tt>
     */
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations);
   
    /**
     * Indicates whether the model knows how to cluster using weighted data
     * points. If it does, the model will train assuming the weights. The values
     * returned by this method may change depending on the parameters set for
     * the model.
     *
     * @return <tt>true</tt> if the model supports weighted data, <tt>false</tt>
     * otherwise
     */
    public boolean supportsWeightedData();
    
    public Clusterer clone();
}
