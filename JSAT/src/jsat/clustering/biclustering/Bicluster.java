/*
 * This code was contributed under the public domain. 
 */
package jsat.clustering.biclustering;

import java.util.List;
import jsat.DataSet;

/**
 *
 * @author Edward Raff
 */
public interface Bicluster 
{    
    default public int[] cluster(DataSet dataSet, int lowK, int highK, boolean parallel, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.size()];
        return designations;
    }
    
    /**
     * Computes a biclustering of the dataset, where the goal is to identify a
     * fixed number of biclusters.
     *
     * @param dataSet the dataset to perform biclustering on
     * @param clusters the number of clusters to search for
     * @param parallel whether or not to use parallel computation
     * @param row_assignments This will store the the assignment of rows to each
     * by bicluster. After this function returns, the primary list will have a
     * sub-list for each bicluster. The i'th sub list contains the rows of the
     * matrix that belong to the i'th bicluster.
     * @param col_assignments This will store the assignment of columns to each
     * bicluster. After this function returns, the primary list will have a
     * sub-list for each bicluster. The i'th sub list contains the columns /
     * features of the matrix that belong to the i'th bicluster.
     */
    public void bicluster(DataSet dataSet, int clusters, boolean parallel,
            List<List<Integer>> row_assignments,
            List<List<Integer>> col_assignments);
    
    /**
     * Computes a biclustering of the dataset, where the goal is to identify an
     * unkown number of biclusters. 
     *
     * @param dataSet the dataset to perform biclustering on
     * @param clusters the number of clusters to search for
     * @param row_assignments This will store the the assignment of rows to each
     * by bicluster. After this function returns, the primary list will have a
     * sub-list for each bicluster. The i'th sub list contains the rows of the
     * matrix that belong to the i'th bicluster.
     * @param col_assingments This will store the assignment of columns to each
     * bicluster. After this function returns, the primary list will have a
     * sub-list for each bicluster. The i'th sub list contains the columns /
     * features of the matrix that belong to the i'th bicluster.
     */
    default public void bicluster(DataSet dataSet, int clusters, 
            List<List<Integer>> row_assignments,
            List<List<Integer>> col_assingments)
    {
        bicluster(dataSet, clusters, false, row_assignments, col_assingments);
    }
    
       
    /**
     * Indicates whether the model knows how to cluster using weighted data
     * points. If it does, the model will train assuming the weights. The values
     * returned by this method may change depending on the parameters set for
     * the model.
     *
     * @return <tt>true</tt> if the model supports weighted data, <tt>false</tt>
     * otherwise
     */
    default public boolean supportsWeightedData()
    {
        return false;
    }
    
    public Bicluster clone();
}
