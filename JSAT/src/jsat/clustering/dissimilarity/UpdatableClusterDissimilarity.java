
package jsat.clustering.dissimilarity;

/**
 * This interface extends the contract of a {@link ClusterDissimilarity} for 
 * more efficient computation. This contract indicates that the dissimilarity 
 * measure being used can be computed in an online fashion, and that the 
 * dissimilarity matrix can be updated to reflect the dissimilarity for a 
 * new merged cluster. 
 * 
 * @author Edward Raff
 */
public interface UpdatableClusterDissimilarity extends ClusterDissimilarity
{
    /**
     * Provides the notion of dissimilarity between two sets of points, that may
     * not have the same number of points. This is done using a matrix 
     * containing all pairwise distance computations between all points. This 
     * distance matrix will then be updated at each iteration and merging, 
     * leaving empty space in the matrix. The updates will be done by the 
     * clustering algorithm. Implementing this interface indicates that this 
     * dissimilarity measure can be accurately computed in an updatable manner 
     * that is compatible with a Lance–Williams update. 
     * 
     * @param i the index of cluster <tt>i</tt>'s distance in the original data set
     * @param ni the number of items in the cluster represented by <tt>i</tt>
     * @param j the index of cluster <tt>j</tt>'s distance in the original data set
     * @param nj the number of items in the cluster represented by <tt>j</tt>
     * @param distanceMatrix a distance matrix originally created by 
     * {@link AbstractClusterDissimilarity#createDistanceMatrix(jsat.DataSet, 
     * jsat.clustering.dissimilarity.ClusterDissimilarity) }
     * @return a value &gt;= 0 that describes the dissimilarity of the two 
     * clusters. The larger the value, the more different the two clusterings are. 
     */
    public double dissimilarity(int i, int ni, int j, int nj, double[][] distanceMatrix);
    
    /**
     * Provides the notion of dissimilarity between two sets of points, that may
     * not have the same number of points. This is done using a matrix 
     * containing all pairwise distance computations between all points. This 
     * distance matrix will then be updated at each iteration and merging, 
     * leaving empty space in the matrix. The updates will be done by the 
     * clustering algorithm. Implementing this interface indicates that this 
     * dissimilarity measure can be accurately computed in an updatable manner 
     * that is compatible with a Lance–Williams update. <br>
     * 
     * This computes the dissimilarity of the union of clusters i and j, 
     * (C<sub>i</sub> &cup; C<sub>j</sub>), with the cluster k. This method is 
     * used by other algorithms to perform an update of the distance matrix in 
     * an efficient manner. 
     * 
     * @param i the index of cluster <tt>i</tt>'s distance in the original data set
     * @param ni the number of items in the cluster represented by <tt>i</tt>
     * @param j the index of cluster <tt>j</tt>'s distance in the original data set
     * @param nj the number of items in the cluster represented by <tt>j</tt>
     * @param k the index of cluster <tt>k</tt>'s distance in the original data set
     * @param nk the number of items in the cluster represented by <tt>k</tt>
     * a distance matrix originally created by 
     * {@link AbstractClusterDissimilarity#createDistanceMatrix(jsat.DataSet, 
     * jsat.clustering.dissimilarity.ClusterDissimilarity) }
     * @return a value &gt;= 0 that describes the dissimilarity of the union of 
     * two clusters with a third cluster. The larger the value, the more 
     * different the resulting clusterings are. 
     */
    public double dissimilarity(int i, int ni, int j, int nj, int k, int nk, double[][] distanceMatrix); 
    
    @Override
    public UpdatableClusterDissimilarity clone();
}
