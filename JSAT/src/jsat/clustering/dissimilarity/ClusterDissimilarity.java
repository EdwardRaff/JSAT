package jsat.clustering.dissimilarity;

import java.util.List;
import java.util.Set;
import jsat.classifiers.DataPoint;

/**
 * This interface provides the basic contract for measuring the dissimilarity 
 * between two clusters, and intended for use in Hierarchical Agglomerative 
 * Clustering. 
 * 
 * @author Edward Raff
 */
public interface ClusterDissimilarity
{
    /**
     * Provides the notion of distance, or dissimilarity, between two data points
     * @param a the first data point
     * @param b the second data point
     * @return a value &gt;= 0 that is a measure of the difference between the 
     * two points. The closer to zero, the more similar the points are. 
     */
    public double distance(DataPoint a, DataPoint b);
    /**
     * Provides the notion of dissimilarity between two sets of points, that may
     * not have the same number of points. 
     * 
     * @param a the first cluster of points
     * @param b the second cluster of points
     * @return a value &gt;= 0 that describes the dissimilarity of the two 
     * clusters. The larger the value, the more different the two clusterings are. 
     */
    public double dissimilarity(List<DataPoint> a, List<DataPoint> b);
    
    /**
     * Provides the notion of dissimilarity between two sets of points, that may
     * not have the same number of points. This is done using a matrix 
     * containing all pairwise distance computations between all points. 
     * 
     * @param a the first set of indices of the original data set that are in a 
     * cluster, which map to <i>distanceMatrix</i>
     * @param b the second set of indices of the original data set that are in a
     * cluster, which map to <i>distanceMatrix</i>
     * @param distanceMatrix the upper triangual distance matrix as created by 
     * {@link AbstractClusterDissimilarity#createDistanceMatrix(jsat.DataSet, jsat.clustering.dissimilarity.ClusterDissimilarity) }
     * @return a value &gt;= 0 that describes the dissimilarity of the two 
     * clusters. The larger the value, the more different the two clusterings are. 
     */
    public double dissimilarity(Set<Integer> a, Set<Integer> b, double[][] distanceMatrix);
    
    public ClusterDissimilarity clone();
}
