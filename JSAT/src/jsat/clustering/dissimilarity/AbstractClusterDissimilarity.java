package jsat.clustering.dissimilarity;

import jsat.DataSet;

/**
 * This base class does not currently provide any inheritable functionality, but
 * stores static methods. 
 * 
 * @author Edward Raff
 */
public abstract class AbstractClusterDissimilarity implements ClusterDissimilarity
{

    /**
     * A convenience method. If the <i>distanceMatrix</i> was created with
     * {@link #createDistanceMatrix(jsat.DataSet, jsat.clustering.dissimilarity.ClusterDissimilarity)
     * }, then this method will return the appropriate value for the desired
     * index.
     *
     * @param distanceMatrix the distance matrix to query from
     * @param i the first index
     * @param j the second index
     * @return the correct value from the distance matrix from the index given
     * as if the distance matrix was of full form
     */
    public static double getDistance(double[][] distanceMatrix, int i, int j)
    {
        if (i > j)
        {
            int tmp = j;
            j = i;
            i = tmp;
        }

        return distanceMatrix[i][j - i - 1];
    }
    
    /**
     * A convenience method. If the <i>distanceMatrix</i> was created with
     * {@link #createDistanceMatrix(jsat.DataSet, jsat.clustering.dissimilarity.ClusterDissimilarity)
     * }, then this method will set the appropriate value for the desired
     * index.
     * 
     * @param distanceMatrix the distance matrix to query from
     * @param i the first index
     * @param j the second index
     * @param dist the new distance value to store in the matrix
     */
    public static void setDistance(double[][] distanceMatrix, int i, int j, double dist)
    {
        if (i > j)
        {
            int tmp = j;
            j = i;
            i = tmp;
        }

        distanceMatrix[i][j - i - 1] = dist;
    }

    /**
     * Creates an upper triangular matrix containing the distance between all
     * points in the data set. The main diagonal will contain all zeros, since
     * the distance between a point and itself is always zero. This main
     * diagonal is not stored, and is implicit <br> To save space, the matrix is
     * staggered, and is of a size such that all elements to the left of the
     * main diagonal are not present. <br> To compute the index into the
     * returned array for the index [i][j], the values should be switched such
     * that i &ge; j, and accessed as [i][j-i-1]
     *
     * @param dataSet the data set to create distance matrix for
     * @param cd the cluster dissimilarity measure to use
     * @return a upper triangular distance matrix
     */
    public static double[][] createDistanceMatrix(DataSet dataSet, ClusterDissimilarity cd)
    {
        double[][] distances = new double[dataSet.getSampleSize()][];


        for (int i = 0; i < distances.length; i++)
        {
            distances[i] = new double[dataSet.getSampleSize() - i - 1];
            for (int j = i + 1; j < distances.length; j++)
                distances[i][j - i - 1] = cd.distance(dataSet.getDataPoint(i), dataSet.getDataPoint(j));
        }

        return distances;
    }

    @Override
    abstract public ClusterDissimilarity clone();
}
