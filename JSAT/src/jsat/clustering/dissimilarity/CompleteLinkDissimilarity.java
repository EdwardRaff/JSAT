
package jsat.clustering.dissimilarity;

import java.util.List;
import java.util.Set;
import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * Measures the dissimilarity of two clusters by returning the value of the 
 * maximal dissimilarity of any two pairs of data points where one is from 
 * each cluster. 
 * 
 * @author Edward Raff
 */
public class CompleteLinkDissimilarity extends DistanceMetricDissimilarity implements UpdatableClusterDissimilarity
{
    /**
     * Creates a new CompleteLinkDissimilarity using the {@link EuclideanDistance}
     */
    public CompleteLinkDissimilarity()
    {
        this(new EuclideanDistance());
    }
    
    /**
     * Creates a new CompleteLinkDissimilarity 
     * @param dm the distance metric to use between individual points
     */
    public CompleteLinkDissimilarity(DistanceMetric dm)
    {
        super(dm);
    }
    
    @Override
    public DistanceMetricDissimilarity clone()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double dissimilarity(List<DataPoint> a, List<DataPoint> b)
    {
        double maxDiss = Double.MIN_VALUE;

        double tmpDist;
        for (DataPoint ai : a)
            for (DataPoint bi : b)
                if ((tmpDist = distance(ai, bi)) > maxDiss)
                    maxDiss = tmpDist;

        return maxDiss;
    }

    @Override
    public double dissimilarity(Set<Integer> a, Set<Integer> b, double[][] distanceMatrix)
    {
        double maxDiss = Double.MIN_VALUE;

        for (int ai : a)
            for (int bi : b)
                if (getDistance(distanceMatrix, ai, bi) > maxDiss)
                    maxDiss = getDistance(distanceMatrix, ai, bi);

        return maxDiss;
    }

    @Override
    public double dissimilarity(int i, int ni, int j, int nj, double[][] distanceMatrix)
    {
        return getDistance(distanceMatrix, i, j);
    }

    @Override
    public double dissimilarity(int i, int ni, int j, int nj, int k, int nk, double[][] distanceMatrix)
    {
        return Math.max(getDistance(distanceMatrix, i, k), getDistance(distanceMatrix, j, k));
    }

}
