
package jsat.clustering.dissimilarity;

import java.util.List;
import java.util.Set;
import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * Measures the dissimilarity of two clusters by returning the minimum 
 * dissimilarity between the two closest data points from the clusters, ie: 
 * the minimum distance needed to link the two clusters. 
 * 
 * @author Edward Raff
 */
public class SingleLinkDissimilarity extends LanceWilliamsDissimilarity implements UpdatableClusterDissimilarity
{
    /**
     * Creates a new SingleLinkDissimilarity using the {@link EuclideanDistance}
     */
    public SingleLinkDissimilarity()
    {
        this(new EuclideanDistance());
    }

    /**
     * Creates a new SingleLinkDissimilarity 
     * @param dm the distance metric to use between individual points
     */
    public SingleLinkDissimilarity(DistanceMetric dm)
    {
        super(dm);
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public SingleLinkDissimilarity(SingleLinkDissimilarity toCopy)
    {
        this(toCopy.dm.clone());
    }
    
    @Override
    public SingleLinkDissimilarity clone()
    {
        return new SingleLinkDissimilarity(this);
    }

    @Override
    public double dissimilarity(List<DataPoint> a, List<DataPoint> b)
    {
        double minDiss = Double.MAX_VALUE;

        double tmpDist;
        for (DataPoint ai : a)
            for (DataPoint bi : b)
                if ((tmpDist = distance(ai, bi)) < minDiss)
                    minDiss = tmpDist;

        return minDiss;
    }

    @Override
    public double dissimilarity(Set<Integer> a, Set<Integer> b, double[][] distanceMatrix)
    {
        double minDiss = Double.MAX_VALUE;

        for (int ai : a)
            for (int bi : b)
                if (getDistance(distanceMatrix, ai, bi) < minDiss)
                    minDiss = getDistance(distanceMatrix, ai, bi);

        return minDiss;
    }

    @Override
    public double dissimilarity(int i, int ni, int j, int nj, double[][] distanceMatrix)
    {
        return getDistance(distanceMatrix, i, j);
    }

    @Override
    public double dissimilarity(int i, int ni, int j, int nj, int k, int nk, double[][] distanceMatrix)
    {
        return Math.min(getDistance(distanceMatrix, i, k), getDistance(distanceMatrix, j, k));
    }

    @Override
    public double dissimilarity(int ni, int nj, int nk, double d_ij, double d_ik, double d_jk)
    {
        return Math.min(d_ik, d_jk);
    }

    @Override
    protected double aConst(boolean iFlag, int ni, int nj, int nk)
    {
        return 0.5;
    }

    @Override
    protected double bConst(int ni, int nj, int nk)
    {
        return 0;
    }

    @Override
    protected double cConst(int ni, int nj, int nk)
    {
        return -0.5;
    }

}
