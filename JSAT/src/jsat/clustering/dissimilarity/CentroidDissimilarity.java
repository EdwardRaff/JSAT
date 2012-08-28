
package jsat.clustering.dissimilarity;

import java.util.List;
import java.util.Set;
import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * Average similarity of all data point pairs between clusters, inter-cluster 
 * pairs are ignored. 
 * 
 * @author Edward Raff
 */
public class CentroidDissimilarity extends DistanceMetricDissimilarity implements UpdatableClusterDissimilarity
{
    /**
     * Creates a new CentroidDissimilarity that used the {@link EuclideanDistance}
     */
    public CentroidDissimilarity()
    {
        this(new EuclideanDistance());
    }
    
    /**
     * Creates a new CentroidDissimilarity
     * @param dm the distance measure to use between individual points
     */
    public CentroidDissimilarity(DistanceMetric dm)
    {
        super(dm);
    }
    
    @Override
    public DistanceMetricDissimilarity clone()
    {
        return new CentroidDissimilarity(dm.clone());
    }

    @Override
    public double dissimilarity(List<DataPoint> a, List<DataPoint> b)
    {
        double sumDIss = 0;

        for (DataPoint ai : a)
            for (DataPoint bi : b)
                sumDIss += distance(ai, bi);

        return sumDIss/(a.size()*b.size());
    }

    @Override
    public double dissimilarity(Set<Integer> a, Set<Integer> b, double[][] distanceMatrix)
    {
        double sumDiss = 0;

        for (int ai : a)
            for (int bi : b)
                sumDiss += getDistance(distanceMatrix, ai, bi);

        return sumDiss/(a.size()*b.size());
    }

    @Override
    public double dissimilarity(int i, int ni, int j, int nj, double[][] distanceMatrix)
    {
        return getDistance(distanceMatrix, i, j);
    }

    @Override
    public double dissimilarity(int i, int ni, int j, int nj, int k, int nk, double[][] distanceMatrix)
    {
        double iPj = ni+nj;
        double ai = ni/iPj;
        double aj = nj/iPj;
        double b = - ni * nj / iPj*iPj;
        
        return ai* getDistance(distanceMatrix, i, k) + aj * getDistance(distanceMatrix, j, k) + b * getDistance(distanceMatrix, i, j);
    }

    
}
