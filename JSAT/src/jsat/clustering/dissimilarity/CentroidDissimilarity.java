
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
    public CentroidDissimilarity(final DistanceMetric dm)
    {
        super(dm);
    }
    
    @Override
    public CentroidDissimilarity clone()
    {
        return new CentroidDissimilarity(dm.clone());
    }

    @Override
    public double dissimilarity(final List<DataPoint> a, final List<DataPoint> b)
    {
        double sumDIss = 0;

        for (final DataPoint ai : a) {
          for (final DataPoint bi : b) {
            sumDIss += distance(ai, bi);
          }
        }

        return sumDIss/(a.size()*b.size());
    }

    @Override
    public double dissimilarity(final Set<Integer> a, final Set<Integer> b, final double[][] distanceMatrix)
    {
        double sumDiss = 0;

        for (final int ai : a) {
          for (final int bi : b) {
            sumDiss += getDistance(distanceMatrix, ai, bi);
          }
        }

        return sumDiss/(a.size()*b.size());
    }

    @Override
    public double dissimilarity(final int i, final int ni, final int j, final int nj, final double[][] distanceMatrix)
    {
        return getDistance(distanceMatrix, i, j);
    }

    @Override
    public double dissimilarity(final int i, final int ni, final int j, final int nj, final int k, final int nk, final double[][] distanceMatrix)
    {
        final double iPj = ni+nj;
        final double ai = ni/iPj;
        final double aj = nj/iPj;
        final double b = - ni * nj / iPj*iPj;
        
        return ai* getDistance(distanceMatrix, i, k) + aj * getDistance(distanceMatrix, j, k) + b * getDistance(distanceMatrix, i, j);
    }

    
}
