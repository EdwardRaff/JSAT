
package jsat.clustering.dissimilarity;

import java.util.List;
import java.util.Set;
import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;

/**
 * Average similarity of all data point pairs between clusters, inter-cluster 
 * pairs are ignored. 
 * 
 * @author Edward Raff
 */
public class CentroidDissimilarity extends DistanceMetricDissimilarity
{

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

    
}
