
package jsat.clustering.dissimilarity;

import java.util.List;
import java.util.Set;
import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;

/**
 * Measures the dissimilarity of two clusters by returning the minimum 
 * dissimilarity between the two closest data points from the clusters, ie: 
 * the minimum distance needed to link the two clusters. 
 * 
 * @author Edward Raff
 */
public class SingleLinkDissimilarity extends DistanceMetricDissimilarity
{

    public SingleLinkDissimilarity(DistanceMetric dm)
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

}
