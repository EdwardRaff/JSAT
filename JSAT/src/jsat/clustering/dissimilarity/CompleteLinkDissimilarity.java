
package jsat.clustering.dissimilarity;

import java.util.List;
import java.util.Set;
import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;

/**
 * Measures the dissimilarity of two clusters by returning the value of the 
 * maximal dissimilarity of any two pairs of data points where one is from 
 * each cluster. 
 * 
 * @author Edward Raff
 */
public class CompleteLinkDissimilarity extends DistanceMetricDissimilarity
{

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

}
