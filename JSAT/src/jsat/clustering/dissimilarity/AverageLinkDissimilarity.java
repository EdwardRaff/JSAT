package jsat.clustering.dissimilarity;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;

/**
 * Also known as Group-Average Agglomerative Clustering (GAAC), this measure 
 * computer the dissimilarity by summing the distances between all possible 
 * data point pairs in the union of the clusters. 
 * 
 * @author Edward Raff
 */
public class AverageLinkDissimilarity extends DistanceMetricDissimilarity
{

    public AverageLinkDissimilarity(DistanceMetric dm)
    {
        super(dm);
    }
    
    @Override
    public DistanceMetricDissimilarity clone()
    {
        return new AverageLinkDissimilarity(dm.clone());
    }

    @Override
    public double dissimilarity(List<DataPoint> a, List<DataPoint> b)
    {
        double disSum = 0;
        
        int allSize = a.size()+b.size();
        
        List<DataPoint> allPoints = new ArrayList<DataPoint>(allSize);
        allPoints.addAll(a);
        allPoints.addAll(b);
        
        for(int i = 0; i < allPoints.size(); i++)
            for(int j = i+1; j < allPoints.size(); j++)
                disSum += distance(allPoints.get(i), allPoints.get(j));
        
        return disSum/(allSize*(allSize-1));
    }

    @Override
    public double dissimilarity(Set<Integer> a, Set<Integer> b, double[][] distanceMatrix)
    {
        double disSum = 0;
        
        int allSize = a.size()+b.size();
        
        int[] allPoints = new int[allSize];
        int z = 0;
        for(int val : a)
            allPoints[z++] = val;
        for(int val : b)
            allPoints[z++] = val;
        
        for(int i = 0; i < allPoints.length; i++)
            for(int j = i+1; j < allPoints.length; j++)
                disSum += getDistance(distanceMatrix, allPoints[i], allPoints[j]);
        
        return disSum/(allSize*(allSize-1));
    }
    
}
