package jsat.clustering.dissimilarity;

import java.util.*;
import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * Also known as Group-Average Agglomerative Clustering (GAAC), this measure 
 * computer the dissimilarity by summing the distances between all possible 
 * data point pairs in the union of the clusters. 
 * 
 * @author Edward Raff
 */
public class AverageLinkDissimilarity extends DistanceMetricDissimilarity implements UpdatableClusterDissimilarity
{
    /**
     * Creates a new AverageLinkDissimilarity using the {@link EuclideanDistance}
     */
    public AverageLinkDissimilarity()
    {
        this(new EuclideanDistance());
    }

    /**
     * Creates a new AverageLinkDissimilarity 
     * @param dm the distance measure to use on individual points
     */
    public AverageLinkDissimilarity(final DistanceMetric dm)
    {
        super(dm);
    }
    
    @Override
    public AverageLinkDissimilarity clone()
    {
        return new AverageLinkDissimilarity(dm.clone());
    }

    @Override
    public double dissimilarity(final List<DataPoint> a, final List<DataPoint> b)
    {
        double disSum = 0;
        
        final int allSize = a.size()+b.size();
        
        final List<DataPoint> allPoints = new ArrayList<DataPoint>(allSize);
        allPoints.addAll(a);
        allPoints.addAll(b);
        
        for(int i = 0; i < allPoints.size(); i++) {
          for (int j = i+1; j < allPoints.size(); j++) {
            disSum += distance(allPoints.get(i), allPoints.get(j));
          }
        }
        
        return disSum/(allSize*(allSize-1));
    }

    @Override
    public double dissimilarity(final Set<Integer> a, final Set<Integer> b, final double[][] distanceMatrix)
    {
        double disSum = 0;
        
        final int allSize = a.size()+b.size();
        
        final int[] allPoints = new int[allSize];
        int z = 0;
        for(final int val : a) {
          allPoints[z++] = val;
        }
        for(final int val : b) {
          allPoints[z++] = val;
        }
        
        for(int i = 0; i < allPoints.length; i++) {
          for (int j = i+1; j < allPoints.length; j++) {
            disSum += getDistance(distanceMatrix, allPoints[i], allPoints[j]);
          }
        }
        
        return disSum/(allSize*(allSize-1));
    }

    @Override
    public double dissimilarity(final int i, final int ni, final int j, final int nj, final double[][] distanceMatrix)
    {
        return getDistance(distanceMatrix, i, j);
    }

    @Override
    public double dissimilarity(final int i, final int ni, final int j, final int nj, final int k, final int nk, final double[][] distanceMatrix)
    {
        final double ai = ni/(double)(ni+nj);
        final double aj = nj/(double)(ni+nj);
        return ai * getDistance(distanceMatrix, i, k) + aj * getDistance(distanceMatrix, j, k);
    }
    
}
