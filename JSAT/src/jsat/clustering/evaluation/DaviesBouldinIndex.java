
package jsat.clustering.evaluation;

import java.util.ArrayList;
import java.util.List;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.ClustererBase;
import jsat.linear.MatrixStatistics;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * A measure for evaluating the quality of a clustering by measuring the 
 * distances of points to their centroids. 
 * 
 * @author Edward Raff
 */
public class DaviesBouldinIndex implements ClusterEvaluation
{
    private final DistanceMetric dm;

    /**
     * Creates a new DaviesBouldinIndex using the {@link EuclideanDistance}.
     */
    public DaviesBouldinIndex()
    {
        this(new EuclideanDistance());
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public DaviesBouldinIndex(final DaviesBouldinIndex toCopy)
    {
        this(toCopy.dm.clone());
    }
    
    /**
     * Creates a new DaviesBouldinIndex 
     * @param dm the distance measure to use when computing 
     */
    public DaviesBouldinIndex(final DistanceMetric dm) 
    {
        this.dm = dm;
    }
    
    @Override
    public double evaluate(final int[] designations, final DataSet dataSet) 
    {
        return evaluate(ClustererBase.createClusterListFromAssignmentArray(designations, dataSet));
    }
    
    @Override
    public double evaluate(final List<List<DataPoint>> dataSets) 
    {
        /**
         * Forumal for the DB measure
         * 
         *                              /sigma   +  sigma \ 
         *        1  __ n               |     i          j| 
         * DB  =  - \         max       |-----------------| 
         *        n /__ i = 1    i neq j|    d(c ,c )     | 
         *                              \       i  j      /
         * 
         * where 
         *   c_i is the centroid of cluster i
         *   sigma_i is the average distance of over point in cluster i to its centroid
         *   d(,) is a distance function
         *   n is the number of clusters
         */
        final List<Vec> centroids = new ArrayList<Vec>(dataSets.size());
        final double[] avrgCentriodDist = new double[dataSets.size()];
        
        for(int i = 0; i < dataSets.size(); i++)
        {
            final Vec mean = MatrixStatistics.meanVector(new SimpleDataSet(dataSets.get(i)));
            centroids.add(mean);
        
            for(final DataPoint dp : dataSets.get(i)) {
              avrgCentriodDist[i] += dm.dist(dp.getNumericalValues(), mean);
            }
            avrgCentriodDist[i]/=dataSets.get(i).size();
        }
        
        double dbIndex = 0;
        
        for(int i = 0; i < dataSets.size(); i++)
        {
            double maxPenalty = Double.NEGATIVE_INFINITY;
            for(int j = 0; j < dataSets.size(); j++)
            {
                if(j == i) {
                  continue;
                }
                final double penalty = (avrgCentriodDist[i] + avrgCentriodDist[j])/dm.dist(centroids.get(i), centroids.get(j));
                maxPenalty = Math.max(maxPenalty, penalty);
            }
            dbIndex += maxPenalty;
        }
        
        return dbIndex / dataSets.size();
    }

    @Override
    public DaviesBouldinIndex clone()
    {
        return new DaviesBouldinIndex(this);
    }
}
