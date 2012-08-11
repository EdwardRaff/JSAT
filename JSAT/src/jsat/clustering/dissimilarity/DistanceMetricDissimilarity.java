
package jsat.clustering.dissimilarity;

import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;

/**
 * A base class for Dissimilarity measures that are build ontop the use of some {@link DistanceMetric distance metric}. 
 * 
 * @author Edward Raff
 */
public abstract class DistanceMetricDissimilarity extends AbstractClusterDissimilarity 
{
    /**
     * The distance metric that will back this dissimilarity measure. 
     */
    protected final DistanceMetric dm;

    public DistanceMetricDissimilarity(DistanceMetric dm)
    {
        this.dm = dm;
    }

    @Override
    public double distance(DataPoint a, DataPoint b)
    {
        return dm.dist(a.getNumericalValues(), b.getNumericalValues());
    }    
    
    @Override
    abstract public DistanceMetricDissimilarity clone();
}
