
package jsat.clustering.dissimilarity;

import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;

/**
 * A base class for Dissimilarity measures that are build ontop the use of some {@link DistanceMetric distance metric}. 
 * 
 * @author Edward Raff
 */
public abstract class DistanceMetricDissimilarity extends AbstractClusterDissimilarity implements DistanceMetric
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
        return dist(a.getNumericalValues(), b.getNumericalValues());
    }    
    
    
    @Override
    public double dist(Vec a, Vec b)
    {
        return dm.dist(a, b);
    }

    @Override
    public boolean isSymmetric()
    {
        return dm.isSymmetric();
    }

    @Override
    public boolean isSubadditive()
    {
        return dm.isSubadditive();
    }

    @Override
    public boolean isIndiscemible()
    {
        return dm.isIndiscemible();
    }

    @Override
    public double metricBound()
    {
        return dm.metricBound();
    }
    
    @Override
    abstract public DistanceMetricDissimilarity clone();
}
