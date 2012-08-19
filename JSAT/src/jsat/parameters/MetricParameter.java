package jsat.parameters;

import jsat.linear.distancemetrics.DistanceMetric;

/**
 * A MetricParameter is a parameter controller for the {@link DistanceMetric} 
 * used by the current algorithm. 
 * 
 * @author Edward Raff
 */
public abstract class MetricParameter extends Parameter
{
    /**
     * Sets the distance metric that should be sued
     * @param val the distance metric to use
     * @return <tt>true</tt> if the metric is valid and was set, <tt>false</tt> 
     * if the metric was not valid for this learner and ignored. 
     */
    abstract public boolean setMetric(DistanceMetric val);
    
    /**
     * Returns the distance metric that was used for this learner 
     * @return the current distance metric
     */
    abstract public DistanceMetric getMetric();
}
