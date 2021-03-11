/*
 * This code contributed in the public domain. 
 */
package jsat.datatransform;

import jsat.classifiers.DataPoint;

/**
 * This interface is meant to be used for convinence when you wish to apply a
 * transformation to a data set using the Java 8 lambda features. It is for
 * transformations that do not need to be trained on any data, or where all
 * training has been done in advance.
 *
 * @author Edward Raff
 */
public interface FixedDataTransform
{

    /**
     * Returns a new data point that is a transformation of the original data
     * point. This new data point is a different object, but may contain the
     * same references as the original data point. It is not guaranteed that you
     * can mutate the transformed point without having a side effect on the
     * original point.
     *
     * @param dp the data point to apply a transformation to
     * @return a transformed data point
     */
    public DataPoint transform(DataPoint dp);
}
