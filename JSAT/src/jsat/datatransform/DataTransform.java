
package jsat.datatransform;

import jsat.classifiers.DataPoint;

/**
 * A pre-processing step may be desirable before training. If a pre-processing 
 * step is used, it is necessary to also apply the same transform on the input 
 * being sent to the learning algorithm. This interface provides the needed 
 * mechanism. <br>
 * A transform may or may not require training, it could be fully specified at 
 * construction, or learned from the data set. 
 * 
 * @author Edward Raff
 */
public interface DataTransform extends Cloneable
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
    
    public DataTransform clone();
}
