
package jsat.datatransform;

import java.io.Serializable;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.exceptions.FailedToFitException;

/**
 * A pre-processing step may be desirable before training. If a pre-processing 
 * step is used, it is necessary to also apply the same transform on the input 
 * being sent to the learning algorithm. This interface provides the needed
 * mechanism. <br>
 * A transform may or may not require training, it could be fully specified at
 * construction, or learned from the data set. Learning is done via the
 * {@link #fit(jsat.DataSet) fit method}. Many DataTransforms will include a
 * constructor that takes a dataset as a parameter. These transforms will fit
 * the data when constructed, and exist for convenience.
 *
 * @author Edward Raff
 */
public interface DataTransform extends Cloneable, Serializable
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
    
    /**
     * Fits this transform to the given dataset. Some transforms can only be
     * learned from classification or regression datasets. If an incompatible
     * dataset type is given, a {@link FailedToFitException} exception may be
     * thrown.
     *
     * @param data the dataset to fir this transform to
     * @throws FailedToFitException if the dataset type is not compatible with
     * the transform
     */
    public void fit(DataSet data);
    
    public DataTransform clone();
}
