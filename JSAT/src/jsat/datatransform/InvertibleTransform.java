package jsat.datatransform;

import jsat.classifiers.DataPoint;

/**
 * A InvertibleTransform is one in which any given transformed vector can be 
 inverse to recover an <i>approximation</i> of the original vector when using 
 * a transform that implements this interface. It may not be possible to 
 * perfectly reproduce the original data point: ie, this process may not be 
 * loss-less. 
 * 
 * @author Edward Raff
 */
public interface InvertibleTransform extends DataTransform
{

    /**
     * Applies the inverse or "reverse" transform to approximately undo the
     * effect of {@link #transform(jsat.classifiers.DataPoint) } to recover an
     * approximation of the original data point.
     *
     * @param dp the transformed data point
     * @return the original data point, or a reasonable approximation
     */
    public DataPoint inverse(DataPoint dp);

    @Override
    public InvertibleTransform clone();
}
