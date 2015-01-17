package jsat.datatransform;

import jsat.classifiers.DataPoint;

/**
 * This interface behaves exactly as {@link InPlaceTransform} specifies, with 
 * the addition of an in-place "reverse" method that can be used to alter any 
 * given transformed data point back into an <i>approximation</i> of the 
 * original vector, without having to new vector object, but altering the one
 * given.
 *
 * @author Edward Raff
 */
public interface InPlaceInvertibleTransform extends InPlaceTransform, InvertibleTransform
{

    /**
     * Mutates the given data point. This causes side effects, altering the data
     * point to have the same value as the output of 
     * {@link #inverse(jsat.classifiers.DataPoint) }
     *
     * @param dp the data point to alter with an inverse transformation 
     */
    public void mutableInverse(DataPoint dp);

    @Override
    public InPlaceInvertibleTransform clone();

}
