package jsat.datatransform;

import jsat.classifiers.DataPoint;

/**
 * An In Place Transform is one that has the same number of categorical and
 * numeric features as the input. This means it can mutableTransform the input data point
 * instead of allocating a new one, which can reduce overhead on memory
 * allocations. This can be useful when performing many data transforms in cross
 * validation or when processing new examples in an environment that is applying
 * an already learned model.
 * <br><br> This interface is assumed that it will be applied to numeric
 * features. Incase this is not true, a {@link #mutatesNominal() } method is
 * provided for the implementation to indicate otherwise.
 *
 * @author Edward Raff
 */
public interface InPlaceTransform extends DataTransform
{

    /**
     * Mutates the given data point. This causes side effects, altering the data
     * point to have the same value as the output of 
     * {@link #transform(jsat.classifiers.DataPoint) }.
     *
     * @param dp the data point to alter
     */
    public void mutableTransform(DataPoint dp);

    /**
     * By default returns {@code false}. Only returns true if this transform
     * will mutableTransform the nominal feature values of a data point.
     *
     * @return {@code true} if nominal feature values are mutated, {@code false}
     * otherwise.
     */
    public boolean mutatesNominal();
}
