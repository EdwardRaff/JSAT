
package jsat.math;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public interface IndexFunction extends Function
{
    /**
     * An index function, meant to be applied to vectors where the 
     * value to be computed may vary based on the position in the 
     * vector of the value. 
     * 
     * @param value the value at the specified index
     * @param index the index the value is from
     * @return the computed result. If a negative index was given, 
     * the function should return 0.0 if indexFunc(0,indx) would 
     * return zero for all valid indices. If this is not the case, any non zero value should be returned. 
     * 
     */
    abstract public double indexFunc(double value, int index);

    @Override
    default public double f(Vec x, boolean parallel)
    {
        return indexFunc(x.get(0), (int)x.get(1));
    }    
}
