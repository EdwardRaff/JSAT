
package jsat.math;

import java.io.Serializable;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * This functional interface defines a function over a vector input and returns
 * a scalar output.
 *
 * @author Edward Raff
 */
public interface Function extends Serializable
{
    /**
     * Evaluates the given function for the specified input vector. 
     * @param x the input to the function
     * @return the scalar output of this function
     */
    default public double f(double... x)
    {
        return f(DenseVector.toDenseVec(x));
    }
    
    /**
     * Evaluates the given function for the specified input vector. 
     * @param x the input to the function
     * @return the scalar output of this function
     */
    default public double f(Vec x)
    {
        return f(x, false);
    }
    
    /**
     * Evaluates the given function for the specified input vector. 
     * @param x the input to the function
     * @param parallel {@code true} if the function should be evaluated with
     * multiple threads, or {@code false} to use a single thread.
     * @return the scalar output of this function
     */
    public double f(Vec x, boolean parallel);
}
