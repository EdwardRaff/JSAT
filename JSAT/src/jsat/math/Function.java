
package jsat.math;

import java.io.Serializable;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * This functional interface defines a function over a vector input
 * @author Edward Raff
 */
public interface Function extends Serializable
{
    default public double f(double... x)
    {
        return f(DenseVector.toDenseVec(x));
    }
    
    public double f(Vec x);
}
