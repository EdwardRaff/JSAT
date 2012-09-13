package jsat.math;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * Simple base abstract class for implementing a {@link Function} by 
 * implementing {@link #f(double[]) } to call the vector version. 
 * 
 * @author Edward Raff
 */
public abstract class FunctionBase implements Function
{

    @Override
    public double f(double... x)
    {
        return f(DenseVector.toDenseVec(x));
    }
    
}
