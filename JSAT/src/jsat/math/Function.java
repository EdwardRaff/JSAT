
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
    
    /**
     * Returns a new function that approximates the derivative of the given one
     * via numerical forward difference approximation.
     *
     * @param f the function to approximate the derivative of
     * @return a function that will return an estimate of the derivative
     */
    public static FunctionVec forwardDifference(Function f)
    {
        FunctionVec fP = (Vec x, Vec s, boolean parallel) -> 
        {
            if(s == null)
            {
                s = x.clone();
                s.zeroOut();
            }
            
            double sqrtEps = Math.sqrt(2e-16);
            
            double f_x = f.f(x, parallel);
            
            Vec x_ph = x.clone();
            
            for(int i = 0; i < x.length(); i++)
            {
                double h = Math.max(Math.abs(x.get(i))*sqrtEps, 1e-5);
                x_ph.set(i, x.get(i)+h);
                double f_xh = f.f(x_ph, parallel);
                s.set(i, (f_xh-f_x)/h);//set derivative estimate
                x_ph.set(i, x.get(i));
            }
            
            return s;
        };
        return fP;
    }
}
