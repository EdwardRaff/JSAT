package jsat.math;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * Interface for representing a function that takes a vector as input should
 * return a vector as the output.
 *
 * @author Edward Raff
 */
public interface FunctionVec
{
    /**
     * Computes the function value given the input {@code x}
     * @param x the input to compute the output from
     * @return the vector containing the results
     */
    default public Vec f(double... x)
    {
        return f(DenseVector.toDenseVec(x));
    }
    
    /**
     * Computes the function value given the input {@code x}
     * @param x the input to compute the output from
     * @return the vector containing the results
     */
    default public Vec f(Vec x)
    {
        return f(x, null);
    }
    
    /**
     * Computes the function value given the input {@code x}
     * @param x the input to compute the output from
     * @param s the vector to store the result in, or {@code null} if a new 
     * vector should be allocated
     * @return the vector containing the results. This is the same object as 
     * {@code s} if {@code s} is not {@code null}
     */
    default public Vec f(Vec x, Vec s)
    {
        return f(x, s, false);
    }
    
    /**
     * Computes the function value given the input {@code x} 
     * @param x the input to compute the output from
     * @param s the vector to store the result in, or {@code null} if a new 
     * vector should be allocated
     * @param parallel {@code true} if multiple threads should be used for
     * evaluation, {@code false} if only a single thread should.
     * @return the vector containing the results. This is the same object as 
     * {@code s} of {@code s} is not {@code null}
     */
    public Vec f(Vec x, Vec s, boolean parallel);
}
