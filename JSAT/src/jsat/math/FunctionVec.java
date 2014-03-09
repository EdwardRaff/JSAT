package jsat.math;

import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;

/**
 * Interface for representing a function that should return a vector as the 
 * result. 
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
    public Vec f(double... x);
    
    /**
     * Computes the function value given the input {@code x}
     * @param x the input to compute the output from
     * @return the vector containing the results
     */
    public Vec f(Vec x);
    
    /**
     * Computes the function value given the input {@code x}
     * @param x the input to compute the output from
     * @param s the vector to store the result in, or {@code null} if a new 
     * vector should be allocated
     * @return the vector containing the results. This is the same object as 
     * {@code s} if {@code s} is not {@code null}
     */
    public Vec f(Vec x, Vec s);
    
    /**
     * Computes the function value given the input {@code x} 
     * @param x the input to compute the output from
     * @param s the vector to store the result in, or {@code null} if a new 
     * vector should be allocated
     * @param ex the source of threads to use for the computation
     * @return the vector containing the results. This is the same object as 
     * {@code s} of {@code s} is not {@code null}
     */
    public Vec f(Vec x, Vec s, ExecutorService ex);
}
