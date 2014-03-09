package jsat.math;

import java.util.concurrent.ExecutorService;
import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * Interface for representing a function that should return a Matrix object as 
 * the result. 
 * 
 * @author Edward Raff
 */
public interface FunctionMat
{
    /**
     * Computes a matrix based on multivariate input
     * @param x the variables to evaluate as part of the function
     * @return the matrix output of the function
     */
    public Matrix f(double... x);
    
    /**
     * Computes a matrix based on multivariate input
     * @param x the variables to evaluate as part of the function
     * @return the matrix output of the function
     */
    public Matrix f(Vec x);
    
    /**
     * Computes a matrix based on multivariate input
     * @param x the variables to evaluate as part of the function
     * @param s the matrix to store the result in, or {@code null} if a new 
     * matrix should be allocated
     * @return the matrix containing the results. This is the same object as 
     * {@code s} if {@code s} is not {@code null}
     */
    public Matrix f(Vec x, Matrix s);
    
    /**
     * Computes a matrix based on multivariate input
     * @param x the variables to evaluate as part of the function
     * @param s the matrix to store the result in, or {@code null} if a new 
     * matrix should be allocated
     * @param ex the source of threads to use for the computation
     * @return the matrix containing the results. This is the same object as 
     * {@code s} if {@code s} is not {@code null}
     */
    public Matrix f(Vec x, Matrix s, ExecutorService ex);
}
