package jsat.math.optimization;

import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionVec;

/**
 * This interface defines a contract for multivariate function minimization.<br>
 * <br>
 * Different optimization methods will use or require different amounts of 
 * information. Depending on the optimizer, the 1st and 2nd derivative may be 
 * {@code null}. 
 * 
 * @author Edward Raff
 */
public interface Optimizer2
{
    /**
     * Attempts to optimize the given function by finding the value of {@code w}
     * that will minimize the value returned by {@code f(w)}, using 
     * <i>w = x<sub>0</sub></i> as an initial starting point. 
     * 
     * @param tolerance the value that the gradient norm must be less than to 
     * consider converged
     * @param w the the location to store the final solution
     * @param x0 the initial guess for the solution. This value will not be 
     * changed, and intermediate matrices will be created as the same type. 
     * @param f the objective function to minimizer
     * @param fp the derivative of the objective function, may be {@code null} 
     * depending on the optimizer
     * @param fpp the Hessian of the objective function, may be {@code null} 
     * depending on the optimizer
     */
    public void optimize(double tolerance, Vec w, Vec x0, Function f, FunctionVec fp, FunctionVec fpp);
    
    /**
     * Attempts to optimize the given function by finding the value of {@code w}
     * that will minimize the value returned by {@code f(w)}, using 
     * <i>w = x<sub>0</sub></i> as an initial starting point. 
     * 
     * @param tolerance the value that the gradient norm must be less than to 
     * consider converged
     * @param w the the location to store the final solution
     * @param x0 the initial guess for the solution. This value will not be 
     * changed, and intermediate matrices will be created as the same type. 
     * @param f the objective function to minimizer
     * @param fp the derivative of the objective function, may be {@code null} 
     * depending on the optimizer
     * @param fpp the Hessian of the objective function, may be {@code null} 
     * depending on the optimizer
     * @param ex the source of threads for parallel computation, may be 
     * {@code null} to perform serial computation. 
     */
    public void optimize(double tolerance, Vec w, Vec x0, Function f, FunctionVec fp, FunctionVec fpp, ExecutorService ex);
    
    /**
     * Sets the maximum number of iterations allowed for the optimization method
     * @param iterations the maximum number of iterations to perform
     */
    public void setMaximumIterations(int iterations);
    
    /**
     * Returns the maximum number of iterations to perform
     * @return the maximum number of iterations to perform
     */
    public int getMaximumIterations();
    
    public Optimizer2 clone();
}
