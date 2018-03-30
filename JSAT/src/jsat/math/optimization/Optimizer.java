package jsat.math.optimization;

import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionVec;

/**
 * This interface defines a contract for multivariate function minimization.<br>
 * <br>
 * Different optimization methods will use or require different amounts of
 * information. Depending on the optimizer, the 1st derivative may not be
 * necessary and can be {@code null}.
 *
 * @author Edward Raff
 */
public interface Optimizer
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
     */
    default public void optimize(double tolerance, Vec w, Vec x0, Function f, FunctionVec fp)
    {
        optimize(tolerance, w, x0, f, fp, false);
    }
    
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
     * @param parallel {@code true} if multiple threads should be used for
     * optimization, or {@code false} if a single thread should be used.
     */
    public void optimize(double tolerance, Vec w, Vec x0, Function f, FunctionVec fp, boolean parallel);
    
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
    
    public Optimizer clone();
}
