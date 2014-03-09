package jsat.math;

import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;

/**
 * FunctionP is the same as {@link Function} except it supports parallel 
 * computation of the result. 
 * 
 * @author Edward Raff
 */
public interface FunctionP extends Function
{
    /**
     * Computes the result of a single variate function 
     * @param x the multivariate input
     * @param ex the source of threads to compute the result from
     * @return the output of the function 
     */
    public double f(Vec x, ExecutorService ex);
}
