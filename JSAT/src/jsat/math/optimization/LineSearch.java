package jsat.math.optimization;

import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionVec;

/**
 * Line search defines a method of minimizing a function &phi;(&alpha;) = 
 * f(<b>x</b>+&alpha; <b>p</b>) where &alpha; &gt; 0 is a scalar value, and 
 * <b>x</b> and <b>p</b> are fixed vectors. <br>
 * <br>
 * Different line search methods may or may not use all the input variables.<br>
 * <br>
 * The LineSearch is allowed to maintain a history of update values to use on 
 * future calls. For this reason, a {@link #clone() clone} of the line search 
 * should be used for each new optimization problem. 
 * 
 * @author Edward Raff
 */
public interface LineSearch
{
    /**
     * Attempts to find the value of &alpha; that minimizes 
     * f(<b>x</b>+&alpha; <b>p</b>)
     * 
     * @param alpha_max the maximum value for &alpha; to search for 
     * @param x_k the initial value to search from
     * @param x_grad the gradient of &nabla; f(x<sub>k</sub>)
     * @param p_k the direction update
     * @param f the function to minimize the value of 
     * f(x<sub>k</sub> + &alpha; p<sub>k</sub>)
     * @param fp the gradient of f, &nabla;f(x), may be {@code null} depending 
     * upon the linesearch method
     * @param f_x the value of f(x<sub>k</sub>), or {@link Double#NaN} if it needs to be computed
     * @param gradP the value of &nabla;f(x<sub>k</sub>)<sup>T</sup>p<sub>k</sub>,
     * or {@link Double#NaN} if it needs to be computed
     * @param x_alpha_pk the location to store the value of 
     * x<sub>k</sub> + &alpha; p<sub>k</sub>
     * @param fxApRet an array to store the computed result of 
     * f(x<sub>k</sub> + &alpha; p<sub>k</sub>) in the first index
     * contain. May be {@code null} and the value will not be returned
     * @param grad_x_alpha_pk location to store the value of &nabla; f(x<sub>k</sub>&alpha;+p<sub>k</sub>). May be {@code null}, local storage will be allocated if needed
     * @return the value of &alpha; that satisfies the line search in minimizing f(x<sub>k</sub> + &alpha; p<sub>k</sub>)
     */
    public double lineSearch(double alpha_max, Vec x_k, Vec x_grad, Vec p_k, Function f, FunctionVec fp, double f_x, double gradP, Vec x_alpha_pk, double[] fxApRet, Vec grad_x_alpha_pk);
    
    /**
     * Attempts to find the value of &alpha; that minimizes 
     * f(<b>x</b>+&alpha; <b>p</b>)
     * 
     * @param alpha_max the maximum value for &alpha; to search for 
     * @param x_k the initial value to search from
     * @param x_grad the gradient of &nabla; f(x<sub>k</sub>)
     * @param p_k the direction update
     * @param f the function to minimize the value of 
     * f(x<sub>k</sub> + &alpha; p<sub>k</sub>)
     * @param fp the gradient of f, &nabla;f(x), may be {@code null} depending 
     * upon the linesearch method
     * @param f_x the value of f(x<sub>k</sub>), or {@link Double#NaN} if it needs to be computed
     * @param gradP the value of &nabla;f(x<sub>k</sub>)<sup>T</sup>p<sub>k</sub>,
     * or {@link Double#NaN} if it needs to be computed
     * @param x_alpha_pk the location to store the value of 
     * x<sub>k</sub> + &alpha; p<sub>k</sub>
     * @param fxApRet an array to store the computed result of 
     * f(x<sub>k</sub> + &alpha; p<sub>k</sub>) in the first index
     * contain. May be {@code null} and the value will not be returned
     * @param grad_x_alpha_pk location to store the value of &nabla; f(x<sub>k</sub>&alpha;+p<sub>k</sub>). May be {@code null}, local storage will be allocated if needed
     * @param ex the source of threads for parallel computation, or {@code null}
     * to perform serial computation
     * @return the value of &alpha; that satisfies the line search in minimizing f(x<sub>k</sub> + &alpha; p<sub>k</sub>)
     */
    public double lineSearch(double alpha_max, Vec x_k, Vec x_grad, Vec p_k, Function f, FunctionVec fp, double f_x, double gradP, Vec x_alpha_pk, double[] fxApRet, Vec grad_x_alpha_pk, ExecutorService ex);

    /**
     * When performing the {@link #lineSearch(double, jsat.linear.Vec, jsat.linear.Vec, jsat.linear.Vec, jsat.math.Function, jsat.math.FunctionVec, double, double, jsat.linear.Vec, double[], jsat.linear.Vec) linear search}
     * step some line searches may or may not use the gradient information. If 
     * the gradient information is used and updated, this method will return 
     * {@code true}. If not the given vector will be unused and not updated, and
     * this method will return {@code false}
     * @return {@code true} if the {@code grad_x_alpha_pk} parameter of 
     * lineSearch will be up-to-date after the call, or {@code false} if the 
     * gradient value will need to be computed after. 
     */
    public boolean updatesGrad();
    
    /**
     * Returns a clone of the line search object
     * @return a clone of the line search object
     */
    public LineSearch clone();
}
