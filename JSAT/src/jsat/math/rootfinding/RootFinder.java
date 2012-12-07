
package jsat.math.rootfinding;

import java.io.Serializable;
import jsat.linear.Vec;
import jsat.math.Function;

/**
 * This interface defines a general contract for the numerical computation of a root of a given function
 * @author Edward Raff
 */
public interface RootFinder extends Serializable
{
    /**
     * Attempts to numerical compute the root of a given function, such that f(<tt>args</tt>) = 0. Only one variable may be altered at a time
     * 
     * @param eps the accuracy desired for the solution
     * @param maxIterations the maximum number of steps allowed before forcing a return of the current solution. 
     * @param initialGuesses an array containing the initial guess values
     * @param f the function to find the root of
     * @param pos the index of the argument that will be allowed to alter in order to find the root. Starts from 0
     * @param args the values to be passed to the function as arguments
     * @return the value of the variable at the index <tt>pos</tt> that makes the function return 0
     */
    public double root(double eps, int maxIterations, double[] initialGuesses, Function f, int pos, double... args);
    
    /**
     * Attempts to numerical compute the root of a given function, such that f(<tt>args</tt>) = 0. Only one variable may be altered at a time
     * 
     * @param eps the accuracy desired for the solution
     * @param maxIterations the maximum number of steps allowed before forcing a return of the current solution. 
     * @param initialGuesses an array containing the initial guess values
     * @param f the function to find the root of
     * @param pos the index of the argument that will be allowed to alter in order to find the root. Starts from 0
     * @param args the values to be passed to the function as arguments
     * @return the value of the variable at the index <tt>pos</tt> that makes the function return 0
     */
    public double root(double eps, int maxIterations, double[] initialGuesses, Function f, int pos, Vec args);
    
    /**
     * Different root finding methods require different numbers of initial guesses. 
     * Some root finding methods require 2 guesses, each with values of opposite 
     * sign so that they bracket the root. Others just need any 2 initial guesses
     * sufficiently close to the root. This method simply returns the number of 
     * guesses that are needed. 
     * 
     * @return the number of initial guesses this root finding method needs
     */
    public int guessesNeeded();
}
