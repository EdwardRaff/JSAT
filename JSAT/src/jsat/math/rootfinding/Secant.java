
package jsat.math.rootfinding;

import jsat.math.Function1D;

/**
 * This class provides an implementation of the Secant method of finding roots
 * of functions.
 *
 * @author Edward Raff
 */
public class Secant implements RootFinder
{  

    private static final long serialVersionUID = -5175113107084930582L;

    
    /**
     * Performs root finding on the function {@code f}.
     *
     * @param a the left bound on the root (i.e., f(a) &lt; 0)
     * @param b the right bound on the root (i.e., f(b) &gt; 0)
     * @param f the function to find the root of
     * @return the value of variable {@code pos} that produces a zero value
     * output
     */
    public static double root(double a, double b, Function1D f)
    {
        return root(1e-15, a, b, f);
    }

    /**
     * Performs root finding on the function {@code f}.
     *
     * @param eps the desired accuracy of the result
     * @param a the left bound on the root (i.e., f(a) &lt; 0)
     * @param b the right bound on the root (i.e., f(b) &gt; 0)
     * @param f the function to find the root of
     * @return the value of variable {@code pos} that produces a zero value
     * output
     */
    public static double root(double eps, double a, double b, Function1D f)
    {
        return root(eps, 1000, a, b, f);
    }
    
    /**
     * Performs root finding on the function {@code f}.
     *
     * @param eps the desired accuracy of the result
     * @param maxIterations the maximum number of iterations to perform
     * @param a the left bound on the root (i.e., f(a) &lt; 0)
     * @param b the right bound on the root (i.e., f(b) &gt; 0)
     * @param f the function to find the root of
     * @return the value of variable {@code pos} that produces a zero value
     * output
     */
    public static double root(double eps, int maxIterations, double a, double b, Function1D f)
    {
        double x0 = a;
        double x1 = b;
        /**
         * f(x0)
         */
        double fx0 = f.f(x0);
        
        while(Math.abs(x1-x0) > 2*eps && maxIterations-- > 0)
        {
            double fx1 = f.f(x1);
            
            double nextX = x1 - fx1*(x1-x0)/(fx1-fx0);
            
            x0 = x1;
            fx0 = fx1;
            x1 = nextX;
        }
        
        return x1;
    }

    @Override
    public double root(double eps, int maxIterations, double[] initialGuesses, Function1D f)
    {
        return root(eps, maxIterations, initialGuesses[0], initialGuesses[1], f);
    }

    @Override
    public int guessesNeeded()
    {
        return 2;
    }
}
