
package jsat.math.rootfinding;

import jsat.math.Function1D;

/**
 * Provides an implementation of the Bisection method of root finding. 
 * @author Edward Raff
 */
public class Bisection implements RootFinder
{
    
    private static final long serialVersionUID = -8107160048637997385L;

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
        if(b <= a)
            throw new ArithmeticException("a musbt be < b for Bisection to work");
        
        double fb = f.f(b);
        double fa = f.f(a);
        
        if(fa* fb >= 0)
            throw new ArithmeticException("The given interval does not appear to bracket the root");
   
        while(b - a > 2*eps && maxIterations-- > 0)
        {
            double midPoint = (a+b)*0.5;
            double ftmp = f.f(midPoint);
            
            if(fa*ftmp < 0)
            {
                b = midPoint;
                fb = ftmp;
            }
            else if(fb * ftmp < 0)
            {
                a = midPoint;
                fa = ftmp;
            }
            else
                break;//We converged
        }
        
        return (a+b)*0.5;
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
