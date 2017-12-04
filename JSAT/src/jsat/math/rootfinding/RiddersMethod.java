
package jsat.math.rootfinding;

import static java.lang.Math.*;
import jsat.math.Function1D;

/**
 * Provides an implementation of Ridder's method for root finding. 
 * @author Edward Raff
 */
public class RiddersMethod implements RootFinder
{

    private static final long serialVersionUID = 8154909945080099018L;

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
        double x1 = a;
        double x2 = b;
        double fx1 = f.f(x1);
        double fx2 = f.f(x2);
        double halfEps = eps*0.5;
        
        if(fx1* fx2 >= 0)
            throw new ArithmeticException("The given interval does not appear to bracket the root");
        
        double dif = 1;//Measure the change interface values
        while( abs(x1-x2) > eps && maxIterations-->0)
        {
            double x3 = (x1+x2)*0.5;
            
            double fx3 = f.f(x3);
            
            double x4 = x3+(x3-x1)*signum(fx1-fx2)*fx3/sqrt(fx3*fx3-fx1*fx2); 
         
            double fx4 = f.f(x4);
            if(fx3 * fx4 < 0)
            {
                x1 = x3;
                fx1 = fx3;
                x2 = x4;
                fx2 = fx4;
            }
            else if(fx1 * fx4 < 0)
            {
                dif = abs(x4 - x2);
                if(dif <= halfEps)//WE are no longer updating, return the value
                    return x4;
                x2 = x4;
                fx2 = fx4;
            }
            else
            {
                dif = abs(x4 - x1);
                if(dif <= halfEps)//WE are no longer updating, return the value
                    return x4;
                x1 = x4;
                fx1 = fx4;
            }
            
        }
        
        
        return x2;
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
