
package jsat.math.rootfinding;

import jsat.linear.Vec;
import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public class Bisection implements RootFinder
{
    

	private static final long serialVersionUID = -8107160048637997385L;

	/**
     * Uses the bisection method to find the argument of some function <tt>f</tt> for which 
     * <tt>f</tt>(<tt>args</tt>) = 0. If no <tt>args</tt> are given, it will be assumed that
     * <tt>f</tt> takes a single variable input 
     * 
     * @param a the lower end of the search interval. <tt>f</tt>(<tt>a</tt>) must be &le; 0
     * @param b the uper end of the search interval. <tt>f</tt>(<tt>b</tt>) must be &ge; 0
     * @param f the function to find a root of
     * @param args the list of initial values for the function. The value at 0
     * can be anything, since it will be over written by the search. 
     * 
     * @return a value that when given to <tt>f</tt> with additional values <tt>args</tt>, will return <tt>x</tt>
     */
    public static double root(double a, double b, Function f, double... args)
    {
        return root(1e-15, 1000, a, b, 0, f, args);
    }
    
    public static double root(double eps, double a, double b, Function f, double... args)
    {
        return root(eps, 1000, a, b, 0, f, args);
    }
    
    public static double root(double eps, double a, double b, int pos, Function f, double... args)
    {
        return root(eps, 1000, a, b, pos, f, args);
    }
    
    /**
     * Uses the bisection method to find the argument of some function <tt>f</tt> for which 
     * <tt>f</tt>(<tt>args</tt>) = 0. If no <tt>args</tt> are given, it will be assumed that
     * <tt>f</tt> takes a single variable input 
     * 
     * @param eps the accuracy desired 
     * @param maxIterations the maximum number of iterations 
     * @param a the lower end of the search interval. <tt>f</tt>(<tt>a</tt>) must be &le; 0
     * @param b the uper end of the search interval. <tt>f</tt>(<tt>b</tt>) must be &ge; 0
     * @param f the function to find a root of
     * @param pos which variable in the arguments is going to be the search variable
     * @param args the list of initial values for the function. The value at <tt>pos</tt> 
     * can be anything, since it will be over written by the search. 
     * 
     * @return the value of the <tt>pos</tt><sup>th</sup> variable that makes this function return 0. 
     */
    public static double root(double eps, int maxIterations, double a, double b, int pos, Function f, double... args)
    {
        if(b <= a)
            throw new ArithmeticException("a musbt be < b for Bisection to work");
        
        //We assume 1 dimensional function then 
        if(args == null ||args.length == 0)
        {
            pos = 0;
            args = new double[1];
        }
        
        args[pos] = b;
        double fb = f.f(args);
        args[pos] = a;
        double fa = f.f(args);
        
        if(fa* fb >= 0)
            throw new ArithmeticException("The given interval does not appear to bracket the root");
   
        while(b - a > 2*eps && maxIterations-- > 0)
        {
            args[pos] = (a+b)*0.5;
            double ftmp = f.f(args);
            
            if(fa*ftmp < 0)
            {
                b = args[pos];
                fb = ftmp;
            }
            else if(fb * ftmp < 0)
            {
                a = args[pos];
                fa = ftmp;
            }
            else
                break;//We converged
        }
        
        return (a+b)*0.5;
    }

    public double root(double eps, int maxIterations, double[] initialGuesses, Function f, int pos, double... args)
    {
        return root(eps, maxIterations, initialGuesses[0], initialGuesses[1], pos, f, args);
    }

    public double root(double eps, int maxIterations, double[] initialGuesses, Function f, int pos, Vec args)
    {
        return root(eps, maxIterations, initialGuesses[0], initialGuesses[1], pos, f, args.arrayCopy());
    }

    public int guessesNeeded()
    {
        return 2;
    }
}
