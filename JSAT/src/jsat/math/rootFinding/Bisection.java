
package jsat.math.rootFinding;

import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public class Bisection
{
    /**
     * Uses the bisection method to find the argument of some function <tt>f</tt> for which the result is <tt>x</tt>. 
     * In other words, find the value <i>z</i> such that <tt>f</tt>(<i>z</i>) = <tt>x</tt>. <tt>x</tt> is the 
     * first value in <tt>args</tt>, and subsequent values in <tt>args</tt> will be constants passed to the 
     * function <tt>f</tt> in addition to the search parameter. 
     * <br><br>
     * The Bisection method is fairly slow. Each step will require one evaluation of the function <tt>f</tt>. 
     * However, if there exist a root, and <tt>f</tt> is continuous on the range [a,b], it is guaranteed to be found. 
     * 
     * @param a the lower side of the search, such that <tt>f</tt>(<tt>a</tt>) <= <tt>x</tt>
     * @param b the upper side of the search, such that <tt>f</tt>(<tt>b</tt>) >= <tt>x</tt>
     * @param f the function to evaluate
     * @param args any additional arguments for <tt>f</tt> that are held constant. 
     * These values will be passed to f in order, with the search parameter being the first argument. 
     * @return a value that when given to <tt>f</tt> with additional values <tt>args</tt>, will return <tt>x</tt>
     */
    public static double root(double a, double b, Function f, double... args)
    {
        if(args.length < 1)
            throw new ArithmeticException("Bisection method requires a value to search for");
        if(b <= a)
            throw new ArithmeticException("a musbt be < b for Bisection to work");
        
        double x = args[0];
   
        while(b - a > 2*1e-15)
        {
            args[0] = (a+b)*0.5;
            
            if(f.f(args) > x)
                b = args[0];
            else
                a = args[0];
        }
        
        return (a+b)*0.5;
    }
}
