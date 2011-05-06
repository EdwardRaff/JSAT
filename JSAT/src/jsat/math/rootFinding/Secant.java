
package jsat.math.rootFinding;

import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public class Secant
{
    /**
     * Uses the Secant method to find the argument of some function <tt>f</tt> for which the result is <tt>x</tt>. 
     * In other words, find the value <i>z</i> such that <tt>f</tt>(<i>z</i>) = <tt>x</tt>. <tt>x</tt> is the 
     * first value in <tt>args</tt>, and subsequent values in <tt>args</tt> will be constants passed to the 
     * function <tt>f</tt> in addition to the search parameter. 
     * <br><br>
     * The secant method will not always work. <tt>x0</tt> and <tt>x1</tt> must be sufficiently close to the true
     * solution. Further, the function's first and second derivative need to exist and be continuous for this 
     * method to work. <br>
     * The secant method converges at a sub linear rate, double the number of accurate digits in the solution by ~1.6 at each step. 
     * 
     * @param x0 a first initial estimate
     * @param x1 a second initial estimate
     * @param f
     * @param args
     * @return 
     */
    public static double root(double x0, double x1, Function f, double... args)
    {
        if(args.length < 1)
            throw new ArithmeticException("Bisection method requires a value to search for");

        /**
         * Shift the function down by the value we want to find
         */
        double s = args[0];
   
        args[0] = x0;
        /**
         * f(x0)
         */
        double fx0 = f.f(args)-s;
        
        while(Math.abs(x1-x0) > 2*1e-15)
        {
            args[0] = x1;
            
            double fx1 = f.f(args)-s;
            
            double nextX = x1 - fx1*(x1-x0)/(fx1-fx0);
            
            x0 = x1;
            fx0 = fx1;
            x1 = nextX;
            
            System.out.println("\t" + x1);
            
        }
        
        return x1;
    }
}
