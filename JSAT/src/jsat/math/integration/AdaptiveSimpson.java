
package jsat.math.integration;

import jsat.math.Function1D;

/**
 * This class provides an implementation of the Adaptive Simpson method for
 * numerically computing an integral
 *
 * @author Edward Raff
 */
public class AdaptiveSimpson
{
    /**
     * Numerically computes the integral of the given function
     *
     * @param f the function to integrate
     * @param tol the precision for the desired result
     * @param a the lower limit of the integral
     * @param b the upper limit of the integral
     * @return an approximation of the integral of
     * &int;<sub>a</sub><sup>b</sup>f(x) , dx
     */
    static public double integrate(Function1D f, double tol, double a, double b)
    {
        return integrate(f, tol, a, b, 100);
    }
    
    /**
     * Numerically computes the integral of the given function
     *
     * @param f the function to integrate
     * @param tol the precision for the desired result
     * @param a the lower limit of the integral
     * @param b the upper limit of the integral
     * @param maxDepth the maximum recursion depth
     * @return an approximation of the integral of
     * &int;<sub>a</sub><sup>b</sup>f(x) , dx
     */
    static public double integrate(Function1D f, double tol, double a, double b, int maxDepth)
    {
        if(a == b)
            return 0;
        else if(a > b)
            throw new RuntimeException("Integral upper limit (" + b+") must be larger than the lower-limit (" + a + ")");

        double h = b-a;
        double c = (a+b)/2;
        
        double f_a = f.f(a);
        double f_b = f.f(b);
        double f_c = f.f(c);

        double one_simpson = h * (f_a + 4 * f_c + f_b) / 6;
        double d = (a + c) / 2;
        double e = (c + b) / 2;

        double two_simpson = h * (f_a + 4 * f.f(d) + 2 * f_c + 4 * f.f(e) + f_b) / 12;
        
        if(maxDepth <= 0)
            return two_simpson;
        
        if(Math.abs(one_simpson-two_simpson) < 15*tol)
            return two_simpson + (two_simpson - one_simpson)/15;
        else
        {
            double left_simpson  = integrate(f, tol/2, a, c, maxDepth-1);
            double right_simpson = integrate(f, tol/2, c, b, maxDepth-1);
            return left_simpson + right_simpson;
        }
    }
}
