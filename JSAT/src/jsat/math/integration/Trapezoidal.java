
package jsat.math.integration;

import jsat.math.Function1D;

/**
 * This class provides an implementation of the Trapezoidal method for
 * numerically computing an integral
 *
 * @author Edward Raff
 */
public class Trapezoidal
{
    /**
     * Numerically computes the integral of the given function
     *
     * @param f the function to integrate
     * @param a the lower limit of the integral
     * @param b the upper limit of the integral
     * @param N the number of points in the integral to take, must be &ge; 2.
     * @return an approximation of the integral of
     * &int;<sub>a</sub><sup>b</sup>f(x) , dx
     */
    static public double trapz(Function1D f, double a, double b, int N)
    {
        if(a == b)
            return 0;
        else if(a > b)
            throw new RuntimeException("Integral upper limit (" + b+") must be larger than the lower-limit (" + a + ")");
        else if(N < 1)
            throw new RuntimeException("At least two integration parts must be used, not " + N);
        /*
         *    b               /              N - 1                 \
         *   /                |              =====                 |
         *  |           b - a |f(a) + f(b)   \      /    k (b - a)\|
         *  | f(x) dx = ----- |----------- +  >    f|a + ---------||
         *  |             N   |     2        /      \        N    /|
         * /                  |              =====                 |
         *  a                 \              k = 1                 /
         */
        double sum =0;
        for(int k = 1; k < N; k++)
            sum += f.f(a+k*(b-a)/N);

        sum+= (f.f(a)+f.f(b))/2;

        return (b-a)/N*sum;
    }
}
