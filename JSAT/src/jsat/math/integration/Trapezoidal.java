
package jsat.math.integration;

import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public class Trapezoidal
{
    static public double trapz(Function f, double a, double b, int N)
    {
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
