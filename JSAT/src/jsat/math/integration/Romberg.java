
package jsat.math.integration;

import static java.lang.Math.*;
import jsat.math.Function1D;
/**
 * This class provides an implementation of the Romberg method for
 * numerically computing an integral
 *
 * @author Edward Raff
 */
public class Romberg
{

    /**
     * Numerically computes the integral of the given function
     *
     * @param f the function to integrate
     * @param a the lower limit of the integral
     * @param b the upper limit of the integral
     * @return an approximation of the integral of
     * &int;<sub>a</sub><sup>b</sup>f(x) , dx
     */
    public static double romb(Function1D f, double a, double b)
    {
        return romb(f, a, b, 20);
    }

    /**
     * Numerically computes the integral of the given function
     *
     * @param f the function to integrate
     * @param a the lower limit of the integral
     * @param b the upper limit of the integral
     * @param max the maximum number of extrapolation steps to perform. 
     * &int;<sub>a</sub><sup>b</sup>f(x) , dx
     */
    public static double romb(Function1D f, double a, double b, int max)
    {
        // see http://en.wikipedia.org/wiki/Romberg's_method

        max+=1;
        double[] s = new double[max];//first index will not be used
        double var = 0;//var is used to hold the value R(n-1,m-1), from the previous row so that 2 arrays are not needed
        double lastVal = Double.NEGATIVE_INFINITY;
        

        for(int k = 1; k < max; k++)
        {
            for(int i = 1; i <= k; i++)
            {
                if(i == 1)
                {
                    var = s[i];
                    s[i] = Trapezoidal.trapz(f, a, b, (int)pow(2, k-1));
                }
                else
                {
                    s[k]= ( pow(4 , i-1)*s[i-1]-var )/(pow(4, i-1) - 1);
                    var = s[i];
                    s[i]= s[k];
                }
            }

            if( abs(lastVal - s[k]) < 1e-15 )//there is only approximatly 15.955 accurate decimal digits in a double, this is as close as we will get
                return s[k];
            else lastVal = s[k];
        }

        return s[max-1];
    }
}
