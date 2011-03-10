
package jsat.math.integration;

import jsat.math.Function;
import static java.lang.Math.*;
/**
 *
 * @author Edward Raff
 */
public class Romberg
{

    public static double romb(Function f, double a, double b)
    {
        return romb(f, a, b, 20);
    }

    public static double romb(Function f, double a, double b, int max)
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
