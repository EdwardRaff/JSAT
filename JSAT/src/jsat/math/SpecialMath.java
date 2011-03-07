



package jsat.math;

import jsat.distributions.NormalDistribution;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public class SpecialMath
{

    public static double gamma(double z)
    {
        if(z == 0)//It is actualy infinity*I, where I - sqrt(-1).
            return Double.NaN;
        else if(z < 0)
        {
            /*
             * Using the identity
             *
             *                      __
             *                     -||
             * gamma(-z) = -------------------
             *                            __
             *             z gamma(z) sin || z
             */

            z = -z;
            return -PI/(z*gamma(z)*sin(PI*z));
        }

        /**
         * General case for z > 0, from Numerical Recipes in C (2nd ed. Cambridge University Press, 1992). |error| is <= 2*10^-10 for all z > 0
         *
         *               ____ /       6        \
         *              /  __ |     =====  p   |
         *            \/ 2 || |     \       n  |          z + 0.5  - (z + 5.5)
         * gamma(z) = ------- |p  +  >    -----| (z + 5.5)        e
         *               z    | 0   /     z + n|
         *                    |     =====      |
         *                    \     n = 1      /
         *
         *
         * see http://www.rskey.org/gamma.htm
         * 
         */

        double p[] =
        {
            1.000000000190015,76.18009172947146,-86.50532032941677,
            24.01409824083091,-1.231739572450155,1.208650973866179e-3,-5.395239384953e-6
        };

        double innerLoop = 0;
        for(int n = 1; n < p.length; n++)
            innerLoop += p[n]/(z+n);

        double result = p[0] + innerLoop;

        result *= sqrt(2*PI)/z;

        return result*pow(z+5.5, z+0.5)*exp(-(z+5.5));
    }

    public static double lnGamma(double z)
    {
        if(z <= 0)
            throw new ArithmeticException("Log gamma takes only positive arguments");

        /*
         * Lanczos approximation for the log of the gamma function, with |error| < 10^-15 for all z > 0 (Almost full double precision)
         */

        int j;
        double x, tmp, y, ser = 0.999999999999997092;
        double[] c = new double[]
        {
            57.1562356658629235, -59.5979603554754912,
            14.1360979747417471, -0.491913816097620199, .339946499848118887e-4,
            .465236289270485756e-4, -.983744753048795646e-4, .158088703224912494e-3,
            -.210264441724104883e-3, .217439618115212643e-3, -.164318106536763890e-3,
            .844182239838527433e-4, -.261908384015814087e-4, .368991826595316234e-5
        };

        y = x = z;
        tmp = x+671.0/128.0;
        tmp = (x+0.5)*log(tmp)-tmp;
        for (j = 0; j < 14; j++)
        {
            y++;
            ser += c[j] / y;
        }

        return tmp+log(2.5066282746310005*ser/x);
    }

    public static double erf(double x)
    {
       /*
        * erf(x) = 2 * cdf(x sqrt(2)) -1
        *
        * where cdf is the cdf of the normal distribution
        */

        return 2 * NormalDistribution.cdf(x * sqrt(2.0), 0, 1)-1; 
    }

    public static double invErf(double x)
    {
        /*
        * inverf(x) = invcdf(x/2+1/2)/sqrt(2)
        *
        * where invcdf is the inverse cdf of the normal distribution
        */

        return NormalDistribution.invcdf(x/2+0.5, 0, 1)/sqrt(2.0);
    }

    public static double erfc(double x)
    {
        /*
        * erf(x) = 2 * cdf(-x sqrt(2))
        *
        * where cdf is the cdf of the normal distribution
        */
        return 2 * NormalDistribution.cdf(-x * sqrt(2.0), 0, 1);
    }

    public static double invErfc(double x)
    {
       /*
        * inverf(x) = invcdf(x/2)/-sqrt(2)
        *
        * where invcdf is the inverse cdf of the normal distribution
        */

        return NormalDistribution.invcdf(x/2, 0, 1)/-sqrt(2.0);
    }
}
