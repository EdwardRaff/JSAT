



package jsat.math;

import jsat.distributions.NormalDistribution;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public class SpecialMath
{

    /**
     * The gamma function is a generalization of the factorial function. 
     * This method provides the gamma function for values from -Infinity to Infinity.
     * <br>Special Values: 
     * <ul>
     * <li>{@link Double#NaN} returned if z = 0 or z = {@link Double#NEGATIVE_INFINITY}</li>
     * </ul>
     * @param z any real value
     * @return Γ(z)
     */
    public static double gamma(double z)
    {
        if(z == 0)//It is actualy infinity*I, where I - sqrt(-1).
            return Double.NaN;
        else if(z == Double.POSITIVE_INFINITY)
            return z;
                
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

    /**
     * Computes the natural logarithm of {@link #gamma(double) }. 
     * This method is more numerically stable than taking the log of the result of Γ(z). 
     * 
     * <br>Special Values: 
     * <ul>
     * <li>{@link Double#POSITIVE_INFINITY} returned if z <= 0</li>
     * <li>{@link Double#NaN} returned if z = {@link Double#NEGATIVE_INFINITY}</li>
     * </ul>
     * 
     * @param z any real number value
     * @return Log(Γ(z))
     */
    public static double lnGamma(double z)
    {
        if(z == Double.NEGATIVE_INFINITY)
            return Double.NaN;
        else if(z == Double.POSITIVE_INFINITY)
            return z;
        else if(z <= 0)
            return Double.POSITIVE_INFINITY;

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

    /**
     * Computes the Beta function B(z,w)
     * @param z
     * @param w
     * @return B(z,w)
     */
    public static double beta(double z, double w)
    {

        /*
         * The beta function is defined by
         *
         *           Gamma(z) Gamma(w)
         * B(z, w) = -----------------
         *             Gamma(z + w)
         *
         * However, the definition is numericaly unstable (large value / large value to small result & small input).
         * Taking the log of each size and then exponentiating gives a more stable method of computing the result
         *
         *            lnGamma(z) + lnGamma(w) - lnGamma(z + w)
         * B(z, w) = e
         */
        return exp(lnGamma(z)+lnGamma(w)-lnGamma(z+w));
    }

    /**
     * Computes the regularized gamma function Q(a,z) = Γ(a,z)/Γ(a). <br> 
     * This method is more numerically stable and accurate than computing
     * it via the direct method, and is always in the range [0,1]. <br><br>
     * Note: The this method returns {@link Double#NaN} for a<0, though real values of Q(a,z) do exist
     * @param a any value >= 0
     * @param z any value > 0
     * @return Q(a,z)
     */
    public static double gammaQ(double a, double z)
    {
        if(z<= 0 || a < 0 )
            return Double.NaN;
        /**
         * On the range of x from 0.5 to 50
         * a=0.15, |rel error| is ~ 3e-15 for most values of x, with a bad spot of |rel error| ~ 3e-11 when x ~= 5.75
         * a=0.5, max |rel error| is 3.9e-15
         * a=1, max |rel error| is 4e-15, rel error groining as x -> infinity
         * a=5, max |rel error| is 7e-13, but only near x ~= 0, most is in the range 3e-15
         * a=10, max |rel error| is 4e-6, but only near x ~= 0, most is in the range 3e-15
         */
        return exp(a*log(z)-z-lnGamma(a))/gammaQ.lentz(a, z);
    }
    
    /**
     * Returns the regularized gamma function P(a,z) = γ(a,z)/Γ(a). <br>
     * This method is more numerically stable and accurate than computing
     * it via the direct method, and is always in the range [0,1]. <br><br>
     * Note: The this method returns {@link Double#NaN} for a<0, though real values of P(a,z) do exist
     * @param a any value >= 0
     * @param z any value > 0
     * @return P(a,z)
     */
    public static double gammaP(double a, double z)
    {
        if(z<= 0 || a < 0)
            return Double.NaN;
        /*
         * This method is currently usntable for values of z that grow larger, so it is not currently in use
         * return exp(a*log(z)-z-lnGamma(a))/gammaP.lentz(a,z);
         */
        return 1 - gammaQ(a, z);
    }
    
    /**
     * Computes the incomplete gamma function, Γ(a,z). 
     * <br>
     * Returns {@link Double#NaN} for z <= 0
     * @param a any value (-Infinity, Infinity)
     * @param z any value > 0
     * @return Γ(a,z)
     */
    public static double gammaIncUp(double a, double z)
    {
        if(z <= 0)
            return Double.NaN;
        /**
         * On the range of x from 0.5 to 50
         * a=0.15, max |rel error| is ~1.3e-11, but only in a small range x ~= 11. Otherwise rel error ~ 2e-15
         * a=0.5, max |rel error| is ~3.6e-15, less in all places otherwise
         * a=1, max |rel error| is ~4e-15, rel error grows as x-> infinity
         * a=5, max |rel error| is 7e-13, but only near x ~= 0, most is in the range 2e-15
         * a=10, max |rel error| is 4.9e-6, but only near x ~= 0, most is in the range 5e-15
         */
        return exp(a*log(z)-z)/upIncGamma.lentz(a,z);
    }
    
    /**
     * Computes the lower incomplete gamma function, γ(a,z). 
     * <br>
     * Returns {@link Double#NaN} for z <= 0
     * @param a any value (-Infinity, Infinity)
     * @param z any value > 0
     * @return γ(a,z)
     */
    public static double gammaIncLow(double a, double z)
    {
         if(z <= 0)
            return Double.NaN;
        /**
         * On the range of x from 0.5 to 50
         * a=0.15, see a=1
         * a=0.5, see a=1
         * a=1, max |rel error| is ~1.7e-13, rel error starts at 1e-15 and grows to the max as z increases
         * a=5, max |rel error| is 3e-13, but only near x ~= 0. for x > epsilon error starts at 5e-15 and grows with z up to 1e-13
         * a=10, max |rel error| is 2e-7, but only near x ~= 0, most is in the range 5e-15, grows to 5e-14 as z-> infinity
         */
        return exp(lnLowIncGamma(a, z));
    }
    
    private static double lnLowIncGamma(double a, double x)
    {

        /*
         * We compute the log of the lower incomplete gamma function by taking the log of
         *
         *                   oo
         *                  =====
         *            -x  a \         Gamma(a)      n
         * y(a, x) = e   x   >    ---------------- x
         *                  /     Gamma(a + 1 + n)
         *                  =====
         *                  n = 0
         *
         * Which becomes
         *
         *
         *                                   / oo                      \
         *                                   |=====                    |
         *                                   |\         Gamma(a)      n|
         * log(y(a, x)) = -x + log(x) a + log| >    ---------------- x |
         *                                   |/     Gamma(a + 1 + n)   |
         *                                   |=====                    |
         *                                   \n = 0                    /
         *
         * To reduce over flow of the gammas and exponentation in the summation, we compute the sum as
         *
         *  oo
         * =====
         * \      LogGamma(a) - LogGamma(a + 1 + n) + ln(x) n
         *  >    e
         * /
         * =====
         * n = 0
         *
         * Testin with x=0.5 to 100 (in increments of 0.5)
         * a=0.15, max relative error is ~ 6e-13, with x < 30 having a relative error smaller than 5e-14
         * a=0.5, maximum relative and absolute eror is ~1.5e-12 and ~1.5e-12 repsectivly, the error starts getting above e-15 when x = 25, and x=50 the error is up to 2.2e-13
         * a=10, maximum relative and absolute eror is ~4e-14 and ~4e-13 repsectivly, the error starts getting abovee-15 when x = 49. For x near zero (up to x ~ 2.5) the error is higher before droping, ~10^-14
         * a=25, maximum relative error is ~9.99e-15. From x~ 0 to 14, the error is worse, then droping to ~e-16 .
         * a=50, accuracy starting to degrad badly. From x~ 0 to 18 the error goes from 1.3 to 1e-7, the erro grows at an exponential rate as x -> 0. As x-> Infinity the error gets back down to ~5e-16
         */

        //Sumation first

        double lnGa = lnGamma(a);
        /**
         * This value will be updated by the property Gamma(z+1) = Gamma(z) * z, which - when taken the log of, is <br>
         * LnGamma(z+1) = LnGamma(z) + ln(z)
         */
        double lnGan = lnGa+log(a);
        double n = 0;
        /**
         * this is the n * ln(x)  term. it will be updated by adding the log of x at each step
         */
        double lnXN = 0;
        double lnX = log(x);

        //Set up, now start summing
        double term = exp(lnGa - lnGan + lnXN);
        double sum = term;
        while(term > 1e-15)
        {
            n++;
            lnXN += lnX;
            lnGan += log(a+n);

            term = exp(lnGa - lnGan + lnXN);

            sum += term;
        }

        //now the rest

        return -x + lnX*a +log(sum);
    }
    
    private static double lnLowIncGamma1(double a, double x)
    {
        double inter = lowIncGamma.lentz(a,x);
        if(inter <= 1e-16)//The result was ~0, in which case Gamma[a,z] ~= Gamma[a]
            return lnGamma(a);
        return a*log(x)-x-log(inter);
    }
    
    private static final ContinuedFraction lowIncGamma = new ContinuedFraction()
    {

        @Override
        public double getA(int pos, double... args)
        {

            if (pos % 2 == 0)
            {
                pos /= 2;//the # of the even term

                return pos * args[1];
            }
            else
            {
                pos = (pos + 0) / 2;

                return -(args[0] + pos) * args[1];
            }
        }

        @Override
        public double getB(int pos, double... args)
        {

            return args[0] + pos;
        }
    };

    /**
     * See http://functions.wolfram.com/GammaBetaErf/GammaRegularized/10/0003/
     */
    private static final ContinuedFraction gammaQ = new ContinuedFraction()
    {

        @Override
        public double getA(int pos, double... args)
        {
            return pos*(args[0]-pos);
        }

        @Override
        public double getB(int pos, double... args)
        {

            return (1 +pos*2) - args[0] + args[1];
        }
    };
    
    /**
     * Using the formula from http://functions.wolfram.com/GammaBetaErf/GammaRegularized/10/0009/
     * 
     * Note the formula is given in terms of gammaQ
     * 
     * {@link #gammaQ} is accurate for P and Q for the range of z > 0, this is used for z <= 0
     * 
     */
    private static final ContinuedFraction gammaP = new ContinuedFraction()
    {

        @Override
        public double getA(int pos, double... args)
        {
            if(pos % 2 == 0)//even step
            {
                pos/=2;
                return args[1]*pos;
            }
            //Else its an odd step
            pos= (pos+1)/2;
            return -args[1]*(args[0]+pos);
        }

        @Override
        public double getB(int pos, double... args)
        {
            return args[0] + pos;
        }
    };
    
    /**
     * continued fraction generated from mathematica
     * 
     * f(a,z) = e^-x*x^a / CF
     * 
     * a_k = (a-k)k
     * b_k = 1+k*2-a+x
     * 
     */
    private static final ContinuedFraction upIncGamma = new ContinuedFraction()
    {

        @Override
        public double getA(int pos, double... args)
        {
            return (args[0]-pos)*pos;
        }

        @Override
        public double getB(int pos, double... args)
        {
            return (1+pos*2)-args[0]+args[1];
        }
    };
}
