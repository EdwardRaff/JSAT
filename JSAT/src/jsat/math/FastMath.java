package jsat.math;

import static java.lang.Double.*;
import static java.lang.Math.PI;
import static java.lang.Math.tan;

/**
 * This class contains fast implementations of many of the methods located in 
 * {@link Math} and {@link SpecialMath}. This speed comes at the cost of 
 * correctness, and in general the methods in this class attempt to have a 
 * relative error no worse than 10<sup>-3</sup> for most inputs. <br>
 * <br>
 * Implementation details - and therfore accuracy / speed - may change over 
 * time. 10<sup>-3</sup> is not a hard guarantee, but a goal. The accuracy near 
 * asymptotes and extreme values will never be guaranteed. The handling of 
 * special values will never be guaranteed. 
 * 
 * @author Edward Raff
 */
public class FastMath
{
    /*
     * Exponent biast for doubles is 1023
     * so exponentValue-1023 = unbiased value
     */
    
    private static long getMantissa(long bits)
    {
        return bits & 0x000fffffffffffffL;
    }
    
    @SuppressWarnings("unused")
    private static long getExponent(long bits)
    {
        return (bits & 0x7ff0000000000000L) >> 52;
    }

    private static final double logConst = Math.log(2);
    
    /**
     * Computes the natural logarithm of the input
     * @param x the input
     * @return log<sub>e</sub>(x)
     */
    public static double log(double x)
    {
        return logConst*log2(x);
    }
    
    /**
     * Computes log<sub>2</sub>(x)
     * @param x the input
     * @return the log base 2 of {@code x}
     */
    public static double log2(double x)
    {
        return log2_2pd1(x);
    }
    
    /**
     * Computes log<sub>2</sub>(x) using a Pade approximation. It is slower than
     * {@link #log2_c11(double) } but dose not use any extra memory. <br>
     * The results are generally accurate to an absolute and relative error of 
     * 10<sup>-4</sup>, but relative error can get as high as 10<sup>-10</sup> 
     * @param x the input
     * @return the log base 2 of {@code x}
     */
    public static double log2_2pd1(double x)
    {
        if(x < 0)
            return Double.NaN;
        long rawBits = doubleToLongBits(x);
        long mantissa = getMantissa(rawBits);
        int e = Math.getExponent(x);
        double m = longBitsToDouble(1023L << 52 | mantissa);//m in [1, 2]

        double log2m = 1.847320661499000 + 0.240449173481494 * m - 3.651821822250191 / (0.750000000000000 + m);

        return log2m + e;
    }
    
    static final double[] log2Cache11 = new double[1 << 11];
    static
    {
        for(int i = 0; i < log2Cache11.length; i++)
        {
            long mantissa = i;
            mantissa <<= (52-11); 
            log2Cache11[i] = Math.log(longBitsToDouble(1023L<<52 | mantissa))/Math.log(2);
        }
    }
    
    /**
     * Computes log<sub>2</sub>(x) using a cache of 2<sup>11</sup> values, 
     * consuming approximately 17 kilobytes of memory.<br>
     * The results are generally accurate to an absolute and relative error of 
     * 10<sup>-4</sup>
     * @param x the input
     * @return the log base 2 of {@code x}
     */
    public static double log2_c11(double x)
    {
        if(x < 0)
            return Double.NaN;
        long rawBits = doubleToLongBits(x);
        long mantissa = getMantissa(rawBits);
        int e = Math.getExponent(x);
        
        return log2Cache11[(int)(mantissa >>> (52-11))] + e;
    }
    
    /**
     * Computes 2<sup>x</sup> exactly be exploiting the IEEE format
     * @param x the integer power to raise 2 too
     * @return 2<sup>x</sup>
     */
    public static double pow2(int x)
    {
        if(x > Double.MAX_EXPONENT)
            return Double.POSITIVE_INFINITY;
        if(x < Double.MIN_EXPONENT)
            return 0;
        return longBitsToDouble((x+1023L)<<52);
    }
    
    /**
     * Computes 2<sup>x</sup>.<br>
     * The results are generally accurate to an relative error of 
     * 10<sup>-4</sup>, but can be as accurate as 10<sup>-10</sup>
     * 
     * @param x the power to raise to
     * @return 
     */
    public static double pow2(double x)
    {
        if(x > Double.MAX_EXPONENT)
            return Double.POSITIVE_INFINITY;
        else if(x < Double.MIN_EXPONENT)
            return 0;
        else if(x < 0)
            return 1.0/pow2(-x);
        //x is positive at this point

        double floorXd = Math.floor(x);
        int floorX = (int) floorXd;
        double frac = x-floorXd;
        
        double pow2frac = -4.704682932438695+27.543765058113320/(4.828085122666891-frac)-0.490129071734273 * frac;
        
        return pow2frac*longBitsToDouble((floorX+1023L)<<52);
    }
    
    /**
     * Computes a<sup>b</sup>.<br>
     *
     *
     * @param a the base
     * @param b the power
     * @return a<sup>b</sup>
     */
    public static double pow(double a, double b)
    {

        /*
         * Wright out a^b as 2^(b log2(a)) and then replace a with 'm 2^e' to get
         * 2^(b * log2(m*2^e)) which simplifies to 
         * m^b 2^(b e) when m, e, and b are positive. 
         * 
         * m & e are by IEEE defintion positive 
         */

        if (b < 0)
            return 1 / pow(a, -b);//b is now made positive

        long rawBits_a = doubleToLongBits(a);
        long mantissa_a = getMantissa(rawBits_a);
        final int e_a = Math.getExponent(a);

        //compute m^b and exploit the fact that we know there is no need for the exponent
        double m = longBitsToDouble(1023L << 52 | mantissa_a);//m in [1, 2]
        
        final double log2m = 1.790711564253215 + 0.248597253161674 * m - 3.495545043418375 / (0.714309275671154 + 1.000000000000000 * m);

        //we end up with 2^(b * log_2(m)) * 2^(b * e), which we can reduce to a single pow2 call
        return pow2(b * log2m + b * e_a);//fun fact, double*int is faster than casting an int to a double...
    }
    
    private static final double expPowConst = 1.0/Math.log(2);
    
    /**
     * Exponentiates the given input value
     * @param x the input
     * @return e<sup>x</sup>
     */
    public static double exp(double x)
    {
        return pow2(expPowConst*x);
    }
    
    /**
     * Computes the digamma function of the input
     * @param x the input value
     * @return &psi;(x)
     */
    public static double digamma(double x)
    {
        if(x == 0)
            return Double.NaN;//complex infinity
        else if(x < 0)//digamma(1-x) == digamma(x)+pi/tan(pi*x), to make x positive
        {
            if(Math.rint(x) == x)
                return Double.NaN;//the zeros are complex infinity
            return digamma(1-x)-PI/tan(PI*x); 
        }
        
        /*
         * shift over 2 values to the left and use truncated approximation 
         * log(x+2)-1/(2 (x+2))-1/(12 (x+2)^2) -1/x-1/(x+1), 
         * the x+2 and x and x+1 are grouped sepratly 
         */
        double xp2 = x+2;
        
        return log(xp2)-(6*x+13)/(12*xp2*xp2)-(2*x+1)/(x*x+x);
    }
}
