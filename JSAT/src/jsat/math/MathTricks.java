package jsat.math;

import static java.lang.Math.*;
import jsat.linear.Vec;

/**
 * This class provides utilities for performing specific arithmetic patterns in 
 * numerically stable / efficient ways. 
 * 
 * @author Edward Raff
 */
public class MathTricks
{

    private MathTricks()
    {
    }
    
    /**
     * 
     * @param vals an array of values to get the maximum of 
     * @return the maximum of all given values
     */
    public static double max(double... vals)
    {
        double m = Double.NEGATIVE_INFINITY;
        for(double v : vals)
            m = Math.max(v, m);
        return m;
    }
    
    /**
     * 
     * @param vals an array of values to get the maximum of 
     * @return the maximum of all given values
     */
    public static double min(double... vals)
    {
        double m = Double.NEGATIVE_INFINITY;
        for(double v : vals)
            m = Math.min(v, m);
        return m;
    }
    
    
    /**
     * Provides a numerically table way to perform the log of a sum of 
     * exponentiations. The computed result is <br>
     * log(<big>&#8721;</big><sub> &forall; val &isin; vals</sub> exp(val) )
     *
     * @param vals the array of values to exponentiate and add
     * @param maxValue the maximum value in the array 
     * @return the log of the sum of the exponentiated values
     */
    public static double logSumExp(Vec vals, double maxValue)
    {
        double expSum = 0.0;
        for(int i = 0; i < vals.length(); i++)
            expSum += exp(vals.get(i)-maxValue);
        
        return maxValue + log(expSum);
    }
    
    /**
     * Provides a numerically table way to perform the log of a sum of 
     * exponentiations. The computed result is <br>
     * log(<big>&#8721;</big><sub> &forall; val &isin; vals</sub> exp(val) )
     * 
     * @param vals the array of values to exponentiate and add
     * @param maxValue the maximum value in the array 
     * @return the log of the sum of the exponentiated values
     */
    public static double logSumExp(double[] vals, double maxValue)
    {
        double expSum = 0.0;
        for(int i = 0; i < vals.length; i++)
            expSum += exp(vals[i]-maxValue);
        
        return maxValue + log(expSum);
    }
    
    /**
     * Applies the softmax function to the given array of values, normalizing 
     * them so that each value is equal to<br><br>
     * exp(x<sub>j</sub>) / &Sigma;<sub>&forall; i</sub> exp(x<sub>i</sub>)
     * 
     * @param x the array of values
     * @param implicitExtra {@code true} if the softmax will assume there is 
     * an extra implicit value not included in the array with a value of 0.0 
     */
    public static void softmax(double[] x, boolean implicitExtra)
    {
        double max = implicitExtra ? 1 : Double.NEGATIVE_INFINITY;
        for(int i = 0; i < x.length; i++)
            max = max(max, x[i]);
        
        double z =implicitExtra ? exp(-max) : 0;
        for (int c = 0; c < x.length; c++)
            z += (x[c] = exp(x[c] - max));
        for (int c = 0; c < x.length; c++)
            x[c] /= z;
    }
    
    /**
     * Applies the softmax function to the given array of values, normalizing 
     * them so that each value is equal to<br><br>
     * exp(x<sub>j</sub>) / &Sigma;<sub>&forall; i</sub> exp(x<sub>i</sub>)<br>
     * Note: If the input is sparse, this will end up destroying sparsity
     * 
     * @param x the array of values
     * @param implicitExtra {@code true} if the softmax will assume there is 
     * an extra implicit value not included in the array with a value of 0.0 
     */
    public static void softmax(Vec x, boolean implicitExtra)
    {
        double max = implicitExtra ? 1 : Double.NEGATIVE_INFINITY;
        max = max(max, x.max());
        
        double z =implicitExtra ? exp(-max) : 0;
        for (int c = 0; c < x.length(); c++)
        {
            double newVal = exp(x.get(c) - max);
            x.set(c, newVal);
            z += newVal;
        }
        x.mutableDivide(z);
    }
    
    /**
     * This evaluates a polynomial using Horner's method. It is assumed that the
     * polynomial is stored in reverse order in the array {@code coef}, ie: from
     * c<sub>n</sub> at index 0, and then decreasing.
     *
     * @param coef the polynomial with coefficients in reverse order
     * @param x the value to evaluate the polynomial at
     * @return the value of the polynomial at {@code x}
     */
    public static double hornerPolyR(double[] coef, double x)
    {
        double result = 0;
        for(double c : coef)
            result = result*x+c;
        return result;
    }
    
    /**
     * This evaluates a polynomial using Horner's method. It is assumed that the
     * polynomial is stored in order in the array {@code coef}, ie: from
     * c<sub>0</sub> at index 0, and then increasing with the index.
     *
     * @param coef the polynomial with coefficients in reverse order
     * @param x the value to evaluate the polynomial at
     * @return the value of the polynomial at {@code x}
     */
    public static double hornerPoly(double[] coef, double x)
    {
        double result = 0;
        for(int i = coef.length-1; i >= 0; i--)
            result = result*x + coef[i];
        return result;
    }
    
    /**
     * Convenience object for taking the {@link Math#sqrt(double) square root} 
     * of the first index
     */
    public static final Function sqrtFunc = new FunctionBase() 
    {
        /**
		 * 
		 */
		private static final long serialVersionUID = -5898515135319116600L;

		@Override
        public double f(Vec x)
        {
            return Math.sqrt(x.get(0));
        }
    };
    
    /**
     * Convenience object for taking the squared value
     * of the first index
     */
    public static final Function sqrdFunc = new FunctionBase() 
    {

        /**
		 * 
		 */
		private static final long serialVersionUID = 6831886040279358142L;

		@Override
        public double f(Vec x)
        {
            double xx = x.get(0);
            return xx*xx;
        }
    };
    
    /**
     * Convenience object for taking the inverse (x<sup>-1</sup>) of the first 
     * index. 
     */
    public static final Function invsFunc = new FunctionBase() 
    {
        /**
		 * 
		 */
		private static final long serialVersionUID = -7745316806635400174L;

		@Override
        public double f(Vec x)
        {
            return 1/x.get(0);
        }
    };
    
    /**
     * Convenience object for taking the {@link Math#log(double) log } of the 
     * first index
     */
    public static final Function logFunc = new FunctionBase() 
    {
        /**
		 * 
		 */
		private static final long serialVersionUID = -4653355640520837353L;

		@Override
        public double f(Vec x)
        {
            return Math.log(x.get(0));
        }
    };
    
    /**
     * Convenience object for taking the {@link Math#exp(double) exp } of the 
     * first index
     */
    public static final Function expFunc = new FunctionBase() 
    {
        /**
		 * 
		 */
		private static final long serialVersionUID = 7075309263321302492L;

		@Override
        public double f(Vec x)
        {
            return Math.exp(x.get(0));
        }
    };
    
    /**
     * Convenience object for taking the {@link Math#abs(double) abs } of the 
     * first index
     */
    public static final Function absFunc = new FunctionBase() 
    {
        /**
		 * 
		 */
		private static final long serialVersionUID = -3706702191562872641L;

		@Override
        public double f(Vec x)
        {
            return Math.abs(x.get(0));
        }
    };
    
}
