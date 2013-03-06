package jsat.math;

import static java.lang.Math.exp;
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
            expSum += Math.exp(vals.get(i)-maxValue);
        
        return maxValue + expSum;
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
            expSum += Math.exp(vals[i]-maxValue);
        
        return maxValue + expSum;
    }
    
    /**
     * Applies the softmax function to the given array of values, normalizing 
     * them so that each value is equal to<br><br>
     * exp(x<sub>j</sub>) / &Sigma;<sub>&forall; i</sub> exp(x<sub>i</sub>)
     * 
     * @param x the array of values
     * @param implicitExtra {@code true} if the softmax will assume there is 
     * an extra implicit value not included in the array with a value of 1.0 
     */
    public static void softmax(double[] x, boolean implicitExtra)
    {
        double max = implicitExtra ? 1 : Double.NEGATIVE_INFINITY;
        for(int i = 0; i < x.length; i++)
            max = Math.max(max, x[i]);
        
        //exp is exp(0 - max), b/c exp(0) gives the implicit 1 value
        double z =implicitExtra ? exp(-max) : 0;
        for (int c = 0; c < x.length; c++)
            z += (x[c] = exp(x[c] - max));
        for (int c = 0; c < x.length; c++)
            x[c] /= z;
    }
    
    /**
     * Convenience object for taking the {@link Math#sqrt(double) square root} 
     * of the first index
     */
    public static final Function sqrtFunc = new FunctionBase() 
    {
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
    public static final Function sqrdFunc = new FunctionBase() {

        @Override
        public double f(Vec x)
        {
            double xx = x.get(0);
            return xx*xx;
        }
    };
    
    /**
     * Convenience object for taking the {@link Math#log(double) log } of the 
     * first index
     */
    public static final Function logFunc = new FunctionBase() 
    {
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
        @Override
        public double f(Vec x)
        {
            return Math.exp(x.get(0));
        }
    };
    
    
    
}
