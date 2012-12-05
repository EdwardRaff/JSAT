package jsat.math;

import jsat.linear.DenseVector;
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
}
