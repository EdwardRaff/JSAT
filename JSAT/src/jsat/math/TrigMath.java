
package jsat.math;
import static java.lang.Math.*;

/**
 * This class includes additional trig and hyperbolic trig that 
 * does not come with Java.Math by default.
 * 
 * @author Edward Raff
 */
public class TrigMath
{
    
    public static double coth(double x)
    {
        double eX = exp(x);
        double eNX = exp(-x);
        
        return (eX + eNX) / (eX - eNX);
    }
    
    public static double sech(double x)
    {
        return 2 / (exp(x) + exp(-x));
    }
    
    public static double csch(double x)
    {
        return 2 / (exp(x) - exp(-x));
    }
    
    public static double asinh(double x)
    {
        return log(x + sqrt(x*x + 1));
    }
    
    public static double acosh(double x)
    {
        if(x < 1)
            return Double.NaN;//Complex result
        return log(x + sqrt(x*x - 1));
    }
    
    public static double atanh(double x)
    {
        if(abs(x) >= 1)
            return Double.NaN;
        return 0.5* log((x+1) / (x-1));
    }
    
    public static double asech(double x)
    {
        if(x <= 0 || x > 1)
            return Double.NaN;
        return log((1 + sqrt(1-x*x))/x);
    }
    
    public static double acsch(double x)
    {
        return log(1/x + sqrt(1+x*x)/abs(x));
    }
    
    public static double acotch(double x)
    {
        if(abs(x) <= 1)
            return Double.NaN;
        return 0.5* log((x+1) / (x-1));
    }
            
}
