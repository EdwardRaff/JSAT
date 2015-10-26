
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
    
    public static double coth(final double x)
    {
        final double eX = exp(x);
        final double eNX = exp(-x);
        
        return (eX + eNX) / (eX - eNX);
    }
    
    public static double sech(final double x)
    {
        return 2 / (exp(x) + exp(-x));
    }
    
    public static double csch(final double x)
    {
        return 2 / (exp(x) - exp(-x));
    }
    
    public static double asinh(final double x)
    {
        return log(x + sqrt(x*x + 1));
    }
    
    public static double acosh(final double x)
    {
        if(x < 1) {
          return Double.NaN;//Complex result
        }
        return log(x + sqrt(x*x - 1));
    }
    
    public static double atanh(final double x)
    {
        if(abs(x) >= 1) {
          return Double.NaN;
        }
        return 0.5* log((x+1) / (x-1));
    }
    
    public static double asech(final double x)
    {
        if(x <= 0 || x > 1) {
          return Double.NaN;
        }
        return log((1 + sqrt(1-x*x))/x);
    }
    
    public static double acsch(final double x)
    {
        return log(1/x + sqrt(1+x*x)/abs(x));
    }
    
    public static double acotch(final double x)
    {
        if(abs(x) <= 1) {
          return Double.NaN;
        }
        return 0.5* log((x+1) / (x-1));
    }
            
}
