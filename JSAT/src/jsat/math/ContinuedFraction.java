
package jsat.math;

/**
 * This class provides a means to represent and evaluate continued fractions in 
 * a multitude of ways. 
 * 
 * @author Edward Raff
 */
public abstract class ContinuedFraction
{
    /**
     * The a term of a continued fraction is the value that occurs as one of the
     * numerators, an its depth starts at 1. 
     * 
     * @param pos the depth of the continued fraction to evaluate at
     * @param args the values for the variables of the continued fraction
     * @return the value that would be computed for the a coefficient at the 
     * specified depth of the fraction
     */
    abstract public double getA(int pos, double... args);
    
    /**
     * The b term of a continued fraction is the value that is added to the 
     * continuing fraction, its depth starts at 0. 
     * 
     * @param pos the depth of the continued fraction to evaluate at
     * @param args the values for the variables of the continued fraction
     * @return the value that would be computed for the b coefficient at the 
     * specified depth of the fraction
     */
    abstract public double getB(int pos, double... args);
    
    /**
     * Approximates the continued fraction using a naive approximation
     * 
     * @param n the number of iterations to perform
     * @param args the values to input for the variables of the continued fraction
     * @return an approximation of the value of the continued fraciton
     */
    public double backwardNaive(int n, double... args)
    {
        double term = getA(n, args)/getB(n,args);
        
        for(n = n-1; n >0; n--)
        {
            term = getA(n, args)/(getB(n,args)+term);
        }
        
        return term + getB(0, args);
    }
    
    /**
     * Uses Thompson and Barnett's modified Lentz's algorithm create an 
     * approximation that should be accurate to full precision. 
     * 
     * @param args the numeric inputs to the continued fraction
     * @return the approximate value of the continued fraction
     */
    public double lentz(double... args)
    {
        double f_n = getB(0, args);
        if(f_n == 0.0)
            f_n = 1e-30;
        double c_n, c_0 = f_n;
        double d_n, d_0 = 0;
        
        double delta = 0;
        
        int j = 0;
        while(Math.abs(delta - 1) > 1e-15)
        {
            
            j++;
            d_n = getB(j, args) + getA(j, args)*d_0;
            if(d_n == 0.0)
                d_n = 1e-30;
            
            c_n = getB(j, args) + getA(j, args)/c_0;
            if(c_n == 0.0)
                c_n = 1e-30;
            
            d_n = 1/d_n;
            delta = c_n*d_n;
            f_n *= delta;
            
            
            d_0 = d_n;
            c_0 = c_n;
        }
        
        
        return f_n;
    }
    
}
