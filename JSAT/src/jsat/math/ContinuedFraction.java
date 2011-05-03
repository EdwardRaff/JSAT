
package jsat.math;

/**
 * This class provides a means to represent and evaluate continued fractions in a multitude of ways. 
 * @author Edward Raff
 */
public abstract class ContinuedFraction
{
    abstract public double getA(int pos, double... args);
    abstract public double getB(int pos, double... args);
    
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
     * @param args
     * @return 
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
