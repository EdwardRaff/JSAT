
package jsat.distributions;

import jsat.linear.Vec;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;
/**
 *
 * @author Edward Raff
 */
public class Kolmogorov extends ContinousDistribution
{

    public Kolmogorov()
    {
    }
    

    @Override
    public double pdf(double x)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double cdf(double x)
    {
        if(x < 0)
            throw new ArithmeticException("Invalid value of x, x must be > 0, not " + x);
        if(x == 0)
            return 0;
        
        /* 
         * Uses 2 formulas, see http://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Kolmogorov_distribution
         * 
         * Each formula converges very rapidly, interface 3 terms to full 
         * IEEE precision for one or the other - crossover point is 1.18 
         * according to Numerical Recipies, 3rd Edition p(334-335)
         */
        double tmp = 0;
        if(x < 1.18)
        {
            
            for(int j = 1; j <= 3; j++ )
                tmp += exp( -pow(2*j-1,2)*PI*PI / (8*x*x) );
            
            return sqrt(2*PI)/x *tmp;
        }
        else
        {
            for(int j = 1; j <= 3; j++ )
                tmp += exp(-2*j*j*x*x)*pow(-1,j-1);
            
            return 1 - 2*tmp;
        }
        
    }

    @Override
    public double invCdf(double p)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double min()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double max()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public String getDistributionName()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public String[] getVariables()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void setVariable(String var, double value)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public ContinousDistribution copy()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void setUsingData(Vec data)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double mean()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double median()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double mode()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double variance()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
}
