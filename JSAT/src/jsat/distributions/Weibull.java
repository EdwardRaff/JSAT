
package jsat.distributions;

import jsat.linear.Vec;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;
/**
 *
 * @author Edward Raff
 */
public class Weibull extends ContinousDistribution
{
    /**
     * Shape parameter
     */
    double k;
    /**
     * Scale parameter
     */
    double gam;

    public Weibull(double k, double gam)
    {
        if(k <= 0)
            throw new ArithmeticException("k must be > 0 not " + k );
        if(gam <= 0)
            throw new ArithmeticException("k must be > 0 not " + gam );
        this.k = k;
        this.gam = gam;
    }


    
    @Override
    public double pdf(double x)
    {
        if(x < 0)
            return 0;
        
        return k/gam * pow(x/gam, k-1)*exp(-pow(x/gam, k));
        
    }

    @Override
    public double cdf(double x)
    {
        return 1 - exp(-pow(x/gam, k));
    }

    @Override
    public double invCdf(double p)
    {
        return gam*pow(-log(1-p),1/k);
    }

    @Override
    public double min()
    {
        return 0;
    }

    @Override
    public double max()
    {
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public String getDistributionName()
    {
        return "Weibull";
    }

    @Override
    public String[] getVariables()
    {
        return new String[]{"k", "gam"};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[]{k, gam};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if (var.equals("k"))
            if (value > 0)
                k = value;
            else
                throw new ArithmeticException("k must be > 0 not " + value);
        else if (var.equals("gam"))
            if (value > 0)
                gam = value;
            else
                throw new ArithmeticException("gam must be > 0 not " + value );
    }

    @Override
    public ContinousDistribution copy()
    {
        return new Weibull(k, gam);
    }

    @Override
    public void setUsingData(Vec data)
    {
        //TODO how do you solve for this sucker? I dont know. 
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double mean()
    {
        return gam * gamma(1+1/k);
    }

    @Override
    public double median()
    {
        return pow(log(2), 1/k)*gam;
    }

    @Override
    public double mode()
    {
        if(k <= 1)
            throw new ArithmeticException("Mode only exists for k > 1");
        
        return gam * pow( (k-1)/k, 1/k);
    }

    @Override
    public double variance()
    {
        return gam*gam * gamma(1+2/k) - pow(median(),2);
    }
    
}
