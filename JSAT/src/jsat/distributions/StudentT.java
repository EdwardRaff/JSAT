
package jsat.distributions;

import jsat.linear.Vec;

/**
 *347
 * @author Edward Raff
 */
public class StudentT extends ContinousDistribution
{
    double df, mu, sig;

    public StudentT(double df, double mu)
    {
        this(df, mu, 1);
    }

    public StudentT(double df, double mu, double sig)
    {
        this.df = df;
        this.mu = mu;
        this.sig = sig;
    }

    
    @Override
    public double pdf(double d)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double invPdf(double d)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double cdf(double d)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double invCdf(double d)
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
    public String getDescriptiveName()
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
    
}
