
package jsat.distributions;

import jsat.linear.Vec;
import jsat.text.GreekLetters;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public final class Laplace extends ContinousDistribution
{
    /**
     * location
     */
    private double mu;
    /*
     * Scale
     */
    private double b;

    public Laplace(double mu, double b)
    {
        setB(b);
        setMu(mu);
    }
    
    

    public void setMu(double mu)
    {
        this.mu = mu;
    }

    public double getMu()
    {
        return mu;
    }

    public void setB(double b)
    {
        if (b <= 0)
            throw new ArithmeticException("The scale parameter must be > 0");
        this.b = b;
    }

    public double getB()
    {
        return b;
    }
    
    @Override
    public double pdf(double x)
    {
        return 1/(2*b)*exp(-abs(x-mu)/b);
    }

    @Override
    public double cdf(double x)
    {
        double xMu = x - mu;
        return 0.5 * (1 + signum(x)*(1-exp(-abs(xMu)/b)) );
    }

    @Override
    public double invCdf(double p)
    {
        return mu - b * signum(p - 0.5) * log(1-2*abs(p-0.5));
    }

    @Override
    public double min()
    {
        return Double.NEGATIVE_INFINITY;
    }

    @Override
    public double max()
    {
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public String getDistributionName()
    {
        return "Laplace";
    }

    @Override
    public String[] getVariables()
    {
        return new String[] {GreekLetters.mu, "b"};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[] {mu, b};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if(var.equals(GreekLetters.mu))
            setMu(value);
        else if(var.equals("b"))
            setB(value);
    }

    @Override
    public ContinousDistribution clone()
    {
        return new Laplace(mu, b);
    }

    @Override
    public void setUsingData(Vec data)
    {
        //Donst sent mu yet incase b turns out to be a bad value
        double tmpMu = data.mean();
        
        double newB = 0;
        //TODO add APIs so that sparce vector can do this more efficiently
        for(int i = 0; i < data.length(); i++)
            newB += abs(data.get(i) - tmpMu);
        newB /= data.length();
        
        setB(newB);
        setMu(tmpMu);
        
    }

    @Override
    public double mean()
    {
        return mu;
    }

    @Override
    public double median()
    {
        return mu;
    }

    @Override
    public double mode()
    {
        return mu;
    }

    @Override
    public double variance()
    {
        return 2 * b*b;
    }

    @Override
    public double skewness()
    {
        return 0;
    }
    
}
