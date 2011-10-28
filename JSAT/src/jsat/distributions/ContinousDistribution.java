
package jsat.distributions;

import java.util.Random;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.rootfinding.Zeroin;

/**
 *
 * @author Edward Raff
 */
public abstract class ContinousDistribution
{
    public double logPdf(double x)
    {
        return Math.log(pdf(x));
    }
    abstract public double pdf(double x);
    abstract public double cdf(double x);
    abstract public double invCdf(double p);
    
    
    /**
     * This method is provided as a quick helper function, as any CDF has a 1 to 1 mapping with
     * an inverse, CDF<sup>.-1</sup>. This does a search for that value, and should only be used 
     * if the quantile function will be used infrequently or no alternative is available. 
     * @param p the [0,1] probability value 
     * @param cdf a function that provides the CDF we want to emulate the inverse of 
     * @return the quantile function, CDF<sup>-1</sup>(p) = x
     */
    protected double invCdf(final double p, final Function cdf)
    {
        if(p < 0 || p > 1)
            throw new ArithmeticException("Value of p must be in the range [0,1], not " + p);
        double a = Double.isInfinite(min()) ? Double.MIN_VALUE : min();
        double b = Double.isInfinite(max()) ? Double.MAX_VALUE : max();
        
        Function newCDF = new Function() {

            public double f(double... x)
            {
                return cdf.f(x)-p;
            }
        };
        return Zeroin.root(a, b, cdf, p);
    }

    /**
     * The minimum value for which the {@link #pdf(double) } is meant to return a value. Note that {@link Double#NEGATIVE_INFINITY} is a valid return value.
     * @return the minimum value for which the {@link #pdf(double) } is meant to return a value.
     */
    abstract public double min();

    /**
     * The maximum value for which the {@link #pdf(double) } is meant to return a value. Note that {@link Double#POSITIVE_INFINITY} is a valid return value.
     * @return the maximum value for which the {@link #pdf(double) } is meant to return a value.
     */
    abstract public double max();

    public String getDescriptiveName()
    {
        StringBuilder sb = new StringBuilder(getDistributionName());
        sb.append("(");
        String[] vars = getVariables();
        double[] vals = getCurrentVariableValues();
        
        sb.append(vars[0]).append(" = ").append(vals[0]);
        
        for(int i  = 1; i < vars.length; i++)
            sb.append(", ").append(vars[i]).append(" = ").append(vals[i]);
        
        sb.append(")");
        
        return sb.toString();
    }

    abstract public String getDistributionName();

    public double[] generateData(Random rnd, int count)
    {
        double[] data = new double[count];
        for(int i =0; i < count; i++)
            data[i] = invCdf(rnd.nextDouble());

        return data;
    }

    /**
     *
     * @return a string of the variable names this distribution uses
     */
    abstract public String[] getVariables();

    /**
     * @return the current values of the parameters used by this distribution, in the same order as their names are returned by {@link #getVariables() }
     */
    abstract public double[] getCurrentVariableValues();

    abstract public void setVariable(String var, double value);
    
    abstract public ContinousDistribution copy();

    /**
     * Attempts to set the variables used by this distribution based on population sample data, assuming the sample data is from this type of distribution.
     * @param data the data to use to attempt to fit against
     */
    abstract public void setUsingData(Vec data);
    
    public double[] sample(int numSamples, Random rand)
    {
        double[] samples = new double[numSamples];
        for(int i = 0; i < samples.length; i++)
            samples[i] = invCdf(rand.nextDouble());
        
        return samples;
    }
    
    public DenseVector sampleVec(int numSamples, Random rand)
    {
        return DenseVector.toDenseVec(sample(numSamples, rand));
    }
    
    abstract public double mean();
    
    public double median()
    {
        //P( x < m) = P(x > m) = 0.5, the x is the median. This is asking for the CDF(x) = 0.5, x is the median. so Q(0.5) = x
        return invCdf(0.5);
    }
    
    abstract public double mode();
    /**
     * Computes the variance of the distribution. Not all distributions have a 
     * finite variance for all parameter values. {@link Double#NaN NaN} may be 
     * returned if the variance is not defined for the current values of the distribution. 
     * {@link Double#POSITIVE_INFINITY Infinity} is a possible value to be returned
     * by some distributions. 
     * 
     * @return the variance of the distribution. 
     */
    abstract public double variance();
    /**
     * Computes the skewness of the distribution. Not all distributions have a 
     * finite skewness for all parameter values. {@link Double#NaN NaN} may be 
     * returned if the skewness is not defined for the current values of the distribution.
     * 
     * @return the skewness of the distribution. 
     */
    abstract public double skewness();
    public double standardDeviation()
    {
        return Math.sqrt(variance());
    }
    

    @Override
    public String toString()
    {
        return getDistributionName();
    }


}
