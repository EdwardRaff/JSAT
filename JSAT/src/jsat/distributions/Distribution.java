
package jsat.distributions;

import java.util.Random;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.rootfinding.Zeroin;

/**
 * The Distribution represents the contract for a continuous in one dimension. 
 * 
 * @author Edward Raff
 */
public abstract class Distribution implements Cloneable
{
    /**
     * Computes the log of the Probability Density Function. Note, that then the probability 
     * is zero, {@link Double#NEGATIVE_INFINITY} would be the true value. Instead, this method
     * will always return the negative of {@link Double#MAX_VALUE}. This is to avoid propagating
     * bad values through computation. 
     * 
     * @param x the value to get the log(PDF) of
     * @return the value of log(PDF(x))
     */
    public double logPdf(double x)
    {
        double pdf = pdf(x);
        if(pdf <= 0)
            return -Double.MAX_VALUE;
        return Math.log(pdf);
    }
    
    /**
     * Computes the value of the Probability Density Function (PDF) at the given point
     * @param x the value to get the PDF 
     * @return the PDF(x)
     */
    abstract public double pdf(double x);
    
    /**
     * Computes the value of the Cumulative Density Function (CDF) at the given point. 
     * The CDF returns a value in the range [0, 1], indicating what portion of values 
     * occur at or below that point. 
     * 
     * @param x the value to get the CDF of
     * @return the CDF(x) 
     */
    abstract public double cdf(double x);
    
    /**
     * Computes the inverse Cumulative Density Function (CDF<sup>-1</sup>) at the given point. 
     * It takes in a value in the range of [0, 1] and returns the value x, such that CDF(x) = <tt>p</tt>
     * @param p the probability value
     * @return the value such that the CDF would return <tt>p</tt>
     */
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

            public double f(Vec x)
            {
                return f(x.get(0), x.get(1));
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

    /**
     * The descriptive name of a distribution returns the name of the distribution, followed by the parameters of the distribution and their values. 
     * @return the name of the distribution that includes parameter values
     */
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

    /**
     * Return the name of the distribution. 
     * @return the name of the distribution. 
     */
    abstract public String getDistributionName();

    /**
     * Returns an array, where each value contains the name of a parameter in the distribution. 
     * The order must always be the same, and match up with the values returned by {@link #getCurrentVariableValues() }
     * 
     * @return a string of the variable names this distribution uses
     */
    abstract public String[] getVariables();

    /**
     * Returns an array, where each value contains the value of a parameter in the distribution. 
     * The order must always be the same, and match up with the values returned by {@link #getVariables() }
     * @return the current values of the parameters used by this distribution, in the same order as their names are returned by {@link #getVariables() }
     */
    abstract public double[] getCurrentVariableValues();

    /**
     * Sets one of the variables of this distribution by the name. 
     * @param var the variable to set
     * @param value  the value to set 
     */
    abstract public void setVariable(String var, double value);
    
    @Override
    abstract public Distribution clone();

    /**
     * Attempts to set the variables used by this distribution based on population sample data, 
     * assuming the sample data is from this type of distribution.
     * 
     * @param data the data to use to attempt to fit against
     */
    abstract public void setUsingData(Vec data);
    
    /**
     * This method returns a double array containing the values of random samples from this distribution. 
     * 
     * @param numSamples the number of random samples to take
     * @param rand the source of randomness
     * @return an array of the random sample values
     */
    public double[] sample(int numSamples, Random rand)
    {
        double[] samples = new double[numSamples];
        for(int i = 0; i < samples.length; i++)
            samples[i] = invCdf(rand.nextDouble());
        
        return samples;
    }
    
    /**
     * This method returns a double array containing the values of random samples from this distribution. 
     * 
     * @param numSamples the number of random samples to take
     * @param rand the source of randomness
     * @return a vector of the random sample values
     */
    public DenseVector sampleVec(int numSamples, Random rand)
    {
        return DenseVector.toDenseVec(sample(numSamples, rand));
    }
    
    /**
     * Computes the mean value of the distribution 
     * @return the mean value of the distribution
     */
    abstract public double mean();
    
    /**
     * Computes the median value of the distribution 
     * @return the median value of the distribution 
     */
    public double median()
    {
        //P( x < m) = P(x > m) = 0.5, the x is the median. This is asking for the CDF(x) = 0.5, x is the median. so Q(0.5) = x
        return invCdf(0.5);
    }
    
    /**
     * Computes the mode of the distribution. Not all distributions have a mode for all parameter values. 
     * {@link Double#NaN NaN} may be returned if the mode is not defined for the current values of the 
     * distribution. 
     * 
     * @return the mode of the distribution
     */
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
    
    /**
     * Computes the standard deviation of the distribution. Not all distributions have a 
     * finite standard deviation for all parameter values. {@link Double#NaN NaN} may be 
     * returned if the variance is not defined for the current values of the distribution. 
     * {@link Double#POSITIVE_INFINITY Infinity} is a possible value to be returned
     * by some distributions. 
     * 
     * @return the standard deviation of the distribution 
     */
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
