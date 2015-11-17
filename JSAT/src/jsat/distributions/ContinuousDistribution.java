
package jsat.distributions;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionBase;
import jsat.math.integration.Romberg;
import jsat.math.optimization.oned.GoldenSearch;
import jsat.math.rootfinding.Zeroin;

/**
 * The ContinuousDistribution represents the contract for a continuous in one
 * dimension.<br>
 * <br>
 * Many of the functions of a Continuous Distribution are implemented by default
 * using numerical calculation and integration. For this reason, the base
 * implementations may be slower or less accurate than desired - and could
 * produce incorrect results for poorly behaved functions or large magnitude
 * inputs. These base implementations are provided for easy completeness, but
 * may not be appropriate for all methods. If needed, the implementer should
 * check if these methods provide the needed level of accuracy and speed.
 *
 * @author Edward Raff
 */
public abstract class ContinuousDistribution extends Distribution
{

    private static final long serialVersionUID = -5079392926462355615L;

    /**
     * Computes the log of the Probability Density Function. Note, that then the
     * probability is zero, {@link Double#NEGATIVE_INFINITY} would be the true
     * value. Instead, this method will always return the negative of
     * {@link Double#MAX_VALUE}. This is to avoid propagating bad values through
     * computation.
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

    @Override
    public double cdf(double x)
    {
        double intMin = getIntegrationMin();
        
        return Romberg.romb(getFunctionPDF(this), intMin, x);
    }
    
    @Override
    public double invCdf(final double p)
    {
        if (p < 0 || p > 1)
            throw new ArithmeticException("Value of p must be in the range [0,1], not " + p);
        double a = getIntegrationMin();
        double b = getIntegrationMax();

        Function newCDF = new Function()
        {

            @Override
            public double f(double... x)
            {
                return cdf(x[0]) - p;
            }

            @Override
            public double f(Vec x)
            {
                return f(x.get(0));
            }
        };
        return Zeroin.root(a, b, newCDF, p);
    }

    @Override
    public double mean()
    {
        double intMin = getIntegrationMin();
        
        double intMax = getIntegrationMax();

        return Romberg.romb(new FunctionBase()
        {

            @Override
            public double f(Vec x)
            {
                return x.get(0)*pdf(x.get(0));
            }
        }, intMin, intMax);
    }

    @Override
    public double variance()
    {
        double intMin = getIntegrationMin();
        
        double intMax = getIntegrationMax();
        final double mean = mean();

        return Romberg.romb(new FunctionBase()
        {

            @Override
            public double f(Vec x)
            {
                
                return Math.pow(x.get(0)-mean, 2)*pdf(x.get(0));
            }
        }, intMin, intMax);
    }
    
    

    @Override
    public double skewness()
    {
        double intMin = getIntegrationMin();
        
        double intMax = getIntegrationMax();
        final double mean = mean();

        return Romberg.romb(new FunctionBase()
        {

            @Override
            public double f(Vec x)
            {
                
                return Math.pow((x.get(0)-mean), 3)*pdf(x.get(0));
            }
        }, intMin, intMax)/Math.pow(variance(), 3.0/2);
    }
    
    @Override
    public double mode()
    {
        double intMin = getIntegrationMin();
        
        double intMax = getIntegrationMax();

        return GoldenSearch.findMin(intMin, intMax, new FunctionBase()
        {

            @Override
            public double f(Vec x)
            {
                return -pdf(x.get(0));
            }
        }, 1e-6, 1000);
    }
    
    protected double getIntegrationMin()
    {
        double intMin = min();
        if(Double.isInfinite(intMin))
        {
            intMin = -Double.MAX_VALUE/4;
            //Lets find a suitbly small PDF starting value for this 
            //first, lets take big steps
            for(int i = 0; i < 8; i++)
            {
                double sqrt = Math.sqrt(-intMin);
                if(pdf(sqrt) < 1e-5)
                    intMin = -sqrt;
                else
                    break;//no more big steps
            }
            
            //keep going until it looks like we should switch signs
            while(pdf(intMin) < 1e-5 && intMin < -0.1)
            {
                intMin/=2;
            }
            
            if(pdf(intMin) < 1e-5)//still?
                intMin *=-1;
            //ok, search positive... keep multiplying till we get there
            while(pdf(intMin) < 1e-5)
            {
                intMin *= 2;
            }
        }
        return intMin;
    }
    
    protected double getIntegrationMax()
    {
        double intMax = max();
        if(Double.isInfinite(intMax))
        {
            intMax = Double.MAX_VALUE / 4;
            //Lets find a suitbly small PDF starting value for this 
            //first, lets take big steps
            for (int i = 0; i < 8; i++)
            {
                double sqrt = Math.sqrt(intMax);
                if (pdf(sqrt) < 1e-5)
                    intMax = sqrt;
                else
                    break;//no more big steps
            }

            //keep going until it looks like we should switch signs
            while (pdf(intMax) < 1e-5 && intMax > 0.1)
            {
                intMax /= 2;
            }

            if (pdf(intMax) < 1e-5)//still?
                intMax *= -1;
            //ok, search negative... keep multiplying till we get there
            while (pdf(intMax) < 1e-5)
            {
                intMax *= 2;
            }
        }
        return intMax;
    }
    
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
    abstract public ContinuousDistribution clone();

    /**
     * Attempts to set the variables used by this distribution based on population sample data, 
     * assuming the sample data is from this type of distribution.
     * 
     * @param data the data to use to attempt to fit against
     */
    abstract public void setUsingData(Vec data);

    @Override
    public String toString()
    {
        return getDistributionName();
    }

    /**
     * Wraps the {@link #pdf(double) } function of the given distribution in a 
     * function object for use. 
     * 
     * @param dist the distribution to wrap the pdf of
     * @return a function for evaluating the pdf of the given distribution
     */
    public static Function getFunctionPDF(final ContinuousDistribution dist)
    {
        return new Function()
        {

            private static final long serialVersionUID = -897452735980141746L;

            @Override
            public double f(double... x)
            {
                return f(DenseVector.toDenseVec(x));
            }

            @Override
            public double f(Vec x)
            {
                return dist.pdf(x.get(0));
            }
        };
    }

}
