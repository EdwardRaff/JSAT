
package jsat.distributions;

import java.util.Random;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public abstract class ContinousDistribution
{
    abstract public double pdf(double x);
    abstract public double invPdf(double p);
    abstract public double cdf(double x);
    abstract public double invCdf(double p);

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

    @Override
    public String toString()
    {
        return getDistributionName();
    }


}
