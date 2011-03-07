
package jsat.distributions;

import java.util.Random;

/**
 *
 * @author Edward Raff
 */
public abstract class ContinousDistribution
{
    abstract public double pdf(double d);
    abstract public double invPdf(double d);
    abstract public double cdf(double d);
    abstract public double invCdf(double d);

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

    abstract public String getDescriptiveName();

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

    abstract public void setVariable(String var, double value);
    
    abstract public ContinousDistribution copy();

    @Override
    public String toString()
    {
        return getDistributionName();
    }


}
