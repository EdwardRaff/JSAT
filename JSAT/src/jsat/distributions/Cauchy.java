
package jsat.distributions;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public final class Cauchy extends Distribution
{
    
    private double location;
    private double scale;

    public Cauchy(double x0, double y)
    {
        setScale(y);
        setLocation(x0);
    }

    public Cauchy()
    {
        this(0, 1);
    }

    public void setLocation(double x0)
    {
        this.location = x0;
    }

    public void setScale(double y)
    {
        if(y <= 0)
            throw new ArithmeticException("The scale parameter must be > 0, not " + y);
        this.scale = y;
    }

    public double getScale()
    {
        return scale;
    }

    public double getLocation()
    {
        return location;
    }
    
    @Override
    public double pdf(double x)
    {
        return 1.0 / ( Math.PI*scale*  (1 + Math.pow((x-location)/scale, 2))  );
    }

    @Override
    public double cdf(double x)
    {
        return Math.atan((x-location)/scale)/Math.PI + 0.5;
    }

    @Override
    public double invCdf(double p)
    {
        return location + scale * Math.tan(  Math.PI * (p - 0.5) );
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
        return "Cauchy";
    }

    @Override
    public String[] getVariables()
    {
        return new String[] {"x0", "y"};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[] {location, scale};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if(var.equals("y"))
            setScale(value);
        else if(var.equals("x0"))
            setLocation(value);
    }

    @Override
    public Distribution clone()
    {
        return new Cauchy(location, scale);
    }

    @Override
    public void setUsingData(Vec data)
    {
        data = data.sortedCopy();
        
        //approximate y by taking | 1st quant - 3rd quantile|
        int n = data.length();
        setScale(Math.abs(data.get(n/4) - data.get(3*n/4)));
        
        //approximate x by taking the median value
        //Note, technicaly, any value is equaly likely to be the true median of a chachy distribution, so we dont care about the exact median
        setLocation(data.get(n/2));
    }

    /**
     * The Cauchy distribution is unique in that it does not have a mean value (undefined). 
     * @return {@link Double#NaN} since there is no mean value
     */
    @Override
    public double mean()
    {
        return Double.NaN;
    }

    @Override
    public double median()
    {
        return location;
    }

    @Override
    public double mode()
    {
        return location;
    }

    /**
     * The Cauchy distribution is unique in that it does not have a variance value (undefined). 
     * @return {@link Double#NaN} since there is no variance value
     */
    @Override
    public double variance()
    {
        return Double.NaN;
    }

    /**
     * The Cauchy distribution is unique in that it does not have a standard deviation value (undefined). 
     * @return {@link Double#NaN} since there is no standard deviation value
     */
    @Override
    public double standardDeviation()
    {
        return Double.NaN;
    }
    
    @Override
    public double skewness()
    {
        return Double.NaN;
    }
    
}
