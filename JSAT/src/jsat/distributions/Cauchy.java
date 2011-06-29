
package jsat.distributions;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public final class Cauchy extends ContinousDistribution
{
    
    private double x0;
    private double y;

    public Cauchy(double x0, double y)
    {
        setY(y);
        setX0(x0);
    }

    public Cauchy()
    {
        this(0, 1);
    }

    public void setX0(double x0)
    {
        this.x0 = x0;
    }

    public void setY(double y)
    {
        if(y <= 0)
            throw new ArithmeticException("The scale parameter must be > 0, not " + y);
        this.y = y;
    }

    public double getY()
    {
        return y;
    }

    public double getX0()
    {
        return x0;
    }
    
    @Override
    public double pdf(double x)
    {
        return 1.0 / ( Math.PI*y*  (1 + Math.pow((x-x0)/y, 2))  );
    }

    @Override
    public double cdf(double x)
    {
        return Math.atan((x-x0)/y)/Math.PI + 0.5;
    }

    @Override
    public double invCdf(double p)
    {
        return x0 + y * Math.tan(  Math.PI * (p - 0.5) );
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
        return new double[] {x0, y};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if(var.equals("y"))
            setY(value);
        else if(var.equals("x0"))
            setX0(value);
    }

    @Override
    public ContinousDistribution copy()
    {
        return new Cauchy(x0, y);
    }

    @Override
    public void setUsingData(Vec data)
    {
        data = data.sortedCopy();
        
        //approximate y by taking | 1st quant - 3rd quantile|
        int n = data.length();
        setY(Math.abs(data.get(n/4) - data.get(3*n/4)));
        
        //approximate x by taking the median value
        //Note, technicaly, any value is equaly likely to be the true median of a chachy distribution, so we dont care about the exact median
        setX0(data.get(n/2));
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
        return x0;
    }

    @Override
    public double mode()
    {
        return x0;
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
