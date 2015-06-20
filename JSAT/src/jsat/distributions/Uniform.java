
package jsat.distributions;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class Uniform extends ContinuousDistribution
{

	private static final long serialVersionUID = 2479606544724378610L;
	private double a, b;

    public Uniform(double a, double b)
    {
        double min = Math.min(a, b);
        double max = Math.max(a, b);
        
        this.a = min;
        this.b = max;  
    }
    
    @Override
    public double pdf(double x)
    {
        if(a == b && a == x)
            return 0;
        else if(a <=  x && x <= b)
            return 1/(b-a);
        else
            return 0;
    }

    @Override
    public double cdf(double x)
    {
        if(a > x)
            return 0;
        else if( x >= b)
            return 1;
        else if(a == b && a == x)
            return 1;
        else
            return (x-a)/(b-a);
    }

    @Override
    public double invCdf(double p)
    {
        if( p < 0 || p > 1)
            throw new ArithmeticException("Probability must be interface the range [0,1], not " + p);
        
        if(a == b && p == 1)
            return a;
        
        return a + p*(b-a);
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
        return "Uniform";
    }

    @Override
    public String[] getVariables()
    {
        return new String[] {"a", "b"};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[] {a, b};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if(var.equals("a"))
            a = value;
        else if(var.equals("b"))
            b = value;
        
        double min = Math.min(a, b);
        double max = Math.max(a, b);
        a = min;
        b = max;
    }

    @Override
    public ContinuousDistribution clone()
    {
        return new Uniform(a, b);
    }

    @Override
    public void setUsingData(Vec data)
    {
        a = data.min();
        b = data.max();
    }

    @Override
    public double mean()
    {
        return (a+b)*0.5;
    }

    @Override
    public double median()
    {
        return mean();
    }

    @Override
    public double mode()
    {
        return mean();//Any value interface [a,b] can actualy be the mode
    }

    @Override
    public double variance()
    {
        return Math.pow(b-a, 2)/12.0;
    }

    @Override
    public double skewness()
    {
        return 0;
    }

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(a);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(b);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (getClass() != obj.getClass()) {
			return false;
		}
		Uniform other = (Uniform) obj;
		if (Double.doubleToLongBits(a) != Double.doubleToLongBits(other.a)) {
			return false;
		}
		if (Double.doubleToLongBits(b) != Double.doubleToLongBits(other.b)) {
			return false;
		}
		return true;
	}
    
}
