
package jsat.distributions;

import jsat.linear.Vec;
import jsat.math.TrigMath;
import jsat.text.GreekLetters;

/**
 *
 * @author Edward Raff
 */
public final class Logistic extends ContinuousDistribution
{

	private static final long serialVersionUID = -8720773286818833591L;
	/** 
     * Location
     */
    private double mu;
    /**
     * Scale
     */
    private double s;

    public Logistic(double mu, double s)
    {
        this.mu = mu;
        setS(s);
    }

    public double getS()
    {
        return s;
    }

    public double getMu()
    {
        return mu;
    }

    public void setMu(double mu)
    {
        this.mu = mu;
    }

    public void setS(double s)
    {
        if(s <= 0)
            throw new ArithmeticException("The scale parameter must be > 0, not " + s);
        this.s = s;
    }
    
    @Override
    public double pdf(double x)
    {
        return 1/(4*s) * Math.pow(TrigMath.sech( (x-mu) / (2*s)), 2);
    }

    @Override
    public double cdf(double x)
    {
        return 0.5 + 0.5 * Math.tanh( (x-mu)/(2*s));
    }

    @Override
    public double invCdf(double p)
    {
        return mu + s * Math.log( p /(1-p));
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
        return "Logistic";
    }

    @Override
    public String[] getVariables()
    {
        return new String[] {GreekLetters.mu, "s"};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[]{mu, s};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if(var.equals(GreekLetters.mu))
            setMu(value);
        else if(var.equals("s"))
            setS(value);
    }

    @Override
    public ContinuousDistribution clone()
    {
        return new Logistic(mu, s);
    }

    @Override
    public void setUsingData(Vec data)
    {
        double newS = data.variance()*(3/(Math.PI*Math.PI));
        newS = Math.sqrt(newS);
        
        setS(newS);
        setMu(data.mean());
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
        return Math.PI*Math.PI/3*s*s;
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
		temp = Double.doubleToLongBits(mu);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(s);
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
		Logistic other = (Logistic) obj;
		if (Double.doubleToLongBits(mu) != Double.doubleToLongBits(other.mu)) {
			return false;
		}
		if (Double.doubleToLongBits(s) != Double.doubleToLongBits(other.s)) {
			return false;
		}
		return true;
	}
    
}
