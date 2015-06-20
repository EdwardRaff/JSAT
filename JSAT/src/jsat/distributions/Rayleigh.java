
package jsat.distributions;

import jsat.linear.Vec;
import jsat.text.GreekLetters;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public final class Rayleigh extends ContinuousDistribution
{


	private static final long serialVersionUID = 1451949391703281531L;
	/**
     * scale parameter
     */
    private double sig;

    public Rayleigh(double sig)
    {
        setScale(sig);
    }
    
    

    public void setScale(double sig)
    {
        if(sig <=0 || Double.isInfinite(sig) || Double.isNaN(sig))
            throw new ArithmeticException("The " + GreekLetters.sigma + " parameter must be > 0, not " + sig);
        this.sig = sig;
    }

    public double getScale()
    {
        return sig;
    }
    
    @Override
    public double pdf(double x)
    {
        if (x < 0)
            return 0;
        double sigSqr = sig*sig;
        return x / sigSqr * exp(-x*x/(2*sigSqr));
    }

    @Override
    public double cdf(double x)
    {
        double sigSqr = sig*sig;
        return 1 - exp(-x*x/(2*sigSqr));
    }

    @Override
    public double invCdf(double p)
    {
        return sqrt(sig*sig*log(1/(1-p)))*sqrt(2.0);
    }

    @Override
    public double min()
    {
        return 0;
    }

    @Override
    public double max()
    {
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public String getDistributionName()
    {
        return "Rayleigh";
    }

    @Override
    public String[] getVariables()
    {
        return new String[]{GreekLetters.sigma};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[]{sig};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if(var.equals(GreekLetters.sigma))
            setScale(value);
    }

    @Override
    public ContinuousDistribution clone()
    {
        return new Rayleigh(sig);
    }

    @Override
    public void setUsingData(Vec data)
    {
        /**
         * 
         *                 ____________
         *                /      N
         *               /     =====
         *              /   1  \      2
         * sigma =     /   ---  >    x
         *            /    2 N /      i
         *           /         =====
         *         \/          i = 1
         * 
         */
        
        
        //TODO Need to add some API to SparceVector to make this summation more efficient
        double tmp = 0;
        for(int i = 0; i < data.length(); i++)
            tmp += pow(data.get(i), 2);
        
        tmp /= (2*data.length());
        tmp = sqrt(tmp);
        
        setScale(tmp);
    }

    @Override
    public double mean()
    {
        return sig*sqrt(PI/2);
    }

    @Override
    public double median()
    {
        return sig*sqrt(log(4));
    }

    @Override
    public double mode()
    {
        return sig;
    }

    @Override
    public double variance()
    {
        return (4-PI)/2*sig*sig;
    }

    @Override
    public double skewness()
    {
        return 2*sqrt(PI)*(PI-3)/(pow(4-PI, 3.0/2.0));
    }



	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(sig);
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
		Rayleigh other = (Rayleigh) obj;
		if (Double.doubleToLongBits(sig) != Double.doubleToLongBits(other.sig)) {
			return false;
		}
		return true;
	}
    
}
