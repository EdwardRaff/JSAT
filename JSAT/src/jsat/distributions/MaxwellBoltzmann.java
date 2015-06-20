
package jsat.distributions;

import jsat.linear.Vec;
import jsat.text.GreekLetters;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;
/**
 *
 * @author Edward Raff
 */
public class MaxwellBoltzmann extends ContinuousDistribution
{

	private static final long serialVersionUID = -8273087046831433430L;
	/**
     * shape
     */
    double sigma;

    public MaxwellBoltzmann()
    {
        this(1);
    }
    
    public MaxwellBoltzmann(double sigma)
    {
        setShape(sigma);
    }
    
    final public void setShape(double sigma)
    {
        if(sigma <= 0 || Double.isInfinite(sigma) || Double.isNaN(sigma))
             throw new ArithmeticException("shape parameter must be > 0, not " + sigma);
        this.sigma = sigma;
    }

    @Override
    public double logPdf(double x)
    {
        if(x <=0 )
            return 0.0;
        return (2*log(x) + (-x*x/(2*sigma*sigma)) - 3*log(sigma) )+ 0.5*(log(2)-log(PI));
    }
    
    @Override
    public double pdf(double x)
    {
        if(x <= 0)
            return 0;
        double x2 = x*x;
        return sqrt(2/PI)*x2*exp(-x2/(2*sigma*sigma))/(sigma*sigma*sigma);
    }

    @Override
    public double cdf(double x)
    {
        if(x <=0 )
            return 0.0;
        return erf(x/(sqrt(2)*sigma))-sqrt(2/PI)*x*exp(-(x*x)/(2*sigma*sigma))/sigma;
    }

    @Override
    public double invCdf(double p)
    {
        if(p < 0 || p > 1)
            throw new ArithmeticException("probability must be in the range [0,1], not " + p);
        
        return sqrt(2)*sigma*sqrt(invGammaP(p, 3.0/2.0));
    }

    @Override
    public double median()
    {
        return sigma*sqrt(2*invGammaP(1.0/2.0, 3.0/2.0));
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
        return "Maxwell–Boltzmann";
    }

    @Override
    public String[] getVariables()
    {
        return new String[] {GreekLetters.sigma};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[] {sigma};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if(var.equals(GreekLetters.sigma))
            setShape(value);
    }

    @Override
    public ContinuousDistribution clone()
    {
        return new MaxwellBoltzmann(sigma);
    }

    @Override
    public void setUsingData(Vec data)
    {
        setShape(data.mean()/sqrt(2));
    }

    @Override
    public double mean()
    {
        return 2*sqrt(2/PI)*sigma;
    }

    @Override
    public double mode()
    {
        return sqrt(2)*sigma;
    }

    @Override
    public double variance()
    {
        return sigma*sigma*(3*PI-8)/PI;
    }

    @Override
    public double skewness()
    {
        return 2*sqrt(2)*(16-5*PI)/pow(3*PI-8, 3.0/2.0);
    }

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(sigma);
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
		MaxwellBoltzmann other = (MaxwellBoltzmann) obj;
		if (Double.doubleToLongBits(sigma) != Double
				.doubleToLongBits(other.sigma)) {
			return false;
		}
		return true;
	}
    
}
