
package jsat.distributions;

import jsat.linear.Vec;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;
/**
 *
 * @author Edward Raff
 */
public class Beta extends ContinuousDistribution
{

	private static final long serialVersionUID = 8001402067928143972L;
	double alpha;
    double beta;

    public Beta(double alpha, double beta)
    {
        if(alpha <= 0)
            throw new ArithmeticException("Alpha must be > 0, not " + alpha);
        else if(beta <= 0)
            throw new ArithmeticException("Beta must be > 0, not " + beta);
        this.alpha = alpha;
        this.beta = beta;
    }

    @Override
    public double pdf(double x)
    {
        if(x <= 0)
            return 0;
        else if(x >= 1)
            return 0;
        return exp((alpha-1)*log(x)+(beta-1)*log(1-x)-lnBeta(alpha, beta));
    }

    @Override
    public double cdf(double x)
    {
        if(x <= 0)
            return 0;
        else if(x >= 1)
            return 1;
        return betaIncReg(x, alpha, beta);
    }

    @Override
    public double invCdf(double p)
    {
        if(p < 0 || p > 1)
            throw new ArithmeticException("p must be in the range [0,1], not " + p);
        return invBetaIncReg(p, alpha, beta);
    }

    @Override
    public double min()
    {
        return 0;
    }

    @Override
    public double max()
    {
        return 1;
    }

    @Override
    public String getDistributionName()
    {
        return "Beta";
    }

    @Override
    public String[] getVariables()
    {
        return new String[]{"alpha", "beta"};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[]{alpha, beta};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if (var.equals("alpha"))
            if (value > 0)
                alpha = value;
            else
                throw new RuntimeException("Alpha must be > 0, not " + value);
        else if (var.equals("beta"))
            if (value > 0)
                beta = value;
            else
                throw new RuntimeException("Beta must be > 0, not " + value);
    }

    @Override
    public ContinuousDistribution clone()
    {
        return new Beta(alpha, beta);
    }

    @Override
    public void setUsingData(Vec data)
    {
        double mean = data.mean();
        double var = data.variance();
        
        //alpha = (mean^2 - mean^3 - mean * var) / var
        alpha = (mean*mean-mean*mean*mean-mean*var)/var;
        beta = (alpha-alpha*mean)/mean;
    }

    @Override
    public double mean()
    {
        return alpha/(alpha+beta);
    }

    @Override
    public double median()
    {
        return invBetaIncReg(0.5, alpha, beta);
    }

    @Override
    public double mode()
    {
        if(alpha > 1 && beta > 1)
            return (alpha-1)/(alpha+beta-2);
        else
            return Double.NaN;
    }

    @Override
    public double variance()
    {
        return alpha*beta / (pow(alpha+beta, 2)*(alpha+beta+1));
    }

    @Override
    public double skewness()
    {
        return 2*(beta-alpha)*sqrt(alpha+beta+1)/((alpha+beta+2)*sqrt(alpha*beta));
    }

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(alpha);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(beta);
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
		Beta other = (Beta) obj;
		if (Double.doubleToLongBits(alpha) != Double
				.doubleToLongBits(other.alpha)) {
			return false;
		}
		if (Double.doubleToLongBits(beta) != Double
				.doubleToLongBits(other.beta)) {
			return false;
		}
		return true;
	}
    
    
}
