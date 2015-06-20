
package jsat.distributions;

import jsat.linear.Vec;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;
import jsat.text.GreekLetters;



/**
 * 
 * @author Edward Raff
 */
public class StudentT extends ContinuousDistribution
{

	private static final long serialVersionUID = -3157525461647767831L;
	double df;
    double mu;
    double sig;

    public StudentT(double df)
    {
        this(df, 0, 1);
    }

    public StudentT(double df, double mu, double sig)
    {
        this.df = df;
        this.mu = mu;
        this.sig = sig;
    }

    
    /**
     * Sets the degrees of freedom used by the test.
     * 
     * @param df the new value for the degrees of freedom. 
     */
    public void setDf(double df)
    {
        this.df = df;
    }

    
    @Override
    public double pdf(double t)
    {
        
        double leftSide = lnGamma((df+1)/2) - lnGamma(df/2) - lnGamma(df*PI)/2 - log(sig);
        double rightSide = -(df+1)/2*log(1+pow((t-mu)/sig, 2)/df);
        
        return exp(leftSide+rightSide);
    }


    @Override
    public double cdf(double t)
    {
        double x = df/(df + pow((t-mu)/sig, 2));
        
        double p =  betaIncReg(x, df/2, 0.5)/2;
        
        if( t > mu)
            return 1 - p;
        else 
            return p;
    }

    @Override
    public double invCdf(double p)
    {
        if(p < 0 || p > 1)
            throw new ArithmeticException("Probability must be in the range [0,1], not " + p);
        double x = invBetaIncReg(2*Math.min(p,1-p), df/2, 0.5);
        x = sig*sqrt(df*(1-x)/x);

        if(p >= 0.5)
            return mu+x;
        else
            return mu-x;
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
    public String getDescriptiveName()
    {
        return "Student-T(df=" + df +", \u03BC=" + mu + ", \u03C3=" + sig + ")";
    }

    @Override
    public String getDistributionName()
    {
        return "Student-T";
    }

    @Override
    public String[] getVariables()
    {
        return new String[]{"df", GreekLetters.mu, GreekLetters.sigma};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[]{df, mu, sig};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if (var.equals("df"))
            if (value > 0)
                df = value;
            else
                throw new ArithmeticException("Degrees of Fredom must be greater than 0");
        else if (var.equals(GreekLetters.mu))
            mu = value;
        else if (var.equals(GreekLetters.sigma))
            if (value > 0)
                sig = value;
            else
                throw new ArithmeticException("Standard deviation must be greater than zero");
                        
    }

    @Override
    public ContinuousDistribution clone()
    {
        return new StudentT(df, mu, sig);
    }

    @Override
    public void setUsingData(Vec data)
    {
        /*
         * While not true in every use of the t-distribution, 
         * we assume degrees of fredom is n-1 if n is the number of samples
         * 
         */
        df = data.length()-1;
        mu = data.mean();
        sig = sqrt(data.variance()*df/(df-2));
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
        if(df<=1)
            return Double.NaN;
        else if(1 < df && df <= 2)
            return Double.POSITIVE_INFINITY;
        return df/(df-2)*sig*sig;
    }

    @Override
    public double skewness()
    {
        if(df <= 3)//Undefined for df <= 3
            return Double.NaN;
        return 0;
    }

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(df);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(mu);
		result = prime * result + (int) (temp ^ (temp >>> 32));
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
		StudentT other = (StudentT) obj;
		if (Double.doubleToLongBits(df) != Double.doubleToLongBits(other.df)) {
			return false;
		}
		if (Double.doubleToLongBits(mu) != Double.doubleToLongBits(other.mu)) {
			return false;
		}
		if (Double.doubleToLongBits(sig) != Double.doubleToLongBits(other.sig)) {
			return false;
		}
		return true;
	}
    
}
