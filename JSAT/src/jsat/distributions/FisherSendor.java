
package jsat.distributions;

import jsat.linear.Vec;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;


/**
 *
 * Also known as the F distribution. 
 * 
 * @author Edward Raff
 */
public class FisherSendor extends ContinuousDistribution
{
    

	private static final long serialVersionUID = 7628304882101574242L;
	double v1;
    double v2;

    public FisherSendor(double v1, double v2)
    {
        if(v1 <= 0)
            throw new ArithmeticException("v1 must be > 0 not " + v1 );
        if(v2 <= 0)
            throw new ArithmeticException("v2 must be > 0 not " + v2 );
        this.v1 = v1;
        this.v2 = v2;
    }

    @Override
    public double logPdf(double x)
    {
        if(x <= 0)
            return 0;
        double leftSide = v1/2 * log(v1) + v2/2*log(v2) - lnBeta(v1/2, v2/2); 
        double rightSide = (v1/2-1)*log(x) - (v1+v2)/2*log(v2+v1*x);
        return leftSide+rightSide;
    }
    
    

    @Override
    public double pdf(double x)
    {
        if(x <= 0)
            return 0;
        return exp(logPdf(x));
    }


    @Override
    public double cdf(double x)
    {
        if(x <= 0)
            return 0;
        return betaIncReg(v1*x / (v1*x + v2), v1/2, v2/2);
    }

    @Override
    public double invCdf(double p)
    {
        if(p < 0 || p > 1)
            throw new ArithmeticException("Probability must be in the range [0,1], not" + p);
        double u = invBetaIncReg(p, v1/2, v2/2);
        return v2*u/(v1*(1-u));
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
        return "F";
    }

    @Override
    public String[] getVariables()
    {
        return new String[]{"v1", "v2"};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[]{v1, v2};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if (var.equals("v1"))
            if (value > 0)
                v1 = value;
            else
                throw new ArithmeticException("v1 must be > 0 not " + value);
        else if (var.equals("v2"))
            if (value > 0)
                v2 = value;
            else
                throw new ArithmeticException("v2 must be > 0 not " + value );
    }

    @Override
    public ContinuousDistribution clone()
    {
        return new FisherSendor(v1, v2);
    }

    @Override
    public void setUsingData(Vec data)
    {
        double mu = data.mean();
        
        //Only true if v2 >  2
        double tmp = 2*mu / (-1 + mu);
        
        
        if(tmp < 2)
        {
            return;//We couldnt approximate anything
            
        }
        else
        {
            v2 = tmp;
            if(v2 < 4)
                return;//We cant approximate v1
        }
        
        //only true if v2 > 4
        double v2sqr = v2*v2;
        double var = data.variance();
        double denom = -2*v2sqr - 16*var + 20*v2*var - 8*v2sqr*var + v2sqr*v2*var;
        
        v1 = 2*(-2*v2sqr + v2sqr*v2)/denom;
    }

    @Override
    public double mean()
    {
        if(v2 <= 2)
            return Double.NaN;
        
        return v2/(v2-2);
    }

    @Override
    public double median()
    {
        return (v2/v1)*(1.0/invBetaIncReg(0.5, v2/2, v1/2)-1);
    }

    @Override
    public double mode()
    {
        if(v1 <= 2)
            return Double.NaN;
        
        return (v1-2)/v1*v2/(v2+2);
    }

    @Override
    public double variance()
    {
        if(v2 <= 4)
            return Double.NaN;
        
        return 2 * v2*v2*(v1+v2-2) / (v1*pow(v2-2,2)*(v2-4));
    }

    @Override
    public double skewness()
    {
        
        if(v2 <= 6)//Does not have a skewness for d2 <= 6
            return Double.NaN;
        double num = (2*v1+v2-2)*sqrt(8*(v2-4));
        double denom = (v2-6)*sqrt(v1*(v1+v2-2));
        
        return num/denom;
    }

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(v1);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(v2);
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
		FisherSendor other = (FisherSendor) obj;
		if (Double.doubleToLongBits(v1) != Double.doubleToLongBits(other.v1)) {
			return false;
		}
		if (Double.doubleToLongBits(v2) != Double.doubleToLongBits(other.v2)) {
			return false;
		}
		return true;
	}
    
}
