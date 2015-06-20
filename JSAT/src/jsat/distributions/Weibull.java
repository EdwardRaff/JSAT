
package jsat.distributions;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.SimpleLinearRegression;
import jsat.text.GreekLetters;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;
/**
 *
 * @author Edward Raff
 */
public class Weibull extends ContinuousDistribution
{

	private static final long serialVersionUID = -4083186674624535562L;
	/**
     * Shape parameter
     */
    double alpha;
    /**
     * Scale parameter
     */
    double beta;
    
    private double logAlpha, logBeta;

    public Weibull(double alpha, double beta)
    {
        setAlpha(alpha);
        setBeta(beta);
    }

    
    public double reliability(double x)
    {
        return exp(-pow(x/alpha, beta));
    }
    
    public double failureRate(double x)
    {
        return beta/alpha * pow(x/alpha, beta-1);
    }

    @Override
    public double logPdf(double x)
    {
        if(x <= 0)
            return -Double.MAX_VALUE;
        return logAlpha-logBeta+(alpha-1)*log(x/beta) -pow(x/beta, alpha) ;
    }

    
    
    @Override
    public double pdf(double x)
    {
        if(x < 0)
            return 0;
        
        return alpha/beta * pow(x/beta, alpha-1)*exp(-pow(x/beta, alpha));
        
    }

    @Override
    public double cdf(double x)
    {
        return 1 - exp(-pow(x/beta, alpha));
    }

    @Override
    public double invCdf(double p)
    {
        return beta*pow(-log(1-p),1/alpha);
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
        return "Weibull";
    }

    @Override
    public String[] getVariables()
    {
        return new String[]{GreekLetters.alpha, GreekLetters.beta};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[]{alpha, beta};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if (var.equals("alpha") || var.equals(GreekLetters.alpha))
            setAlpha(value);
        else if (var.equals("beta") || var.equals(GreekLetters.beta))
            setBeta(value);
    }

    final public void setAlpha(double alpha)
    {
        if(alpha > 0)
        {
            this.alpha = alpha;
            logAlpha = log(alpha);
        }
        else
            throw new ArithmeticException("alpha must be > 0 not " + alpha);
    }

    final public void setBeta(double beta)
    {
        if(beta > 0)
        {
            this.beta = beta;
            logBeta = log(beta);
        }
        else
            throw new ArithmeticException("beta must be > 0 not " + beta);
    }
    
    

    @Override
    public ContinuousDistribution clone()
    {
        return new Weibull(alpha, beta);
    }

    @Override
    public void setUsingData(Vec data)
    {
        /* Method of parameter esstimation is more complex than for 
         * other distirbutions, see 
         * http://www.qualitydigest.com/jan99/html/body_weibull.html 
         * for the method used. NOTE the above article has alpha and beta in oposite order
         */
        
        Vec sData = data.sortedCopy();
        DenseVector ranks = new DenseVector(sData.length());
        for(int i = 0; i < sData.length(); i++)
        {
            //Get the median rank
            double tmp = (i+1.0-0.3)/(sData.length()+0.4);

            tmp = 1/(1-tmp);
            
            tmp = log(log(tmp));
                        
            ranks.set(i, tmp);
            
            sData.set(i, log(sData.get(i)));
        }
        
        
        double[] s = SimpleLinearRegression.regres(sData, ranks);
        
        //The shape parameter is approximatly the slope
        
        setAlpha(s[1]);
        
        /*
         * We can now compute alpha directly. 
         * Note the page use y = m x + b, instead of y = b x + a
         * 
         */
        
        setBeta(exp(-s[0]/alpha));
    }

    @Override
    public double mean()
    {
        return beta * gamma(1+1/alpha);
    }

    @Override
    public double median()
    {
        return pow(log(2), 1/alpha)*beta;
    }

    @Override
    public double mode()
    {
        if(alpha <= 1)
            throw new ArithmeticException("Mode only exists for k > 1");
        
        return beta * pow( (alpha-1)/alpha, 1/alpha);
    }

    @Override
    public double variance()
    {
        return beta*beta * gamma(1+2/alpha) - pow(median(),2);
    }

    @Override
    public double skewness()
    {
        double mu = mean();
        double stnDev = standardDeviation();
        return (gamma(1 + 3/alpha)*pow(beta, 3)-3*mu*pow(stnDev, 2)-pow(mu, 3))/pow(stnDev, 3);
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
		Weibull other = (Weibull) obj;
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
