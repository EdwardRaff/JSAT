/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package jsat.distributions;

import static java.lang.Math.*;
import jsat.linear.Vec;
import jsat.text.GreekLetters;
/**
 *
 * @author Edward Raff
 */
public class Normal extends ContinuousDistribution
{

	private static final long serialVersionUID = -5298346576152986165L;
	private double mean;
    private double stndDev;

    public Normal()
    {
        this(0, 1);
    }

    public Normal(final double mean, final double stndDev)
    {
        if(stndDev <= 0) {
          throw new RuntimeException("Standerd deviation of the normal distribution needs to be greater than zero");
        }
        setMean(mean);
        setStndDev(stndDev);
    }

    public void setMean(final double mean)
    {
        if(Double.isInfinite(mean) || Double.isNaN(mean)) {
          throw new ArithmeticException("Mean can not be infinite of NaN");
        }
        this.mean = mean;
    }

    public void setStndDev(final double stndDev)
    {
        if(Double.isInfinite(stndDev) || Double.isNaN(stndDev)) {
          throw new ArithmeticException("Standard devation can not be infinite of NaN");
        }
        if(stndDev <= 0) {
          throw new ArithmeticException("The standard devation can not be <= 0");
        }
        this.stndDev = stndDev;
    }
    
    public static double cdf(final double x, final double mu, final double sigma)
    {
        if (Double.isNaN(x) || Double.isInfinite(x)) {
          throw new ArithmeticException("X is not a real number");
        }
        
        return cdfApproxMarsaglia2004(zTransform(x, mu, sigma));
    }

  @Override
    public double cdf(final double x)
    {
        return cdf(x, mean, stndDev);
    }

    public static double invcdf(final double x, final double mu, final double sigma)
    {
        if(x < 0 || x > 1) {
          throw new RuntimeException("Inverse of a probability requires a probablity in the range [0,1], not " + x);
        }
        //http://home.online.no/~pjacklam/notes/invnorm/
        final double a[] =
        {
            -3.969683028665376e+01,2.209460984245205e+02,
            -2.759285104469687e+02,1.383577518672690e+02,
            -3.066479806614716e+01,2.506628277459239e+00
        };

        final double b[] =
        {
            -5.447609879822406e+01,1.615858368580409e+02,
            -1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01
        };

        final double c[] =
        {
            -7.784894002430293e-03,-3.223964580411365e-01,
            -2.400758277161838e+00,-2.549732539343734e+00,
            4.374664141464968e+00,2.938163982698783e+00
        };

        final double d[] =
        {
            7.784695709041462e-03,3.224671290700398e-01,
            2.445134137142996e+00,3.754408661907416e+00
        };

        final double p_low = 0.02425;
        final double p_high = 1 - p_low;

        final double p = x;
        double result;

        if(0 < p && p < p_low)
        {
            final double q = sqrt(-2*log(p));
            result = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
        }
        else if (p_low <= p && p <= p_high)
        {
            final double q = p - 0.5;
            final double r = q*q;
            result = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
                        (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
        }
        else//upper region
        {
            final double q = sqrt(-2*log(1-p));
            result = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
        }

        //Refining step

        final double e = cdf(result, 0, 1) - p;
        final double u = e*sqrt(2*PI)*exp(result*result/2);
        result = result - u / (1 + result*u/2);

        return result * sigma + mu;
    }

  @Override
    public double invCdf(final double d)
    {
        return invcdf(d, mean, stndDev);
    }

    public static double pdf(final double x, final double mu, final double sigma)
    {
        return 1/sqrt(2*PI*sigma*sigma)*exp(-pow(x-mu,2)/(2*sigma*sigma));
    }

  @Override
    public double pdf(final double d)
    {
        return pdf(d, mean, stndDev);
    }
    
    /**
     * Computes the log probability of a given value
     * @param x the value to the get log(pdf) of
     * @param mu the mean of the distribution
     * @param sigma the standard deviation of the distribution
     * @return the log probability 
     */
    public static double logPdf(final double x, final double mu, final double sigma)
    {
        return -0.5*log(2*PI) - log(sigma) + -pow(x-mu,2)/(2*sigma*sigma);
    }

    @Override
    public double logPdf(final double x)
    {
        return logPdf(x, mean, stndDev);
    }

    public double invPdf(final double d)
    {
        /**
         * inverse pdf of a normal distribution is
         *
         *          2
         *  (mu - x)
         *  ---------
         *         2     ____
         *  2 sigma     /  __
         * e          \/ 2 || sigma
         *
         */
        return exp(pow(mean-d, 2)/(2*pow(stndDev, 2)))*sqrt(2*PI)*stndDev;
    }

    public static double zTransform(final double x, final double mu, final double sigma)
    {
        return (x-mu)/sigma;
    }

    public double zTransform(final double x)
    {
        return zTransform(x, mean, stndDev);
    }

    private static double cdfApproxMarsaglia2004(final double x)
    {
        /*
         * Journal of Statistical Software (July 2004, Volume 11, Issue 5), George Marsaglia
         * Algorithum to compute the cdf of the normal distribution for some z score
         */
        double s = x, t = 0, b = x;
        final double q = x*x;
        double i = 1;
        //XXX double comparison
        while(s != t) {
          s=(t=s)+(b*=q/(i+=2));
        }
        return 0.5+s*exp(-.5*q-0.91893853320467274178);
    }

    @Override
    public String getDescriptiveName()
    {
        return "Normal(\u03BC=" + mean + ", \u03C3=" + stndDev + ")";
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
        return "Normal";
    }

    @Override
    public String[] getVariables()
    {
        return new String[]{GreekLetters.mu, GreekLetters.sigma};
    }

    @Override
    public void setVariable(final String var, final double value)
    {
        if(var.equals(GreekLetters.mu)) {
          mean = value;
        } else if(var.equals(GreekLetters.sigma)) {
          setStndDev(value);
        }

    }

    @Override
    public ContinuousDistribution clone()
    {
        return new Normal(mean, stndDev);
    }

    @Override
    public void setUsingData(final Vec data)
    {
        mean = data.mean();
        setStndDev(data.standardDeviation());
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[]{mean, stndDev};
    }

    @Override
    public double mean()
    {
        return mean;
    }

    @Override
    public double median()
    {
        return mean;
    }

    @Override
    public double mode()
    {
        return mean;
    }

    @Override
    public double variance()
    {
        return stndDev*stndDev;
    }

    @Override
    public double standardDeviation()
    {
        return stndDev;
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
		temp = Double.doubleToLongBits(mean);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(stndDev);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(final Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (getClass() != obj.getClass()) {
			return false;
		}
		final Normal other = (Normal) obj;
		if (Double.doubleToLongBits(mean) != Double
				.doubleToLongBits(other.mean)) {
			return false;
		}
		return Double.doubleToLongBits(stndDev) == Double
            .doubleToLongBits(other.stndDev);
	}
    
}
