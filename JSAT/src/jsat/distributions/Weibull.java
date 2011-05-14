
package jsat.distributions;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.SimpleLinearRegression;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;
/**
 *
 * @author Edward Raff
 */
public class Weibull extends ContinousDistribution
{
    /**
     * Shape parameter
     */
    double alpha;
    /**
     * Scale parameter
     */
    double beta;

    public Weibull(double k, double gam)
    {
        if(k <= 0)
            throw new ArithmeticException("k must be > 0 not " + k );
        if(gam <= 0)
            throw new ArithmeticException("k must be > 0 not " + gam );
        this.alpha = k;
        this.beta = gam;
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
                throw new ArithmeticException("k must be > 0 not " + value);
        else if (var.equals("beta"))
            if (value > 0)
                beta = value;
            else
                throw new ArithmeticException("gam must be > 0 not " + value );
    }

    @Override
    public ContinousDistribution copy()
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
        
        alpha = s[1];
        
        /*
         * We can now compute alpha directly. 
         * Note the page use y = m x + b, instead of y = b x + a
         * 
         */
        
        beta = exp(-s[0]/alpha);
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
    
}
