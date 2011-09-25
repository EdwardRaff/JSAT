
package jsat.distributions;

import jsat.linear.Vec;
import jsat.text.GreekLetters;
import static java.lang.Math.*;
/**
 *
 * @author Edward Raff
 */
public class Pareto extends ContinousDistribution
{
    /**
     * scale
     */
    private double xm;
    /**
     * shape
     */
    private double alpha;

    public Pareto()
    {
        this(1, 3);
    }

    public Pareto(double xm, double alpha)
    {
        setXm(xm);
        setAlpha(alpha);
    }
    
    public final void setAlpha(double alpha)
    {
        if(alpha <= 0)
            throw new ArithmeticException("Shape parameter must be > 0, not " + alpha);
        this.alpha = alpha;
    }

    public final void setXm(double xm)
    {
        if(xm <= 0)
            throw new ArithmeticException("Scale parameter must be > 0, not " + xm);
        this.xm = xm;
    }
    
    public double logPdf(double x)
    {
        if(x < xm )
            return Double.NEGATIVE_INFINITY;
        
        return log(alpha) + alpha*log(xm) - (alpha+1)*log(x);
    }

    @Override
    public double pdf(double x)
    {
        if(x < xm )
            return 0;
        return exp(logPdf(x));
    }

    @Override
    public double cdf(double x)
    {
        return 1 - exp( alpha * log(xm/x));
    }

    @Override
    public double invCdf(double p)
    {
        return xm * pow(1-p, -1/alpha);
    }

    @Override
    public double min()
    {
        return xm;
    }

    @Override
    public double max()
    {
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public String getDistributionName()
    {
        return "Pareto";
    }

    @Override
    public String[] getVariables()
    {
        return new String[] {"x_m", GreekLetters.alpha};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[] {xm, alpha};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if(var.equals("x_m"))
            setXm(value);
        else if(var.equals(GreekLetters.alpha))
            setAlpha(value);
    }

    @Override
    public ContinousDistribution copy()
    {
        return new Pareto(xm, alpha);
    }

    @Override
    public void setUsingData(Vec data)
    {
        double mean = data.mean();
        double var = data.variance();
        
        double aP = sqrt( (mean*mean+var)/var), alphaC = aP+1;
        double xmC = mean*aP /alphaC;
        
        if(alphaC > 0 && xmC > 0)
        {
            setAlpha(alphaC);
            setXm(xmC);
        }
        
    }

    @Override
    public double mean()
    {
        if(alpha > 1)
            return alpha*xm/(alpha-1);
        return Double.NaN;
    }

    @Override
    public double mode()
    {
        return xm;
    }

    @Override
    public double variance()
    {
        if(alpha > 2)
            return xm*xm*alpha/ (pow(alpha-1, 2)*(alpha-2) );
        
        return Double.NaN;
    }

    @Override
    public double skewness()
    {
        return sqrt((alpha-2)/alpha)*(2*(1+alpha)/(alpha-3));
    }
    
}
