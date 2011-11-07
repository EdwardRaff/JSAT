
package jsat.distributions.empirical;

import java.util.Arrays;
import jsat.distributions.ContinousDistribution;
import jsat.distributions.Normal;
import jsat.distributions.empirical.kernelfunc.GaussKF;
import jsat.distributions.empirical.kernelfunc.KernelFunction;
import jsat.linear.Vec;
import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public class KernelDensityEstimator extends ContinousDistribution 
{
    /*
     * README
     * Implementation note:
     * The values are stored in sorted order, which allows for fast evaluations. 
     * Instead of doing the full loop on each function call, O(n) time, 
     * we know the bounds on the values that will effect results, so we 
     * can do 2 binary searchs and then a loop. Though this is still 
     * technicaly, O(n), its more accuracly describe as O(n * epsilon * log(n)) , where n * epsilon << n
     */
    
    private double[] X;
    /**
     * The bandwidth
     */
    private double h;
    double Xmean, Xvar, Xskew;
    
    private KernelFunction k;
    
    
    public static double BandwithGuassEstimate(Vec X)
    {
        return 1.06 * X.standardDeviation() * Math.pow(X.length(), -1.0/5.0);
    }
    
    public KernelDensityEstimator(Vec dataPoints)
    {
        this(dataPoints, new GaussKF());
    }
    
    public KernelDensityEstimator(Vec dataPoints, KernelFunction k)
    {
        this(dataPoints, k, BandwithGuassEstimate(dataPoints));
    }

    public KernelDensityEstimator(Vec dataPoints, KernelFunction k, double h)
    {
        setUpX(dataPoints);
        this.k = k;
        this.h = h;
    }

    /**
     * Copy constructor 
     */
    private KernelDensityEstimator(double[] X, double h, double Xmean, double Xvar, double Xskew, KernelFunction k)
    {
        this.X = Arrays.copyOf(X, X.length);
        this.h = h;
        this.Xmean = Xmean;
        this.Xvar = Xvar;
        this.Xskew = Xskew;
        this.k = k;
    }
    
    private void setUpX(Vec S)
    {
        Xmean = S.mean();
        Xvar = S.variance();
        Xskew = S.skewness();
        X = S.arrayCopy();
        Arrays.sort(X);
    }
    

    @Override
    public double pdf(double x)
    {
        /*
         *              n
         *            =====  /x - x \
         *         1  \      |     i|
         * f(x) = ---  >    K|------|
         *        n h /      \   h  /
         *            =====
         *            i = 1
         * 
         */
        
        
        //Only values within a certain range will have an effect on the result, so we will skip to that range!
        int from = Arrays.binarySearch(X, x-h*k.cutOff());
        int to = Arrays.binarySearch(X, x+h*k.cutOff());
        //Mostly likely the exact value of x is not in the list, so it retursn the inseration points
        from = from < 0 ? -from-1 : from;
        to = to < 0 ? -to-1 : to;
                
        
        double sum = 0;
        for(int i = Math.max(0, from); i < Math.min(X.length, to+1); i++)
            sum += k.k( (x-X[i])/h );
        
        return sum / (X.length * h);
    }

    @Override
    public double cdf(double x)
    {
        //Only values within a certain range will have an effect on the result, so we will skip to that range!
        int from = Arrays.binarySearch(X, x-h*k.cutOff());
        int to = Arrays.binarySearch(X, x+h*k.cutOff());
        //Mostly likely the exact value of x is not in the list, so it retursn the inseration points
        from = from < 0 ? -from-1 : from;
        to = to < 0 ? -to-1 : to;
        
        double sum = 0;
        
        for(int i = Math.max(0, from); i < Math.min(X.length, to+1); i++)
            sum += k.intK( (x-X[i]) /h );
        
        /* 
         * Slightly different, all things below the from value for the cdf would be 
         * adding 1 to the value, as the value of x would be the integration over 
         * the entire range, which by definition, is equal to 1.
         */
        //We perform the addition after the summation to reduce the differnce size
        sum += Math.max(0, from);
            
        
        return sum / (X.length);
    }

    private final Function cdfFunc = new Function() {

        public double f(double... x)
        {
            return cdf(x[0]);
        }

        public double f(Vec x)
        {
            return f(x.get(0));
        }
    };
    
    @Override
    public double invCdf(double p)
    {
        double r = p*X.length, N =X.length;
        int index = (int)r;
        
        double pd0 = r - index, pd1 = 1-pd0;
        double kd0 = k.intK(pd1);
        double x  = X[index]*kd0 + X[index+1]*(1-kd0);
        
        return x;
    }

    @Override
    public double min()
    {
        return X[0]-h;
    }

    @Override
    public double max()
    {
        return X[X.length-1]+h;
    }

    @Override
    public String getDistributionName()
    {
        return "Kernel Density Estimate";
    }

    @Override
    public String[] getVariables()
    {
        return new String[] { "h" } ;
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[] { h };
    }

    /**
     * Sets the bandwidth used for smoothing. Higher values make the pdf smoother, but can 
     * obscure features. Too small a bandwidth will causes spikes at only the data points.  
     * @param val new bandwidth 
     */
    public void setH(double val)
    {
        if(val <= 0 || Double.isInfinite(val))
            throw new ArithmeticException("Bandwith parameter h must be greater than zero, not " + 0);
        this.h = val;
    }

    /**
     * 
     * @return the bandwidth parameter 
     */
    public double getH()
    {
        return h;
    }
    
    @Override
    public void setVariable(String var, double value)
    {
        if(var.equals("h"))
            setH(value);
    }

    @Override
    public ContinousDistribution copy()
    {
        return new KernelDensityEstimator(X, h, Xmean, Xvar, Xskew, k);
    }

    @Override
    public void setUsingData(Vec data)
    {
        setUpX(data);
        this.h = BandwithGuassEstimate(data);
    }

    @Override
    public double mean()
    {
        return Xmean;
    }


    @Override
    public double mode()
    {
        double maxP = 0, pTmp;
        double maxV = Double.NaN;
        for(int i = 0; i < X.length; i++)
            if( (pTmp = pdf(X[i]) ) > maxP)
            {
                maxP = pTmp;
                maxV = X[i];
            }
        
        return maxV;
    }

    @Override
    public double variance()
    {
        return Xvar + h*h*k.k2();
    }

    @Override
    public double skewness()
    {
        //TODO cant find anything about what this should really be... 
        return Xskew;
    }
    
}
