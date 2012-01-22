
package jsat.distributions.empirical;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import jsat.distributions.ContinousDistribution;
import jsat.distributions.Normal;
import jsat.distributions.empirical.kernelfunc.GaussKF;
import jsat.distributions.empirical.kernelfunc.KernelFunction;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.OnLineStatistics;
import jsat.utils.IndexTable;
import jsat.utils.ProbailityMatch;

/**
 * Kernel Density Estimator, KDE, uses the data set itself to approximate the underlying probability 
 * distribution using {@link KernelFunction Kernel Functions}. 
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
    
    /**
     * The various values 
     */
    private double[] X;
    /**
     * Weights coresponding to each value. If all the same, weights should have a length of 0
     */
    private double[] weights;
    /**
     * For unweighted data, this is equal to X.length 
     */
    private double sumOFWeights;
    /**
     * The bandwidth
     */
    private double h;
    double Xmean, Xvar, Xskew;
    
    private KernelFunction k;
    
    
    public static double BandwithGuassEstimate(Vec X)
    {
        if(X.length() == 1 )
            return 1;
        else if(X.standardDeviation() == 0)
            return 1.06 * Math.pow(X.length(), -1.0/5.0);
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
    
    public KernelDensityEstimator(Vec dataPoints, KernelFunction k, double[] weights)
    {
        this(dataPoints, k, BandwithGuassEstimate(dataPoints), weights);
    }

    public KernelDensityEstimator(Vec dataPoints, KernelFunction k, double h)
    {
        setUpX(dataPoints);
        this.k = k;
        this.h = h;
    }
    
    public KernelDensityEstimator(Vec dataPoints, KernelFunction k, double h, double[] weights)
    {
        setUpX(dataPoints, weights);
        this.k = k;
        this.h = h;
    }

    /**
     * Copy constructor 
     */
    private KernelDensityEstimator(double[] X, double h, double Xmean, double Xvar, double Xskew, KernelFunction k, double sumOfWeights, double[] weights)
    {
        this.X = Arrays.copyOf(X, X.length);
        this.h = h;
        this.Xmean = Xmean;
        this.Xvar = Xvar;
        this.Xskew = Xskew;
        this.k = k;
        this.sumOFWeights = sumOfWeights;
        this.weights = Arrays.copyOf(weights, weights.length);
    }
    
    private void setUpX(Vec S)
    {
        Xmean = S.mean();
        Xvar = S.variance();
        Xskew = S.skewness();
        X = S.arrayCopy();
        Arrays.sort(X);
        sumOFWeights = X.length;
        weights = new double[0];
    }
    
    private void setUpX(Vec S, double[] weights)
    {
        if(S.length() != weights.length)
            throw new RuntimeException("Weights and variables do not have the same length");
        
        OnLineStatistics stats = new OnLineStatistics();
        
        X = new double[S.length()];
        this.weights = Arrays.copyOf(weights, weights.length);
        
        //Probability is the X value, mattch is the weights - so that they can be sorted together. 
        List<ProbailityMatch<Double>> sorter = new ArrayList<ProbailityMatch<Double>>(S.length());
        for(int i = 0; i < S.length(); i++)
            sorter.add(new ProbailityMatch<Double>(S.get(i), weights[i]));
        Collections.sort(sorter);
        for(int i = 0; i < sorter.size(); i++)
        {
            this.X[i] = sorter.get(i).getProbability();
            this.weights[i] = sorter.get(i).getMatch();
            stats.add(this.X[i], this.weights[i]);
        }
        //Now do some helpfull preprocessing on weights. We will make index i store the sum for [0, i]. 
        //Each individual weight can still be retrived in O(1) by accesing a 2nd index and a subtraction
        //Methods that need the sum can now acces it in O(1) time from the weights array isntead of doing an O(n) summations
        for(int i = 1; i < this.weights.length; i++)
            this.weights[i] += this.weights[i-1];
        sumOFWeights = this.weights[this.weights.length-1];
        this.Xmean = stats.getMean();
        this.Xvar = stats.getVarance();
        this.Xskew = stats.getSkewness();
    }
    
    private double getWeight(int i)
    {
        if(weights.length == 0)
            return 1.0;
        else if(i == 0)
            return weights[i];
        else
            return weights[i] - weights[i-1];
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
            sum += k.k( (x-X[i])/h )*getWeight(i);
        
        return sum / (sumOFWeights * h);
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
            sum += k.intK( (x-X[i]) /h )*getWeight(i);
        
        /* 
         * Slightly different, all things below the from value for the cdf would be 
         * adding 1 to the value, as the value of x would be the integration over 
         * the entire range, which by definition, is equal to 1.
         */
        //We perform the addition after the summation to reduce the differnce size
        if(weights.length == 0)//No weights
            sum += Math.max(0, from);
        else
            sum += weights[from];
            
        
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
        int index;
        double kd0;
        
        if(weights.length == 0)
        {
            double r = p*X.length;
            index = (int)r;
            double pd0 = r - index, pd1 = 1-pd0;
            kd0 = k.intK(pd1);
        }
        else//CDF can be found from the weights summings
        {
            double XEstimate = p*sumOFWeights;
            index = Arrays.binarySearch(weights, XEstimate);
            index = index < 0 ? -index-1 : index;
            if(X[index] != 0)//TODO fix this bit
                kd0 = 1.0;//-Math.abs((XEstimate-X[index])/X[index]);
            else
                kd0 = 1.0;
        }
        
        
        if(index == X.length-1)//at the tail end
            return X[index]*kd0;
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
    public void setBandwith(double val)
    {
        if(val <= 0 || Double.isInfinite(val))
            throw new ArithmeticException("Bandwith parameter h must be greater than zero, not " + 0);
        this.h = val;
    }

    /**
     * 
     * @return the bandwidth parameter 
     */
    public double getBandwith()
    {
        return h;
    }
    
    @Override
    public void setVariable(String var, double value)
    {
        if(var.equals("h"))
            setBandwith(value);
    }

    @Override
    public ContinousDistribution clone()
    {
        return new KernelDensityEstimator(X, h, Xmean, Xvar, Xskew, k, sumOFWeights, weights);
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
