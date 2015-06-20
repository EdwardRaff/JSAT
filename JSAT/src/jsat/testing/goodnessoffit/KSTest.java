
package jsat.testing.goodnessoffit;

import jsat.distributions.ContinuousDistribution;
import jsat.distributions.Kolmogorov;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class KSTest
{
    private static final Kolmogorov k = new Kolmogorov();
    
    private Vec v;

    /**
     * Creates a new statistical test for testing. The 1 sample test, with <tt>v</tt> 
     * being the 1 sample. The 1 sample test compare the data to a given distribution, 
     * and see if it does not belong to the given distribution. The 2 sample test is 
     * designed to tell if the data is not from the same population. 
     * 
     * @param v the date to be one of the samples
     */
    public KSTest(Vec v)
    {
        this.v = v.sortedCopy();
    }

    
    /**
     * Change the original sample to <tt>v</tt>
     * @param v the new original sample. 
     */
    public void setBaseData(Vec v)
    {
        this.v = v;
    }
    
    
    
    
    /**
     * Calculates the D statistic for comparison against a continous distribution
     * @param cd the distribution to compare against
     * @return the max difference between the empirical CDF and the 'true' CDF of the given distribution
     */
    protected double dCalc(ContinuousDistribution cd)
    {
        double max = 0;
        
        for(int i = 0; i < v.length(); i++)
        {
            //ECDF(x) - F(x)
            if(v.get(i) >= cd.min() && v.get(i) <= cd.max() )
            {
                double tmp = (i+1.0)/v.length() - cd.cdf(v.get(i));
                max = Math.max(max, Math.abs(tmp));
            }
            else//The data dose not fit in the rang eof the distribution
            {
                max = Math.max(max, Math.abs((i+1.0)/v.length()));
            }
        }
        
        
        return max;
    }
    
    private static double ECDF(Vec s, double x)
    {
        int min = 0;
        int max = s.length()-1;
        int mid = (min+max) /2;
        do
        {
            if(x > s.get(mid))
                min = mid+1;
            else
                max = mid-1;
        }
        while(s.get(mid) != x && min <= max);
        
        return (mid+1.0)/s.length();
    }
    
    /**
     * Calculates the D statistic for comparison against another data set
     * @param o the other data set
     * @return the max difrence in empirical CDF of the distributions.
     */
    protected double dCaldO(Vec o)
    {
        double max = 0;
        
        for(int i = 0; i < v.length(); i++)
        {
            //ECDF(x) - F(x)
            double tmp = (i+1.0)/v.length() - ECDF(o, v.get(i));
            max = Math.max(max, Math.abs(tmp));
        }
        
        for(int i = 0; i < o.length(); i++)
        {
            //ECDF(x) - F(x)
            double tmp = (i+1.0)/o.length() - ECDF(v, o.get(i));
            max = Math.max(max, Math.abs(tmp));
        }
        
        return max;
    }
    
    /**
     * Returns the p-value for the KS Test against the given distribution <tt>cd</tt>. <br>
     * The null hypothesis of this test is that the given data set belongs to the given distribution. <br>
     * The alternative hypothesis is that the data set does not belong to the given distribution. 
     * 
     * 
     * @param cd the distribution to compare against
     * @return the p-value of the test against this distribution
     */
    public double testDist(ContinuousDistribution cd)
    {
        double d = dCalc(cd);
        double n = v.length();
        
        return pValue(n, d);
    }
    
    /**
     * Returns the p-value for the 2 sample KS Test against the given data set <tt>data</tt>. <br>
     * The null hypothesis of this test is that the given data set is from the same population as <tt>data</tt> <br>
     * The alternative hypothesis is that the data set does not belong to the same population as <tt>data</tt>
     * @param data the other distribution to compare against
     * @return the p-value of the test against this data set
     */
    public double testData(Vec data)
    {
        double d = dCaldO(data);
        double n = v.length()*data.length() / ((double) v.length() +data.length());
        
        return pValue(n, d);
    }
    
    private double pValue(double n, double d)
    {
        return 1 - k.cdf( (Math.sqrt(n) + 0.12 + 0.11/Math.sqrt(n)) * d);
    }
    
}
