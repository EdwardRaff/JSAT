
package jsat.distributions;

import jsat.linear.Vec;
import jsat.testing.goodnessoffit.KSTest;

/**
 *
 * @author Edward Raff
 */
public class DistributionSearch
{
    private static ContinousDistribution[] possibleDistributions = new ContinousDistribution[] 
    { 
        new Normal(), 
        new LogNormal(), new Exponential(),
        new Gamma(2, 1), new FisherSendor(10, 10), new Weibull(2, 1), 
        new Uniform(0, 1)
    };
    
    /**
     * Searches the distributions that are known for a possible fit, and returns 
     * what appears to be the best fit. 
     * 
     * @param v all the values from a sample
     * @return the distribution that provides the best fit to the data that this method could find.
     */
    public static ContinousDistribution getBestDistribution(Vec v)
    {
        //Thread Safety, copy the possible distributions
        
        ContinousDistribution[] possDistCopy = new ContinousDistribution[possibleDistributions.length];
        
        for(int i = 0; i < possibleDistributions.length; i++)
            possDistCopy[i] = possibleDistributions[i].copy();
        
        
        KSTest ksTest = new KSTest(v);
        
        ContinousDistribution bestDist = null;
        double bestProb = 0;
        
        for(ContinousDistribution cd : possDistCopy)
        {
            try
            {
                cd.setUsingData(v);
                double prob = ksTest.testDist(cd);
                
                if(prob > bestProb)
                {
                    bestDist = cd;
                    bestProb = prob;
                }
                
            }
            catch(Exception ex)
            {
                
            }
        }
        
        ///Return the best distribution, or if somehow everythign went wrong, a normal distribution
        try
        {
            return bestDist == null ? new Normal(v.mean(), v.standardDeviation()) : bestDist.copy();
        }
        catch (RuntimeException ex)//Mostly likely occurs if all values are all zero
        {
            if(v.standardDeviation() == 0)
                return new Normal(v.mean(), 0.1);
            throw new ArithmeticException("Catistrophic faulure getting a distribution");
        }
    }
}
