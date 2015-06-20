
package jsat.distributions;

import jsat.distributions.empirical.KernelDensityEstimator;
import jsat.linear.Vec;
import jsat.testing.goodnessoffit.KSTest;

/**
 * Provides methods for selecting the distribution that best fits a given data set. 
 * @author Edward Raff
 */
public class DistributionSearch
{
    private static ContinuousDistribution[] possibleDistributions = new ContinuousDistribution[] 
    { 
        new Normal(), 
        new LogNormal(), new Exponential(),
        new Gamma(2, 1), new FisherSendor(10, 10), new Weibull(2, 1), 
        new Uniform(0, 1), new Logistic(3, 2), new MaxwellBoltzmann(), 
        new Pareto(), new Rayleigh(2)
    };
    
    /**
     * Searches the distributions that are known for a possible fit, and returns 
     * what appears to be the best fit. 
     * 
     * @param v all the values from a sample
     * @return the distribution that provides the best fit to the data that this method could find.
     */
    public static ContinuousDistribution getBestDistribution(Vec v)
    {
        return getBestDistribution(v, possibleDistributions);
    }
    
    /**
     * Searches the distributions that are known for a possible fit, and returns 
     * what appears to be the best fit. If no suitable fit can be found, a 
     * {@link KernelDensityEstimator} is fit to the data. 
     * 
     * @param v all the values from a sample
     * @param KDECutOff the cut off value used for using the KDE. Should be in 
     * the range (0, 1). Values less than zero means the KDE will never be used,
     * and greater then 1 means the KDE will always be used. 
     * @return the distribution that provides the best fit to the data that this method could find.
     */
    public static ContinuousDistribution getBestDistribution(Vec v, double KDECutOff)
    {
        return getBestDistribution(v, KDECutOff, possibleDistributions);
    }
    
    /**
     * Searches the distributions that are given for a possible fit, and returns 
     * what appears to be the best fit. 
     * 
     * @param v all the values from a sample
     * @param possibleDistributions the array of distribution to try and fit to the data
     * @return the distribution that provides the best fit to the data that this method could find.
     */
    public static ContinuousDistribution getBestDistribution(Vec v, ContinuousDistribution... possibleDistributions)
    {
        return getBestDistribution(v, 0.0, possibleDistributions);
    }
    
    /**
     * Searches the distributions that are given for a possible fit, and returns 
     * what appears to be the best fit. If no suitable fit can be found, a 
     * {@link KernelDensityEstimator} is fit to the data. 
     * 
     * @param v all the values from a sample
     * @param KDECutOff the cut off value used for using the KDE. Should be in 
     * the range (0, 1). Values less than zero means the KDE will never be used,
     * and greater then 1 means the KDE will always be used. 
     * @param possibleDistributions the array of distribution to try and fit to the data
     * @return  the distribution that provides the best fit to the data that this method could find.
     */
    public static ContinuousDistribution getBestDistribution(Vec v, double KDECutOff, ContinuousDistribution... possibleDistributions)
    {
        if(v.length() == 0)
            throw new ArithmeticException("Can not fit a distribution to an empty set");
        //Thread Safety, clone the possible distributions
        
        ContinuousDistribution[] possDistCopy = new ContinuousDistribution[possibleDistributions.length];
        
        for(int i = 0; i < possibleDistributions.length; i++)
            possDistCopy[i] = possibleDistributions[i].clone();
        
        
        KSTest ksTest = new KSTest(v);
        
        ContinuousDistribution bestDist = null;
        double bestProb = 0;
        
        for(ContinuousDistribution cd : possDistCopy)
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
            if(bestProb >= KDECutOff)
                return bestDist == null ? new Normal(v.mean(), v.standardDeviation()) : bestDist.clone();
            else
                return new KernelDensityEstimator(v);
        }
        catch (RuntimeException ex)//Mostly likely occurs if all values are all zero
        {
            if(v.standardDeviation() == 0)
                return null;
            throw new ArithmeticException("Catistrophic faulure getting a distribution");
        }
    }
}
