
package jsat.classifiers;

import java.util.List;
import jsat.distributions.ChiSquared;
import jsat.distributions.ContinousDistribution;
import jsat.distributions.Exponential;
import jsat.distributions.FisherSendor;
import jsat.distributions.Gamma;
import jsat.distributions.LogNormal;
import jsat.distributions.Normal;
import jsat.distributions.Uniform;
import jsat.distributions.Weibull;
import jsat.linear.Vec;
import jsat.testing.goodnessoffit.KSTest;

/**
 *
 * Naive Bayes (Multinomial) classifier. 
 * 
 * @author Edward Raff
 */
public class NaiveBayes implements Classifier
{
    /**
     * 
     */
    private double[][][] apriori;
    private ContinousDistribution[][] distributions; 

    public NaiveBayes()
    {
    }
    
    

    public CategoricalResults classify(DataPoint data)
    {
        
        CategoricalResults results = new CategoricalResults(distributions.length);
        
        
        double sum = 0;
        for( int i = 0; i < distributions.length; i++)
        {
            double prob = 1;
            for(int j = 0; j < distributions[i].length; j++)
            {
                double pdf = distributions[i][j].pdf(data.getNumericalValues().get(j));
                prob *= pdf;
            }
            
            //the i goes up to the number of categories, same for aprioror
            for(int j = 0; j < apriori[i].length; j++)
            {
                double p = apriori[i][j][data.getCategoricalValue(j)];
                prob *= p;
            }
            
            results.setProb(i, prob);
            
            sum += prob;
        }
        
        
        if(sum != 0)
            results.divideConst(sum);
        
        return results;
    }

    
    ContinousDistribution[] possibleDistributions = new ContinousDistribution[] 
    { 
        new Normal(), 
        new LogNormal(), new Exponential(),
        new Gamma(2, 1), new FisherSendor(10, 10), new Weibull(2, 1), 
        new Uniform(0, 1)
    };
    
    private ContinousDistribution getBestDistribution(Vec v)
    {
        KSTest ksTest = new KSTest(v);
        
        ContinousDistribution bestDist = null;
        double bestProb = 0;
        
        for(ContinousDistribution cd : possibleDistributions)
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
        return bestDist == null ? new Normal(v.mean(), v.standardDeviation()) : bestDist.copy();
    }
    
    
    
    public void trainC(ClassificationDataSet dataSet)
    {
        int nCat = dataSet.getPredicting().getNumOfCategories();
        apriori = new double[nCat][dataSet.getNumCategoricalVars()][];
        distributions = new ContinousDistribution[nCat][dataSet.getNumNumericalVars()] ;
        
        //Go through each classification
        for(int i = 0; i < nCat; i++)
        {
            //Set ditribution for the numerical values
            for(int j = 0; j < dataSet.getNumNumericalVars(); j++)
                distributions[i][j] = getBestDistribution(dataSet.getSampleVariableVector(i, j));
            
            
            
            List<DataPoint> dataSamples = dataSet.getSamples(i);
            
            //Iterate through the categorical variables
            for(int j = 0; j < dataSet.getNumCategoricalVars(); j++)
            {
                apriori[i][j] = new double[dataSet.getCategories()[j].getNumOfCategories()];
                    
                
                for(DataPoint point : dataSamples)//Count each occurance
                {
                    apriori[i][j][point.getCategoricalValue(j)]++;
                }
                
                //Convert the coutns to apriori probablities by dividing the count by the total occurances
                double sum = 0;
                for(int z = 0; z < apriori[i][j].length; z++)
                    sum += apriori[i][j][z];
                for(int z = 0; z < apriori[i][j].length; z++)
                    apriori[i][j][z] /= sum;
            }
            
            
            
            
            
        }
        
    }
    
}
