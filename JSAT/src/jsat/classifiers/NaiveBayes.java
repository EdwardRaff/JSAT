
package jsat.classifiers;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import jsat.distributions.ContinousDistribution;
import jsat.linear.Vec;
import jsat.utils.FakeExecutor;
import static jsat.distributions.DistributionSearch.*;

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

        
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    public Classifier copy()
    {
        throw new UnsupportedOperationException("Not Yet Implemeneted");
    }

    public boolean supportsWeightedData()
    {
        return false;
    }
    
    /**
     * Runnable task for selecting the right distribution for each task 
     */
    private class DistributionSelectRunable implements Runnable
    {
        int i;
        int j;
        Vec v;
        CountDownLatch countDown;

        public DistributionSelectRunable(int i, int j, Vec v, CountDownLatch countDown)
        {
            this.i = i;
            this.j = j;
            this.v = v;
            this.countDown = countDown;
        }

        
        
        public void run()
        {
            distributions[i][j] = getBestDistribution(v);
            countDown.countDown();
        }
        
    }
    
    private class AprioriCounterRunable implements Runnable
    {
        int i;
        int j;
        List<DataPoint> dataSamples;
        CountDownLatch latch;

        public AprioriCounterRunable(int i, int j, List<DataPoint> dataSamples, CountDownLatch latch)
        {
            this.i = i;
            this.j = j;
            this.dataSamples = dataSamples;
            this.latch = latch;
        }
        
        
        
        public void run()
        {
            for (DataPoint point : dataSamples)//Count each occurance
            {
                apriori[i][j][point.getCategoricalValue(j)]++;
            }

            //Convert the coutns to apriori probablities by dividing the count by the total occurances
            double sum = 0;
            for (int z = 0; z < apriori[i][j].length; z++)
                sum += apriori[i][j][z];
            for (int z = 0; z < apriori[i][j].length; z++)
                apriori[i][j][z] /= sum;
            latch.countDown();
        }
        
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        int nCat = dataSet.getPredicting().getNumOfCategories();
        apriori = new double[nCat][dataSet.getNumCategoricalVars()][];
        distributions = new ContinousDistribution[nCat][dataSet.getNumNumericalVars()] ;
        
        
        int totalWorkers = nCat*(dataSet.getNumNumericalVars() + dataSet.getNumCategoricalVars());
        CountDownLatch latch = new CountDownLatch(totalWorkers);
        
        
        //Go through each classification
        for(int i = 0; i < nCat; i++)
        {
            //Set ditribution for the numerical values
            for(int j = 0; j < dataSet.getNumNumericalVars(); j++)
            {
                Runnable rn = new DistributionSelectRunable(i, j, dataSet.getSampleVariableVector(i, j), latch);
                threadPool.submit(rn);
            }
            
            
            
            List<DataPoint> dataSamples = dataSet.getSamples(i);
            
            //Iterate through the categorical variables
            for(int j = 0; j < dataSet.getNumCategoricalVars(); j++)
            {
                apriori[i][j] = new double[dataSet.getCategories()[j].getNumOfCategories()];
                
                //Laplace correction, put in an extra occurance for each variable
                for(int z = 0; z < apriori[i][j].length; z++)
                    apriori[i][j][z] = 1;
                    
                Runnable rn = new AprioriCounterRunable(i, j, dataSamples, latch);
                threadPool.submit(rn);
            }
        }
        
        
        //Wait for all the threads to finish
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            ex.printStackTrace();
        }
    }
    
}
