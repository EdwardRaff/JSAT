
package jsat.classifiers.bayesian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.distributions.ContinousDistribution;
import jsat.distributions.DistributionSearch;
import jsat.distributions.Normal;
import jsat.distributions.empirical.KernelDensityEstimator;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.utils.FakeExecutor;
import static jsat.distributions.DistributionSearch.*;
import static java.lang.Math.*;

/**
 *
 * Naive Bayes (Multinomial) classifier. Naive Bayes assumes that all attributes are perfectly independent.
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
    private NumericalHandeling numericalHandling;
    /**
     * Handles how vectors are handled. If true, it is assumed vectors are sparce - and zero values will be ignored when training and classifying.  
     */
    private boolean sparceInput = true;

    /**
     * The default method of handling numeric attributes is {@link NumericalHandeling#NORMAL}. 
     */
    public static final NumericalHandeling defaultHandling = NumericalHandeling.NORMAL;
    
    /**
     * There are multiple ways of handling numerical attributes. These provide the 
     * different ways that NaiveBayes can deal with them. 
     */
    public enum NumericalHandeling 
    {
        /**
         * All numerical attributes are fit to a {@link NORMAL} distribution. 
         */
        NORMAL
        {
            protected ContinousDistribution fit(Vec v)
            {
                return getBestDistribution(v, new Normal(0, 1));
            }
        },
        /**
         * The best fitting {@link ContinousDistribution} is selected by 
         * {@link DistributionSearch#getBestDistribution(jsat.linear.Vec) }
         */
        BEST_FIT
        {

            protected ContinousDistribution fit(Vec v)
            {
                return getBestDistribution(v);
            }
        },
        /**
         * The best fitting {@link ContinousDistribution} is selected by 
         * {@link DistributionSearch#getBestDistribution(jsat.linear.Vec, double) }, 
         * and provides a cut off value to use the {@link KernelDensityEstimator} instead
         */
        BEST_FIT_KDE
        {

            private double cutOff = 0.9;

            /**
             * Sets the cut off value used before fitting an empirical distribution
             * @param c the cut off value, should be between (0, 1).
             */
            public void setCutOff(double c)
            {
                cutOff = c;
            }

            /**
             * Returns the cut off value used before fitting an empirical distribution
             * @return the cut off value used before fitting an empirical distribution
             */
            public double getCtrOff()
            {
                return cutOff;
            }
           
            protected ContinousDistribution fit(Vec v)
            {
                return getBestDistribution(v, cutOff);
            }
        };

        abstract protected ContinousDistribution fit(Vec y);
    }

    public NaiveBayes(NumericalHandeling numericalHandling)
    {
        this.numericalHandling = numericalHandling;
    }

    public NaiveBayes()
    {
        this(defaultHandling);
    }

    /**
     * Sets the method used by this instance for handling numerical attributes. 
     * This has no effect on an already trained classifier, but will change the result if trained again. 
     * 
     * @param numericalHandling the method to use for numerical attributes
     */
    public void setNumericalHandling(NumericalHandeling numericalHandling)
    {
        this.numericalHandling = numericalHandling;
    }

    /**
     * Returns the method used to handle numerical attributes
     * @return the method used to handle numerical attributes 
     */
    public NumericalHandeling getNumericalHandling()
    {
        return numericalHandling;
    }

    /**
     * Returns <tt>true</tt> if the Classifier assumes that data points are sparce. 
     * @return <tt>true</tt> if the Classifier assumes that data points are sparce. 
     * @see #setSparceInput(boolean) 
     */
    public boolean isSparceInput()
    {
        return sparceInput;
    }

    /**
     * Tells the Naive Bayes classifier to 
     * assume the importance of sparseness 
     * in the numerical values. This means 
     * that values of zero will be ignored
     * in computation and classification.<br>
     * This allows faster, more efficient 
     * computation of results if the data 
     * points are indeed sparce. This will
     * also produce different results. 
     * This value should not be changed 
     * after training and before classification.  
     * 
     * @param sparceInput <tt>true</tt> to assume sparseness in the data, <tt>false</tt> to ignore it and assume zeros are meaningful values. 
     * @see #isSparceInput() 
     */
    public void setSparceInput(boolean sparceInput)
    {
        this.sparceInput = sparceInput;
    }
    
    public CategoricalResults classify(DataPoint data)
    {
        
        CategoricalResults results = new CategoricalResults(distributions.length);
        
        Vec numVals = data.getNumericalValues();
        double sum = 0;
        for( int i = 0; i < distributions.length; i++)
        {
            double logProb = 0;
            if(sparceInput)
            {
                Iterator<IndexValue> iter = numVals.getNonZeroIterator();
                while(iter.hasNext())
                {
                    IndexValue indexValue = iter.next();
                    int j = indexValue.getIndex();
                    if(distributions[i][j] == null)
                        continue;
                    double logPDF = distributions[i][j].logPdf(indexValue.getValue());
                    if(Double.isInfinite(logPDF))//Avoid propigation -infinty when the probability is zero
                        logProb += log(1e-16);//
                    else
                        logProb += logPDF;
                }
            }
            else
            {
                for(int j = 0; j < distributions[i].length; j++)
                {
                    if(distributions[i][j] == null)
                        continue;
                    double logPDF = distributions[i][j].logPdf(numVals.get(j));
                    if(Double.isInfinite(logPDF))//Avoid propigation -infinty when the probability is zero
                        logProb += log(1e-16);//
                    else
                        logProb += logPDF;
                }
            }
            
            //the i goes up to the number of categories, same for aprioror
            for(int j = 0; j < apriori[i].length; j++)
            {
                double p = apriori[i][j][data.getCategoricalValue(j)];
                logProb += log(p);
            }
            
            double prob = exp(logProb);
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

    @Override
    public Classifier clone()
    {
        NaiveBayes newBayes = new NaiveBayes();
        
        newBayes.apriori = new double[this.apriori.length][][];
        for(int i = 0; i < this.apriori.length; i++)
        {
            newBayes.apriori[i] = new double[this.apriori[i].length][];
            for(int j = 0; this.apriori[i].length > 0 && j < this.apriori[i][j].length; j++)
                newBayes.apriori[i][j] = Arrays.copyOf(this.apriori[i][j], this.apriori[i][j].length);
        }
        
        newBayes.distributions = new ContinousDistribution[this.distributions.length][];
        for(int i = 0; i < this.distributions.length; i++)
        {
            newBayes.distributions[i] = new ContinousDistribution[this.distributions[i].length];
            for(int j = 0; j < this.distributions[i].length; j++)
                newBayes.distributions[i][j] = this.distributions[i][j].clone();
        }
        
        return newBayes;
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
            try
            {
                distributions[i][j] = numericalHandling.fit(v);
            }
            catch (ArithmeticException e)
            {
                distributions[i][j] = null;
            }
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

    private Vec getSampleVariableVector(ClassificationDataSet dataSet, int category, int j)
    {
        Vec vals =  dataSet.getSampleVariableVector(category, j);
        
        if(sparceInput)
        {
            List<Double> nonZeroVals = new ArrayList<Double>();
            for(int i = 0; i < vals.length(); i++)
                if(vals.get(i) != 0)
                    nonZeroVals.add(vals.get(i));
            vals = new DenseVector(nonZeroVals);
        }
        
        return vals;
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
                Runnable rn = new DistributionSelectRunable(i, j, getSampleVariableVector(dataSet, i, j), latch);
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
