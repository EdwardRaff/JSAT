package jsat.classifiers.bayesian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.FailedToFitException;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.distributions.multivariate.NormalM;

/**
 * This classifier can be seen as an extension of {@link NaiveBayes}. Instead of treating the variables as independent,
 * each class uses all of its variables to fit a {@link NormalM Multivariate Normal} distribution. As such, it can only 
 * handle numerical attributes. However, if the classes are normally distributed, it will produce optimal classification
 * results. The less normal the true distributions are, the less accurate the classifier will be.
 * 
 * @author Edward Raff
 */
public class MultivariateNormals implements Classifier
{
    /**
     * List of distributions, index corresponds to the class. null is permissible,
     * and indicates that a distribution is never likely (IE: no data)
     */
    private List<NormalM> distributions;
    /**
     * The prior probabilities of each class
     */
    private double priors[];
    
    /**
     * Controls whether or no the prior probability will be used when computing probabilities
     */
    private boolean usePriors;
    /**
     * The default value for whether or not to use the prior probability of a class when making classification decisions is {@value #USE_PRIORS}. 
     */
    public static final boolean USE_PRIORS = false;

    /**
     * Creates a new class for classification by feating each class to a {@link NormalM Multivariate Normal Distribution}. 
     * 
     * @param usePriors controls whether or not the prior probabilities will be taken into account when performing classification
     */
    public MultivariateNormals(boolean usePriors)
    {
        this.usePriors = usePriors;
    }

    /**
     * Creates a new class for classification by feating each class to a {@link NormalM Multivariate Normal Distribution}. 
     */
    public MultivariateNormals()
    {
        this(USE_PRIORS);
    }

    /**
     * Controls whether or not the priors will be used for classification. This value can be 
     * changed at any time, before or after training has occurred. 
     * 
     * @param usePriors <tt>true</tt> to use the prior probabilities for each class, <tt>false</tt> to ignore them. 
     */
    public void setUsePriors(boolean usePriors)
    {
        this.usePriors = usePriors;
    }

    /**
     * Returns whether or not this object uses the prior probabilities for classification. 
     * @return 
     */
    public boolean useusPriors()
    {
        return usePriors;
    }
    
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(distributions.size());
        
        for(int i = 0; i < distributions.size(); i++)
            if(distributions.get(i) != null)
                cr.setProb(i, distributions.get(i).pdf(data.getNumericalValues()) * (usePriors ? priors[i] : 1.0) );
        
        return cr;
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        try
        {
            this.distributions = new ArrayList<NormalM>();
            this.priors = new double[dataSet.getPredicting().getNumOfCategories()];
            List<Future<NormalM>> newNormals = new ArrayList<Future<NormalM>>();
            for(int i = 0; i < dataSet.getPredicting().getNumOfCategories(); i++)//Calculate the Multivariate normal for each category
            {
                final int ii = i;
                final List<DataPoint> class_i = dataSet.getSamples(i);
                Future<NormalM> tmp = threadPool.submit(new Callable<NormalM>() {

                    public NormalM call() throws Exception
                    {
                        if(class_i.isEmpty())//Nowthing we can do 
                            return null;
                        //Add the weights to priors and normalized later
                        for(DataPoint dp : class_i)
                            priors[ii] += dp.getWeight();
                        NormalM normalM = new NormalM();
                        normalM.setUsingDataList(class_i);
                        return normalM;
                    }
                });
                
                newNormals.add(tmp);
            }
            for(Future<NormalM> future : newNormals)
                this.distributions.add(future.get());
            double sumOfWeights = 0.0;
            for(double prior : priors)
                sumOfWeights += prior;
            for(int i = 0; i < priors.length; i++)
                priors[i] /= sumOfWeights;
        }


        catch(ArithmeticException ex)
        {
            Logger.getLogger(MultivariateNormals.class.getName()).log(Level.SEVERE, null, ex);
            throw new FailedToFitException(ex);
        }        catch (InterruptedException ex)
        {
            Logger.getLogger(MultivariateNormals.class.getName()).log(Level.SEVERE, null, ex);
            throw new FailedToFitException(ex);
        }        catch (ExecutionException ex)
        {
            Logger.getLogger(MultivariateNormals.class.getName()).log(Level.SEVERE, null, ex);
            throw new FailedToFitException(ex);
        }
        
        
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        try
        {
            this.priors = new double[dataSet.getPredicting().getNumOfCategories()];
            this.distributions = new ArrayList<NormalM>(priors.length);
            double sumOfWeights = 0.0;
            for (int i = 0; i < priors.length; i++)//Calculate the Multivariate normal for each category
            {
                
                final List<DataPoint> class_i = dataSet.getSamples(i);
                
                if (class_i.isEmpty())///Nothing we can do. Call it null and skip it
                {
                    this.distributions.add(null);
                    continue;
                }
                for(DataPoint dp : class_i)
                    priors[i] += dp.getWeight();
                sumOfWeights += priors[i];
                NormalM normalM = new NormalM();
                if(normalM.setUsingDataList(class_i))
                    this.distributions.add(normalM);
                else
                    this.distributions.add(null);//Failed to set
            }
            
            for(int i = 0; i < priors.length; i++)
                priors[i] /= sumOfWeights;
        }
        catch (Exception e)
        {
            throw new FailedToFitException(e);
        }
    }

    public boolean supportsWeightedData()
    {
        return true;
    }

    @Override
    public Classifier clone()
    {
        MultivariateNormals clone = new MultivariateNormals();
        if (this.distributions != null)
        {
            clone.distributions = new ArrayList<NormalM>(this.distributions.size());
            for (NormalM dist : this.distributions)
                if (dist == null)
                    clone.distributions.add(null);
                else
                    clone.distributions.add(dist.clone());
        }
        if(this.priors != null)
            clone.priors = Arrays.copyOf(this.priors, this.priors.length);
        return clone;
    }
}
