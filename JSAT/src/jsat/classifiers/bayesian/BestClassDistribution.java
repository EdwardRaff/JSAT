
package jsat.classifiers.bayesian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.distributions.multivariate.MultivariateDistribution;
import jsat.exceptions.FailedToFitException;

/**
 * BestClassDistribution is a generic class for performing classification by fitting a {@link MultivariateDistribution} to each class. The distribution 
 * is supplied by the user, and each class if fit to the same type of distribution. Classification is then performed by returning
 * the class of the most likely distribution given the data point. 
 * 
 * @author Edward Raff
 */
public class BestClassDistribution implements Classifier
{
    private MultivariateDistribution baseDist;
    private List<MultivariateDistribution> dists;
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

    public BestClassDistribution(MultivariateDistribution baseDist)
    {
        this(baseDist, USE_PRIORS);
    }
    
    public BestClassDistribution(MultivariateDistribution baseDist, boolean usePriors)
    {
        this.baseDist = baseDist;
        this.usePriors = usePriors;
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
        CategoricalResults cr = new CategoricalResults(dists.size());
        
        for(int i = 0; i < dists.size(); i++)
        {
            if(dists.get(i) == null)
                continue;
            double  p = dists.get(i).pdf(data.getNumericalValues());
            if(usePriors)
                p *= priors[i];
            cr.setProb(i, p);
        }
        cr.normalize();
        return cr;
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        try
        {
            this.dists = new ArrayList<MultivariateDistribution>();
            this.priors = dataSet.getPriors();
            List<Future<MultivariateDistribution>> newDists = new ArrayList<Future<MultivariateDistribution>>();
            final MultivariateDistribution sourceDist = baseDist;
            for (int i = 0; i < dataSet.getPredicting().getNumOfCategories(); i++)//Calculate the Multivariate normal for each category
            {
                final List<DataPoint> class_i = dataSet.getSamples(i);
                Future<MultivariateDistribution> tmp = threadPool.submit(new Callable<MultivariateDistribution>()
                {

                    public MultivariateDistribution call() throws Exception
                    {
                        if (class_i.isEmpty())//Nowthing we can do 
                            return null;
                        MultivariateDistribution dist = sourceDist.clone();
                        dist.setUsingDataList(class_i);
                        return dist;
                    }
                });

                newDists.add(tmp);
            }
            for (Future<MultivariateDistribution> future : newDists)
                this.dists.add(future.get());
        }
        catch (Exception ex)
        {
            Logger.getLogger(MultivariateNormals.class.getName()).log(Level.SEVERE, null, ex);
            throw new FailedToFitException(ex);
        }
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        priors = dataSet.getPriors();
        dists = new ArrayList<MultivariateDistribution>(dataSet.getClassSize());
        
        for(int i = 0; i < dataSet.getClassSize(); i++)
        {
            MultivariateDistribution dist = baseDist.clone();
            List<DataPoint> samp = dataSet.getSamples(i);
            if(samp.isEmpty())
            {
                dists.add(null);
                continue;
            }
            dist.setUsingDataList(samp);
            dists.add(dist);
        }
    }

    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public Classifier clone()
    {
        BestClassDistribution clone = new BestClassDistribution(baseDist.clone(), usePriors);
        
        if(this.priors != null)
            clone.priors = Arrays.copyOf(this.priors, this.priors.length);
        if(this.dists  != null)
        {
            clone.dists = new ArrayList<MultivariateDistribution>(this.dists.size());
            for(MultivariateDistribution md : this.dists)
                clone.dists.add(md.clone());
        }
        
        return clone;
    }
    
}
