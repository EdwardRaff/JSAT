
package jsat.classifiers.bayesian;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.*;
import jsat.distributions.multivariate.MultivariateDistribution;
import jsat.exceptions.FailedToFitException;
import jsat.parameters.*;

/**
 * BestClassDistribution is a generic class for performing classification by fitting a {@link MultivariateDistribution} to each class. The distribution 
 * is supplied by the user, and each class if fit to the same type of distribution. Classification is then performed by returning
 * the class of the most likely distribution given the data point. 
 * 
 * @author Edward Raff
 */
public class BestClassDistribution implements Classifier, Parameterized
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
    
    private final Parameter usePriorsParam = new BooleanParameter() {

        @Override
        public boolean getValue()
        {
            return useusPriors();
        }

        @Override
        public boolean setValue(boolean val)
        {
            setUsePriors(val);
            return true;
        }

        @Override
        public String getASCIIName()
        {
            return "Use Priors";
        }
    };

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
    

    @Override
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

    @Override
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

    @Override
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

    @Override
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

    @Override
    public List<Parameter> getParameters()
    {
        List<Parameter> parameters = new ArrayList<Parameter>();
        
        parameters.add(usePriorsParam);
        if(baseDist instanceof Parameterized)
            parameters.addAll(((Parameterized)baseDist).getParameters());
        return parameters;
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        if(paramName.equals(usePriorsParam.getASCIIName()))
            return usePriorsParam;
        else if(baseDist instanceof Parameterized)
            return ((Parameterized)baseDist).getParameter(paramName);
        else
            return null;
    }
    
}
