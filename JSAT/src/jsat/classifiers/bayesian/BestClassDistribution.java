
package jsat.classifiers.bayesian;

import java.util.*;
import java.util.stream.Collectors;

import jsat.classifiers.*;
import jsat.distributions.multivariate.MultivariateDistribution;
import jsat.parameters.*;
import jsat.utils.concurrent.ParallelUtils;

/**
 * BestClassDistribution is a generic class for performing classification by fitting a {@link MultivariateDistribution} to each class. The distribution 
 * is supplied by the user, and each class if fit to the same type of distribution. Classification is then performed by returning
 * the class of the most likely distribution given the data point. 
 * 
 * @author Edward Raff
 */
public class BestClassDistribution implements Classifier, Parameterized
{

    private static final long serialVersionUID = -1746145372146154228L;
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
     * The default value for whether or not to use the prior probability of a 
     * class when making classification decisions is {@value #USE_PRIORS}. 
     */
    public static final boolean USE_PRIORS = true;
    
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
     * Copy constructor
     * @param toCopy the object to copy
     */
    public BestClassDistribution(BestClassDistribution toCopy)
    {
        if(toCopy.priors != null)
            this.priors = Arrays.copyOf(toCopy.priors, toCopy.priors.length);
        this.baseDist = toCopy.baseDist.clone();
        if(toCopy.dists  != null)
        {
            this.dists = new ArrayList<>(toCopy.dists.size());
            for(MultivariateDistribution md : toCopy.dists)
                this.dists.add(md == null? null : md.clone());
        }
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
     * @return {@code true} if the prior probabilities are being used, 
     * {@code false} if not. 
     */
    public boolean isUsePriors()
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
            double  p = 0 ;
            try
            {
                p = dists.get(i).pdf(data.getNumericalValues());
            }
            catch(ArithmeticException ex)
            {
                
            }
            if(usePriors)
                p *= priors[i];
            cr.setProb(i, p);
        }
        cr.normalize();
        return cr;
    }

    @Override
    public void train(ClassificationDataSet dataSet, boolean parallel)
    {
        priors = dataSet.getPriors();
        dists = ParallelUtils.range(dataSet.getClassSize(), parallel).mapToObj((int i) -> 
        {
            MultivariateDistribution dist = baseDist.clone();
            List<DataPoint> samp = dataSet.getSamples(i);
            if(samp.isEmpty())
            {
                return (MultivariateDistribution) null;
            }
            dist.setUsingDataList(samp);
            return dist;
        }).collect(Collectors.toList());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public BestClassDistribution clone()
    {
        return new BestClassDistribution(this);
    }
    
}
