package jsat.classifiers.bayesian;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.*;
import jsat.exceptions.FailedToFitException;

/**
 * Averaged One-Dependence Estimators (AODE) is an extension of Naive Bayes that
 * attempts to be more accurate by reducing the independence assumption. For 
 * <i>n</i> data points with <i>d</i> categorical features, <i>d</i> 
 * {@link ODE} classifier are created, each with a dependence on a different 
 * attribute. The results of these classifiers is averaged to produce a final
 * result.  The construction time is <i>O(n d<sup>2</sup>)</i>. Because of this 
 * extra dependence requirement, the implementation only allows for categorical 
 * features. <br>
 * <br>
 * See: Webb, G., &amp; Boughton, J. (2005). <i>Not so naive bayes: Aggregating 
 * one-dependence estimators</i>. Machine Learning, 1–24. Retrieved from 
 * <a href="http://www.springerlink.com/index/U8W306673M1P866K.pdf">here</a>
 * 
 * @author Edward Raff
 */
public class AODE extends BaseUpdateableClassifier
{

	private static final long serialVersionUID = 8386506277969540732L;
	protected CategoricalData predicting;
    protected ODE[] odes;

    /**
     * The minimum value to use a probability
     */
    private double m = 20;
    
    /**
     * Creates a new AODE classifier. 
     */
    public AODE()
    {
    }
    
    /**
     * Creates a copy of an AODE classifier
     * @param toClone the classifier to clone
     */
    protected AODE(AODE toClone)
    {
        if(toClone.odes != null)
        {
            this.odes = new ODE[toClone.odes.length];
            for(int i = 0; i < this.odes.length; i++)
                this.odes[i] = toClone.odes[i].clone();
            this.predicting = toClone.predicting.clone();
        }
        this.m = toClone.m;
    }

    @Override
    public AODE clone()
    {
        return new AODE(this);
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if(categoricalAttributes.length < 1)
            throw new FailedToFitException("At least 2 categorical varaibles are needed for AODE");
        this.predicting = predicting;
        odes = new ODE[categoricalAttributes.length];
        
        for(int i = 0; i < odes.length; i++)
        {
            odes[i] = new ODE(i);
            odes[i].setUp(categoricalAttributes, numericAttributes, predicting);
        }
    }
    
    @Override
    public void trainC(final ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        setUp(dataSet.getCategories(), dataSet.getNumNumericalVars(), 
                dataSet.getPredicting());
        
        final CountDownLatch latch = new CountDownLatch(odes.length);

        for (int i = 0; i < odes.length; i++)
        {
            final ODE ode = odes[i];
            threadPool.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    for (int i = 0; i < dataSet.getSampleSize(); i++)
                        ode.update(dataSet.getDataPoint(i), dataSet.getDataPointCategory(i));
                    latch.countDown();
                }
            });
        }
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            trainC(dataSet);
        }
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        for(ODE ode : odes)
            ode.update(dataPoint, targetClass);
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());
        
        int[] catVals = data.getCategoricalValues();
        for(int c = 0; c < cr.size(); c++)
        {
            double prob = 0.0;
            for (ODE ode : odes)
                if (ode.priors[c][catVals[ode.dependent]] < m)
                    continue;
                else
                    prob += Math.exp(ode.getLogPrb(catVals, c));
            cr.setProb(c, prob);
        }
        cr.normalize();
        
        return cr;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }

    /**
     * Sets the minimum prior observation value needed for an attribute 
     * combination to have enough support to be included in the final estimate. 
     * 
     * @param m the minimum needed score
     */
    public void setM(double m)
    {
        if(m < 0 || Double.isInfinite(m) || Double.isNaN(m))
            throw new ArithmeticException("The minimum count must be a non negative number");
        this.m = m;
    }

    /**
     * Returns the minimum needed score
     * @return the minimum needed score
     */
    public double getM()
    {
        return m;
    }
    
    
    
}
