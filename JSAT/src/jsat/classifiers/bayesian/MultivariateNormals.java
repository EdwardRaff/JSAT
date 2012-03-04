package jsat.classifiers.bayesian;

import jsat.distributions.multivariate.NormalM;

/**
 * This classifier can be seen as an extension of {@link NaiveBayes}. Instead of treating the variables as independent,
 * each class uses all of its variables to fit a {@link NormalM Multivariate Normal} distribution. As such, it can only 
 * handle numerical attributes. However, if the classes are normally distributed, it will produce optimal classification
 * results. The less normal the true distributions are, the less accurate the classifier will be.
 * 
 * @author Edward Raff
 */
public class MultivariateNormals extends BestClassDistribution
{
    public MultivariateNormals(boolean usePriors)
    {
        super(new NormalM(), usePriors);
    }

    /**
     * Creates a new class for classification by feating each class to a {@link NormalM Multivariate Normal Distribution}. 
     */
    public MultivariateNormals()
    {
        super(new NormalM());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }
}
