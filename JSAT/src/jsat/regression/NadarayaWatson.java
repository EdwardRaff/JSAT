
package jsat.regression;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.DataPoint;
import jsat.classifiers.bayesian.BestClassDistribution;
import jsat.distributions.multivariate.MultivariateKDE;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;

/**
 * The Nadaraya-Watson regressor uses the {@link MultivariateKDE Kernel Density Estimator } to perform regression on a data set. <br>
 * Nadaraya-Watson can also be expressed as a classifier, and equivalent results can be obtained by combining a KDE with {@link BestClassDistribution}. 
 * 
 * @author Edward Raff
 */
public class NadarayaWatson implements Regressor, Parameterized
{

	private static final long serialVersionUID = 8632599345930394763L;
	@ParameterHolder
    private final MultivariateKDE kde;

    public NadarayaWatson(final MultivariateKDE kde)
    {
        this.kde = kde;
    }
    
  @Override
    public double regress(final DataPoint data)
    {
        final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> nearBy = kde.getNearby(data.getNumericalValues());
        if(nearBy.isEmpty()) {
          return 0;///hmmm... what should be retruned in this case?
        }
        double weightSum = 0;
        double sum = 0;
        
        for(final VecPaired<VecPaired<Vec, Integer>, Double> v : nearBy)
        {
            final double weight = v.getPair();
            final double regressionValue = ( (VecPaired<Vec, Double>) v.getVector().getVector()).getPair();
            weightSum += weight;
            sum += weight*regressionValue;
        }
        
        return sum / weightSum;
    }

    @Override
    public void train(final RegressionDataSet dataSet, final ExecutorService threadPool)
    {
        final List<VecPaired<Vec, Double>> vectors = collectVectors(dataSet);
        
        kde.setUsingData(vectors, threadPool);
    }

    private List<VecPaired<Vec, Double>> collectVectors(final RegressionDataSet dataSet)
    {
        final List<VecPaired<Vec, Double>> vectors = new ArrayList<VecPaired<Vec, Double>>(dataSet.getSampleSize());
        for(int i = 0; i < dataSet.getSampleSize(); i++) {
          vectors.add(new VecPaired<Vec, Double>(dataSet.getDataPoint(i).getNumericalValues(), dataSet.getTargetValue(i)));
        }
        return vectors;
    }

    @Override
    public void train(final RegressionDataSet dataSet)
    {
        final List<VecPaired<Vec, Double>> vectors = collectVectors(dataSet);;
        
        kde.setUsingData(vectors);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }

    @Override
    public NadarayaWatson clone()
    {
        return new NadarayaWatson((MultivariateKDE)kde.clone());
    }

    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(final String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
    
}
