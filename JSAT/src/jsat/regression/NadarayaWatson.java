
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
    private MultivariateKDE kde;

    public NadarayaWatson(MultivariateKDE kde)
    {
        this.kde = kde;
    }
    
    public double regress(DataPoint data)
    {
        List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> nearBy = kde.getNearby(data.getNumericalValues());
        if(nearBy.isEmpty())
            return 0;///hmmm... what should be retruned in this case?
        double weightSum = 0;
        double sum = 0;
        
        for(VecPaired<VecPaired<Vec, Integer>, Double> v : nearBy)
        {
            double weight = v.getPair();
            double regressionValue = ( (VecPaired<Vec, Double>) v.getVector().getVector()).getPair();
            weightSum += weight;
            sum += weight*regressionValue;
        }
        
        return sum / weightSum;
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        List<VecPaired<Vec, Double>> vectors = collectVectors(dataSet);
        
        kde.setUsingData(vectors, threadPool);
    }

    private List<VecPaired<Vec, Double>> collectVectors(RegressionDataSet dataSet)
    {
        List<VecPaired<Vec, Double>> vectors = new ArrayList<VecPaired<Vec, Double>>(dataSet.getSampleSize());
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            vectors.add(new VecPaired<Vec, Double>(dataSet.getDataPoint(i).getNumericalValues(), dataSet.getTargetValue(i)));
        return vectors;
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        List<VecPaired<Vec, Double>> vectors = collectVectors(dataSet);;
        
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
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
    
}
