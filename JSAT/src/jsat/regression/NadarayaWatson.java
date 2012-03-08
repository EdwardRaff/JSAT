
package jsat.regression;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.DataPoint;
import jsat.classifiers.bayesian.BestClassDistribution;
import jsat.distributions.multivariate.MultivariateKDE;
import jsat.linear.Vec;
import jsat.linear.VecPaired;

/**
 * The Nadaraya-Watson regressor uses the {@link MultivariateKDE Kernel Density Estimator } to perform regression on a data set. <br>
 * Nadaraya-Watson can also be expressed as a classifier, and equivalent results can be obtained by combining a KDE with {@link BestClassDistribution}. 
 * 
 * @author Edward Raff
 */
public class NadarayaWatson implements Regressor
{
    private MultivariateKDE kde;

    public NadarayaWatson(MultivariateKDE kde)
    {
        this.kde = kde;
    }
    
    public double regress(DataPoint data)
    {
        List<VecPaired<Double, VecPaired<Integer, Vec>>> nearBy = kde.getNearby(data.getNumericalValues());
        if(nearBy.isEmpty())
            return 0;///hmmm... what should be retruned in this case?
        double weightSum = 0;
        double sum = 0;
        
        for(VecPaired<Double, VecPaired<Integer, Vec>> v : nearBy)
        {
            double weight = v.getPair();
            double regressionValue = ( (VecPaired<Double, Vec>) v.getVector().getVector()).getPair();
            weightSum += weight;
            sum += weight*regressionValue;
        }
        
        return sum / weightSum;
    }

    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        List<VecPaired<Double, Vec>> vectors = collectVectors(dataSet);
        
        kde.setUsingData(vectors, threadPool);
    }

    private List<VecPaired<Double, Vec>> collectVectors(RegressionDataSet dataSet)
    {
        List<VecPaired<Double, Vec>> vectors = new ArrayList<VecPaired<Double, Vec>>(dataSet.getSampleSize());
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            vectors.add(new VecPaired<Double, Vec>(dataSet.getDataPoint(i).getNumericalValues(), dataSet.getTargetValue(i)));
        return vectors;
    }

    public void train(RegressionDataSet dataSet)
    {
        List<VecPaired<Double, Vec>> vectors = collectVectors(dataSet);;
        
        kde.setUsingData(vectors);
    }

    public boolean supportsWeightedData()
    {
        return true;
    }

    @Override
    public NadarayaWatson clone()
    {
        return new NadarayaWatson((MultivariateKDE)kde.clone());
    }
    
}
