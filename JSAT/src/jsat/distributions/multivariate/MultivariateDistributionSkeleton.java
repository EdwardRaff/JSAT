
package jsat.distributions.multivariate;

import jsat.DataSet;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * Common class for implementing a multivariate distribution. A number of methods are pre implemented, building off of the implementation of the remaining methods. 
 * @author Edward Raff
 */
public abstract class MultivariateDistributionSkeleton implements MultivariateDistribution
{
    public double logPdf(double... x)
    {
        return logPdf(DenseVector.toDenseVec(x));
    }
    
    public double logPdf(Vec x)
    {
        double logPDF = Math.log(pdf(x));
        if(Double.isInfinite(logPDF) && logPDF < 0)//log(0) == -Infinty
            return -Double.MAX_VALUE;
        return logPDF;
    }
    
    public double pdf(double... x)
    {
        return pdf(DenseVector.toDenseVec(x));
    }
    
    public boolean setUsingData(DataSet dataSet)
    {
        return setUsingDataList(dataSet.getDataPoints());
    }

    @Override
    abstract public MultivariateDistribution clone();
}
