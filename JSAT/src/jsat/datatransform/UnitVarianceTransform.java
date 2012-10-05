
package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * Creates a transform to alter data points so that each attribute has a 
 * standard deviation of 1, which means a variance of 1. 
 * 
 * @author Edward Raff
 */
public class UnitVarianceTransform implements DataTransform
{
    private Vec stndDevs;
    
    public UnitVarianceTransform(DataSet d)
    {
        stndDevs = new DenseVector(d.getNumNumericalVars());
        
        for(int i = 0; i < d.getNumNumericalVars(); i++)
            stndDevs.set(i, d.getNumericColumn(i).standardDeviation());
        
    }
    
    /**
     * Copy constructor 
     * @param other the transform to make a copy of
     */
    private UnitVarianceTransform(UnitVarianceTransform other)
    {
        this.stndDevs = other.stndDevs.clone();
    }

    @Override
    public DataPoint transform(DataPoint dp)
    {
        DataPoint newDp = dp.clone();
        newDp.getNumericalValues().mutablePairwiseDivide(stndDevs);
        return newDp;
    }

    @Override
    public DataTransform clone()
    {
        return new UnitVarianceTransform(this);
    }
    
    static public class UnitVarianceTransformFactory implements DataTransformFactory
    {
        @Override
        public DataTransform getTransform(DataSet dataset)
        {
            return new UnitVarianceTransform(dataset);
        }
    }
}
