
package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
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
        stndDevs = d.getColumnMeanVariance()[1];
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
