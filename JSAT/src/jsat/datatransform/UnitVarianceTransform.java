
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
public class UnitVarianceTransform implements InPlaceTransform
{

    private static final long serialVersionUID = 3645532503475641917L;
    private Vec stndDevs;

    /**
     * Creates a new object for transforming datasets
     */
    public UnitVarianceTransform()
    {
    }
    
    /**
     * Creates a new object for making datasets unit variance fit to the given
     * dataset
     *
     * @param d the dataset to learn this transform from
     */
    public UnitVarianceTransform(DataSet d)
    {
        fit(d);
    }

    @Override
    public void fit(DataSet d)
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
        mutableTransform(newDp);
        return newDp;
    }
    
    @Override
    public void mutableTransform(DataPoint dp)
    {
        dp.getNumericalValues().mutablePairwiseDivide(stndDevs);
    }

    @Override
    public boolean mutatesNominal()
    {
        return false;
    }
    
    @Override
    public DataTransform clone()
    {
        return new UnitVarianceTransform(this);
    }

}
