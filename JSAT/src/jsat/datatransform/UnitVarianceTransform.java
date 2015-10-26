
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
	private final Vec stndDevs;
    
    public UnitVarianceTransform(final DataSet d)
    {
        stndDevs = d.getColumnMeanVariance()[1];
    }
    
    /**
     * Copy constructor 
     * @param other the transform to make a copy of
     */
    private UnitVarianceTransform(final UnitVarianceTransform other)
    {
        this.stndDevs = other.stndDevs.clone();
    }

    @Override
    public DataPoint transform(final DataPoint dp)
    {
        final DataPoint newDp = dp.clone();
        mutableTransform(newDp);
        return newDp;
    }
    
    @Override
    public void mutableTransform(final DataPoint dp)
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

    /**
     * Factory for producing new {@link UnitVarianceTransform} transforms. 
     */
    static public class UnitVarianceTransformFactory implements DataTransformFactory
    {
        @Override
        public DataTransform getTransform(final DataSet dataset)
        {
            return new UnitVarianceTransform(dataset);
        }

        @Override
        public UnitVarianceTransformFactory clone()
        {
            return new UnitVarianceTransformFactory();
        }
    }
}
