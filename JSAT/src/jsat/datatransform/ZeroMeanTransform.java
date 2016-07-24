
package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * A transformation to shift all numeric variables so that their mean is zero
 * 
 * @author Edward Raff
 */
public class ZeroMeanTransform implements InPlaceInvertibleTransform
{

    private static final long serialVersionUID = -7411115746918116163L;
    /**
     * Shift vector stores the mean value of each variable in the original data set. 
     */
    private Vec shiftVector;

    /**
     * Creates a new object for transforming datapoints by centering the data
     */
    public ZeroMeanTransform()
    {
    }
    
    /**
     * Creates a new object for transforming datapoints by centering the data
     * @param dataset the data to learn this transform from
     */
    public ZeroMeanTransform(DataSet dataset)
    {
        fit(dataset);
    }

    @Override
    public void fit(DataSet dataset)
    {
        shiftVector = new DenseVector(dataset.getNumNumericalVars());
        shiftVector = dataset.getColumnMeanVariance()[0];
    }
    
    /**
     * Copy constructor
     * @param other the transform to make a copy of
     */
    private ZeroMeanTransform(ZeroMeanTransform other)
    {
        this.shiftVector = other.shiftVector.clone();
    }

    @Override
    public DataPoint transform(DataPoint dp)
    {
        DataPoint newDP = dp.clone();
        mutableTransform(newDP);
        return newDP;
    }
    
    
    @Override
    public void mutableInverse(DataPoint dp)
    {
        dp.getNumericalValues().mutableAdd(shiftVector);
    }

    @Override
    public DataPoint inverse(DataPoint dp)
    {
        DataPoint newDP = dp.clone();
        mutableInverse(dp);
        return newDP;
    }
    
    @Override
    public void mutableTransform(DataPoint dp)
    {
        dp.getNumericalValues().mutableSubtract(shiftVector);
    }

    @Override
    public boolean mutatesNominal()
    {
        return false;
    }
    
    @Override
    public ZeroMeanTransform clone()
    {
        return new ZeroMeanTransform(this);
    }
}
