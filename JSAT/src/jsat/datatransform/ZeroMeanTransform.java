
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
public class ZeroMeanTransform implements DataTransform
{
    /**
     * Shift vector stores the mean value of each variable in the original data set. 
     */
    private Vec shiftVector;

    public ZeroMeanTransform(DataSet dataset)
    {
        shiftVector = new DenseVector(dataset.getNumNumericalVars());
        for(int i = 0; i < shiftVector.length(); i++)
            shiftVector.set(i, dataset.getNumericColumn(i).mean());
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
        newDP.getNumericalValues().mutableSubtract(shiftVector);
        return newDP;
    }

    @Override
    public DataTransform clone()
    {
        return new ZeroMeanTransform(this);
    }
    
    static public class ZeroMeanTransformFactory implements DataTransformFactory
    {

        public ZeroMeanTransformFactory()
        {
        }
        
        @Override
        public DataTransform getTransform(DataSet dataset)
        {
            return new ZeroMeanTransform(dataset);
        }
        
    }
}
