
package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;

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
    private DenseVector shiftVector;

    public ZeroMeanTransform(DataSet dataset)
    {
        shiftVector = new DenseVector(dataset.getNumNumericalVars());
        for(int i = 0; i < shiftVector.length(); i++)
            shiftVector.set(i, dataset.getNumericColumn(i).mean());
    }
    
    

    public DataPoint transform(DataPoint dp)
    {
        DataPoint newDP = dp.clone();
        newDP.getNumericalValues().mutableSubtract(shiftVector);
        return newDP;
    }
    
}
