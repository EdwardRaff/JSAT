
package jsat.classifiers;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.OnLineStatistics;

/**
 * This class transforms all numerical values into a specified range
 * @author Edward Raff
 */
public class LinearTransform implements DataTransform
{
    double A;
    double B;
    Vec mins;
    /**
     * Represents 
     * 
     *     A - B
     *  -----------
     *   max - min
     */
    Vec mutliplyConstants;

    public LinearTransform()
    {
        this(1, 0);
    }

    
    
    public LinearTransform(double A, double B)
    {
        if(A == B)
            throw new RuntimeException("Values must be different");
        else if(B > A)
        {
            double tmp = A;
            A = B;
            B = tmp;
        }
        this.A = A;
        this.B = B;
    }
    
    
    
    public LinearTransform(ClassificationDataSet cds)
    {
        mins = new DenseVector(cds.getNumNumericalVars());
        Vec maxs = new DenseVector(mins.length());
        mutliplyConstants = new DenseVector(mins.length());
        
        OnLineStatistics[] stats = cds.singleVarStats();
        
        for(int i = 0; i < mins.length(); i++)
        {
            mins.set(i, stats[i].getMin());
            maxs.set(i, stats[i].getMax());
            mutliplyConstants.set(i, A - B);
        }
        
        /**
         * Now we set up the vectors to perform transformations
         * 
         * if x := the variable to be transformed to the range [A, B]
         * Then the transformation we want is
         * 
         *      (A - B)
         * B + ---------  * (-min+x)
         *     max - min
         * 
         * This middle constant will be placed in "maxs"
         * 
         */
        
        maxs.mutableSubtract(mins);
        mutliplyConstants.mutablePairwiseDivide(maxs);
        
    }
    
    public DataPoint transform(DataPoint dp)
    {
        Vec v = dp.getNumericalValues().subtract(mins);
        v.mutablePairwiseMultiply(mutliplyConstants);
        v.mutableAdd(B);
        
        return new DataPoint(v, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
    }
    
}
