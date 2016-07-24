
package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.OnLineStatistics;

/**
 * This class transforms all numerical values into a specified range by a linear
 * scaling of all the data point values. 
 * 
 * @author Edward Raff
 */
public class LinearTransform implements InPlaceInvertibleTransform
{

    private static final long serialVersionUID = 5580283565080452022L;
    /**
     * The max value
     */
    private double A;
    /**
     * The min value
     */
    private double B;
    
    /**
     * The minimum observed value for each attribute 
     */
    private Vec mins;
    
    /**
     * Represents 
     * 
     *     A - B
     *  -----------
     *   max - min
     */
    private Vec mutliplyConstants;
    
    /**
     * Creates a new Linear Transformation that will scale 
     * values to the [0, 1] range. 
     * 
     */
    public LinearTransform()
    {
        this(1, 0);
    }

    /**
     * Creates a new Linear Transformation for the input data set so that all
     * values are in the [0, 1] range. 
     * 
     * @param dataSet the data set to learn the transform from
     */
    public LinearTransform(DataSet dataSet)
    {
        this(dataSet, 1, 0);
    }
    
    /**
     * Creates a new Linear Transformation. 
     * 
     * @param dataSet the data set to learn the transform from
     * @param A the maximum value for the transformed data set
     * @param B the minimum value for the transformed data set
     */
    public LinearTransform(double A, double B)
    {
        setRange(A, B);
    }

    /**
     * Creates a new Linear Transformation for the input data set. 
     * 
     * @param dataSet the data set to learn the transform from
     * @param A the maximum value for the transformed data set
     * @param B the minimum value for the transformed data set
     */
    public LinearTransform(DataSet dataSet, double A, double B)
    {
        this(A, B);
        fit(dataSet);
    }

    /**
     * Sets the min and max value to scale the data to. If given in the wrong order, this method will swap them
     * @param A the maximum value for the transformed data set
     * @param B the minimum value for the transformed data set
     */
    public void setRange(double A, double B)
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
    
    

    @Override
    public void fit(DataSet dataSet)
    {
        mins = new DenseVector(dataSet.getNumNumericalVars());
        Vec maxs = new DenseVector(mins.length());
        mutliplyConstants = new DenseVector(mins.length());
        
        OnLineStatistics[] stats = dataSet.getOnlineColumnStats(false);
        
        for(int i = 0; i < mins.length(); i++)
        {
            double min = stats[i].getMin();
            double max = stats[i].getMax();
            if (max - min < 1e-6)//No change
            {
                mins.set(i, 0);
                maxs.set(i, 1);
                mutliplyConstants.set(i, 1.0);
            }
            else
            {
                mins.set(i, min);
                maxs.set(i, max);
                mutliplyConstants.set(i, A - B);
            }
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
    
    /**
     * Copy constructor
     * @param other the transform to copy
     */
    private LinearTransform(LinearTransform other)
    {
        this.A = other.A;
        this.B = other.B;
        if(other.mins != null)
            this.mins = other.mins.clone();
        if(other.mutliplyConstants != null)
            this.mutliplyConstants = other.mutliplyConstants.clone();
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        DataPoint toRet = dp.clone();
        mutableTransform(toRet);
        return toRet;
    }
    
    @Override
    public LinearTransform clone()
    {
        return new LinearTransform(this);
    }

    @Override
    public void mutableInverse(DataPoint dp)
    {
        Vec v = dp.getNumericalValues();
        v.mutableSubtract(B);
        v.mutablePairwiseDivide(mutliplyConstants);
        v.mutableAdd(mins);
    }


    @Override
    public void mutableTransform(DataPoint dp)
    {
        Vec v = dp.getNumericalValues();
        v.mutableSubtract(mins);
        v.mutablePairwiseMultiply(mutliplyConstants);
        v.mutableAdd(B);
    }

    @Override
    public boolean mutatesNominal()
    {
        return false;
    }

    @Override
    public DataPoint inverse(DataPoint dp)
    {
        DataPoint toRet = dp.clone();
        mutableInverse(toRet);
        return toRet;
    }
}
