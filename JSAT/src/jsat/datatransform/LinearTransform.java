
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
public class LinearTransform implements DataTransform
{
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
     * Creates a new Linear Transformation for the input data set. 
     * 
     * @param dataSet the data set to learn the transform from
     * @param A the maximum value for the transformed data set
     * @param B the minimum value for the transformed data set
     */
    public LinearTransform(DataSet dataSet, double A, double B)
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
        
        
        mins = new DenseVector(dataSet.getNumNumericalVars());
        Vec maxs = new DenseVector(mins.length());
        mutliplyConstants = new DenseVector(mins.length());
        
        OnLineStatistics[] stats = dataSet.singleVarStats();
        
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
        Vec v = dp.getNumericalValues().subtract(mins);
        v.mutablePairwiseMultiply(mutliplyConstants);
        v.mutableAdd(B);
        
        return new DataPoint(v, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
    }

    @Override
    public DataTransform clone()
    {
        return new LinearTransform(this);
    }
    
    
    public class LinearTransformFactory implements DataTransformFactory
    {
        private Double A;
        private Double B;

        /**
         * Creates a new Linear Transform factory 
         * @param A the maximum value for the transformed data set
         * @param B the minimum value for the transformed data set
         */
        public LinearTransformFactory(double A, double B)
        {
            this.A = A;
            this.B = B;
        }

        /**
         * Creates a new Linear Transform factory for the range [0, 1]
         */
        public LinearTransformFactory()
        {
            this.A = this.B = null;
        }
        
        
        
        @Override
        public DataTransform getTransform(DataSet dataset)
        {
            return new LinearTransform(dataset, A, B);
        }
        
    }
}
