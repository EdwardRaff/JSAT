
package jsat.datatransform;

import java.util.Arrays;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * This transform converts numerical features into categorical ones via a simple
 * histogram. Bins will be created for each numeric feature of equal sizes. Each
 * numeric feature will be converted to the same number of bins. 
 * 
 * @author Edward Raff
 */
public class NumericalToHistogram implements DataTransform
{
    private int n;
    //First index is the vector index, 2nd index is the min value then the increment value
    double[][] conversionArray;
    CategoricalData[] newDataArray;

    /**
     * Creates a new transform which will use O(sqrt(n)) bins for each numeric 
     * feature, where <i>n</i> is the number of data points in the dataset. 
     * 
     * @param dataSet the data set to create the transform from
     */
    public NumericalToHistogram(DataSet dataSet)
    {
        this(dataSet, (int) Math.ceil(Math.sqrt(dataSet.getSampleSize())));
    }

    /**
     * Creates a new transform which will use the specified number of bins for
     * each numeric feature. 
     * @param dataSet the data set to create the transform from
     * @param n the number of bins to create
     */
    public NumericalToHistogram(DataSet dataSet, int n)
    {
        if(n <= 0)
            throw new RuntimeException("Must partition into a positive number of groups");
        this.n = n;
        
        conversionArray = new double[dataSet.getNumNumericalVars()][2];
        
        double[] mins = new double[conversionArray.length];
        double[] maxs = new double[conversionArray.length];
        for(int i = 0; i < mins.length; i++)
        {
            mins[i] = Double.MAX_VALUE;
            maxs[i] = Double.MIN_VALUE;
        }
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            Vec v = dataSet.getDataPoint(i).getNumericalValues();
            for(int j = 0; j < mins.length; j++)
            {
                mins[j] = Math.min(mins[j], v.get(j));
                maxs[j] = Math.max(maxs[j], v.get(j));
            }
        }
        
        for(int i = 0; i < conversionArray.length; i++)
        {
            conversionArray[i][0] = mins[i];
            conversionArray[i][1] = (maxs[i]-mins[i])/n;
        }
        
        newDataArray = new CategoricalData[dataSet.getNumNumericalVars() + dataSet.getNumCategoricalVars()];
        for(int i = 0; i < dataSet.getNumNumericalVars(); i++)
            newDataArray[i] = new CategoricalData(n);
        System.arraycopy(dataSet.getCategories(), 0, newDataArray, dataSet.getNumNumericalVars(), dataSet.getNumCategoricalVars());
    }
    
    /**
     * Copy constructor
     * @param other the transform to copy
     */
    private NumericalToHistogram(NumericalToHistogram other)
    {
        this.n = other.n;
        this.conversionArray = new double[other.conversionArray.length][];
        for(int i = 0; i < other.conversionArray.length; i++)
            this.conversionArray[i] = Arrays.copyOf(other.conversionArray[i], other.conversionArray[i].length);
        this.newDataArray = new CategoricalData[other.newDataArray.length];
        for(int i = 0; i < other.newDataArray.length; i++)
            this.newDataArray[i] = other.newDataArray[i].clone();
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        
        int[] newCatVals = new int[newDataArray.length];
        
        Vec v = dp.getNumericalValues();
        for(int i = 0; i < conversionArray.length; i++)
        {
            double val = v.get(i) - conversionArray[i][0];
            
            int catVal = (int) Math.floor(val / conversionArray[i][1]);
            if(catVal < 0)
                catVal = 0;
            else if(catVal >= n)
                catVal = n-1;
            
            newCatVals[i] = catVal;
        }
        System.arraycopy(dp.getCategoricalValues(), 0, newCatVals, conversionArray.length, dp.numCategoricalValues());
        
        return new DataPoint(new DenseVector(0), newCatVals, newDataArray);
    }

    @Override
    public DataTransform clone()
    {
        return new NumericalToHistogram(this);
    }
    
    /**
     * Factory for the creation of {@link NumericalToHistogram} transforms. 
     */
    static public class NumericalToHistogramTransformFactory extends DataTransformFactoryParm
    {
        private int n;

        /**
         * Creates a new NumericalToHistogram factory. 
         */
        public NumericalToHistogramTransformFactory()
        {
            this(Integer.MAX_VALUE);
        }

        /**
         * Creates a new NumericalToHistogram factory. 
         * @param n the number of bins to create
         */
        public NumericalToHistogramTransformFactory(int n)
        {
            setBins(n);
        }
        
        public NumericalToHistogramTransformFactory(NumericalToHistogramTransformFactory toCopy)
        {
            this(toCopy.n);
        }

        /**
         * Sets the number of numeric bins to use. {@link Integer#MAX_VALUE} is
         * used as a special value to indicate that the square root of the 
         * number of data points should be used as the number of bins. 
         * 
         * @param n the number of bins to use, or {@link Integer#MAX_VALUE} to
         * use sqrt(n) bins. 
         */
        public void setBins(int n)
        {
            if(n < 1)
                throw new IllegalArgumentException("Number of bins must be a positive value");
            this.n = n;
        }

        /**
         * Returns the number of bins to use
         * @return the number of bins to use
         */
        public int getBins()
        {
            return n;
        }
        
        @Override
        public DataTransform getTransform(DataSet dataset)
        {
            if(n == Integer.MAX_VALUE)
                return new NumericalToHistogram(dataset);
            else
                return new NumericalToHistogram(dataset, n);
        }

        @Override
        public NumericalToHistogramTransformFactory clone()
        {
            return new NumericalToHistogramTransformFactory(this);
        }
        
    }
}
