
package jsat.datatransform;

import java.util.Arrays;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.discrete.UniformDiscrete;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * This transform converts numerical features into categorical ones via a simple
 * histogram. Bins will be created for each numeric feature of equal sizes. Each
 * numeric feature will be converted to the same number of bins. <br>
 * This transform will handle missing values by simply ignoring them, and
 * leaving the value missing in the transformed categorical variable.
 *
 * 
 * @author Edward Raff
 */
public class NumericalToHistogram implements DataTransform
{

    private static final long serialVersionUID = -2318706869393636074L;
    private int n;
    //First index is the vector index, 2nd index is the min value then the increment value
    double[][] conversionArray;
    CategoricalData[] newDataArray;

    /**
     * Creates a new transform which will use at most 25 bins when converting
     * numeric features. This may not be optimal for any given dataset
     *
     */
    public NumericalToHistogram()
    {
        this(25);
    }
    
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
     * Creates a new transform which will use at most the specified number of bins
     * 
     * @param n the number of bins to create
     */
    public NumericalToHistogram(int n)
    {
        setNumberOfBins(n);
    }

    /**
     * Creates a new transform which will use the specified number of bins for
     * each numeric feature. 
     * @param dataSet the data set to create the transform from
     * @param n the number of bins to create
     */
    public NumericalToHistogram(DataSet dataSet, int n)
    {
        this(n);
        fit(dataSet);
    }

    @Override
    public void fit(DataSet dataSet)
    {
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
                final double val = v.get(j);
                if(Double.isNaN(val))
                    continue;
                mins[j] = Math.min(mins[j], val);
                maxs[j] = Math.max(maxs[j], val);
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
     * Sets the maximum number of histogram bins to use when creating the categorical version of numeric features. 
     * @param n the number of bins to create
     */
    public void setNumberOfBins(int n)
    {
        if(n <= 0)
            throw new RuntimeException("Must partition into a positive number of groups");
        this.n = n;
    }

    /**
     * 
     * @return the maximum number of bins to create
     */
    public int getNumberOfBins()
    {
        return n;
    }
    
    /**
     * Attempts to guess the number of bins to use 
     * @param data the dataset to be transforms
     * @return a distribution of the guess
     */
    public static Distribution guessNumberOfBins(DataSet data)
    {
        if(data.getSampleSize() < 20)
            return new UniformDiscrete(2, data.getSampleSize()-1);
        else if(data.getSampleSize() >= 1000000)
            return new LogUniform(50, 1000);
        int sqrt = (int) Math.sqrt(data.getSampleSize());
        return new UniformDiscrete(Math.max(sqrt/3, 2), Math.min(sqrt*3, data.getSampleSize()-1));
    }
    
    /**
     * Copy constructor
     * @param other the transform to copy
     */
    private NumericalToHistogram(NumericalToHistogram other)
    {
        this.n = other.n;
        if(other.conversionArray != null)
        {
            this.conversionArray = new double[other.conversionArray.length][];
            for(int i = 0; i < other.conversionArray.length; i++)
                this.conversionArray[i] = Arrays.copyOf(other.conversionArray[i], other.conversionArray[i].length);
        }
        
        if(other.newDataArray != null)
        {
            this.newDataArray = new CategoricalData[other.newDataArray.length];
            for(int i = 0; i < other.newDataArray.length; i++)
                this.newDataArray[i] = other.newDataArray[i].clone();
        }
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        
        int[] newCatVals = new int[newDataArray.length];
        
        Vec v = dp.getNumericalValues();
        for(int i = 0; i < conversionArray.length; i++)
        {
            double val = v.get(i) - conversionArray[i][0];
            
            if(Double.isNaN(val))
            {
                newCatVals[i] = -1;//missing
                continue;
            }
            
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
    public NumericalToHistogram clone()
    {
        return new NumericalToHistogram(this);
    }
}
