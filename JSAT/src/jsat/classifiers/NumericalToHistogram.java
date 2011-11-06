
package jsat.classifiers;

import jsat.DataSet;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class NumericalToHistogram implements DataTransform
{
    private int n;
    //First index is the vector index, 2nd index is the min value then the increment value
    double[][] conversionArray;
    CategoricalData[] newDataArray;

    public NumericalToHistogram(DataSet dataSet)
    {
        this(dataSet, (int) Math.ceil(Math.sqrt(dataSet.getSampleSize())));
    }

    
    
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
    
}
