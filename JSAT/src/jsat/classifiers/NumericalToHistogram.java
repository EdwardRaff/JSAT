
package jsat.classifiers;

import java.util.Arrays;
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

    public NumericalToHistogram(ClassificationDataSet cds)
    {
        this(cds, (int) Math.ceil(Math.sqrt(cds.getSampleSize())));
    }

    
    
    public NumericalToHistogram(ClassificationDataSet cds, int n)
    {
        if(n <= 0)
            throw new RuntimeException("Must partition into a positive number of groups");
        this.n = n;
        
        conversionArray = new double[cds.getNumNumericalVars()][2];
        
        double[] mins = new double[conversionArray.length];
        double[] maxs = new double[conversionArray.length];
        for(int i = 0; i < mins.length; i++)
        {
            mins[i] = Double.MAX_VALUE;
            maxs[i] = Double.MIN_VALUE;
        }
        for(int i = 0; i < cds.getSampleSize(); i++)
        {
            Vec v = cds.getDataPoint(i).getNumericalValues();
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
        
        newDataArray = new CategoricalData[cds.getNumNumericalVars() + cds.getNumCategoricalVars()];
        for(int i = 0; i < cds.getNumNumericalVars(); i++)
            newDataArray[i] = new CategoricalData(n);
        System.arraycopy(cds.getCategories(), 0, newDataArray, cds.getNumNumericalVars(), cds.getNumCategoricalVars());
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
