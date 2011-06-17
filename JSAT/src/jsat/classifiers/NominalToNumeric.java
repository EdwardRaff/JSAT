
package jsat.classifiers;

import jsat.linear.DenseVector;
import jsat.linear.SparceVector;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class NominalToNumeric implements DataTransform
{
    private final int origNumericalCount;
    private final CategoricalData[] categoricalData;
    private int addedNumers;

    public NominalToNumeric(int origNumericalCount, CategoricalData[] categoricalData)
    {
        this.origNumericalCount = origNumericalCount;
        this.categoricalData = categoricalData;
        addedNumers = 0;
        
        for(CategoricalData cd : categoricalData)
            addedNumers += cd.getNumOfCategories();
                
    }
    
    public DataPoint transform(DataPoint dp)
    {
        Vec v;
        
        //TODO we should detect if there are going to be so many sparce spaces added by the categorical data that we should just choose a sparce vector anyway
        if(dp.getNumericalValues() instanceof SparceVector)
            v = new SparceVector(origNumericalCount+addedNumers);
        else
            v = new DenseVector(origNumericalCount+addedNumers);
        
        
        Vec oldV = dp.getNumericalValues();
        int i = 0;
        for(i = 0; i < origNumericalCount; i++)
            v.set(i, oldV.get(i)); 
        for(int j =0; j < categoricalData.length; j++)
        {
            v.set(i+dp.getCategoricalValue(j), 1.0);
            i += categoricalData[j].getNumOfCategories();
        }
        
        return new DataPoint(v, new int[0], new CategoricalData[0]);
    }
    
}
