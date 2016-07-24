
package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.*;

/**
 * This transform converts nominal feature values to numeric ones be adding a 
 * new numeric feature for each possible categorical value for each nominal 
 * feature. The numeric features will be all zeros, with only a single numeric 
 * feature having a value of "1.0" for each nominal variable. 
 * 
 * @author Edward Raff
 */
public class NominalToNumeric implements DataTransform
{

    private static final long serialVersionUID = -7765605678836464143L;
    private int origNumericalCount;
    private CategoricalData[] categoricalData;
    private int addedNumers;
    
    /**
     * Creates a new transform to convert categorical to numeric features
     */
    public NominalToNumeric()
    {
    }
    
    /**
     * Creates a new transform to convert categorical to numeric features for the given dataset
     * @param dataSet the dataset to fit the transform to
     */
    public NominalToNumeric(DataSet dataSet)
    {
        fit(dataSet);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public NominalToNumeric(NominalToNumeric toCopy)
    {
        this.origNumericalCount = toCopy.origNumericalCount;
        this.categoricalData = toCopy.categoricalData;
        this.addedNumers = toCopy.addedNumers;
    }
    

    @Override
    public void fit(DataSet data)
    {
        this.origNumericalCount = data.getNumNumericalVars();
        this.categoricalData = data.getCategories();
        addedNumers = 0;
        
        for(CategoricalData cd : categoricalData)
            addedNumers += cd.getNumOfCategories();
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        Vec v;
        
        //TODO we should detect if there are going to be so many sparce spaces added by the categorical data that we should just choose a sparce vector anyway
        if(dp.getNumericalValues().isSparse())
            v = new SparseVector(origNumericalCount+addedNumers);
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

    @Override
    public NominalToNumeric clone()
    {
        return new NominalToNumeric(this);
    }
}
