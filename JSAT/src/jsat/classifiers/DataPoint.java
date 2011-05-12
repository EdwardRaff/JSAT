
package jsat.classifiers;

import jsat.linear.Vec;

/**
 * 
 * @author Edward Raff
 */
public class DataPoint
{
    protected Vec numericalValues;
    protected int[] categoricalValues;
    protected CategoricalData[] categoricalData;

    public DataPoint(Vec numericalValues, int[] categoricalValues, CategoricalData[] categoricalData)
    {
        this.numericalValues = numericalValues;
        this.categoricalValues = categoricalValues;
        this.categoricalData = categoricalData;
    }
    
    public boolean containsCategoricalData()
    {
        return categoricalValues.length > 0;
    }
    
    public boolean containsNumericalData()
    {
        return numericalValues != null && numericalValues.length() > 0;
    }
    
    public Vec getNominalValues()
    {
        return numericalValues;
    }
    
    public int numNominalValues()
    {
        return numericalValues == null ? 0 : numericalValues.length();
    }
    
    public int[] getCategoricalValues()
    {
        return categoricalValues;
    }
    
    public int numCategoricalValues()
    {
        return categoricalValues.length;
    }
    
    public int getCategoricalValue(int i)
    {
        return categoricalValues[i];
    }
    
    public CategoricalData[] getCategoricalData()
    {
        return categoricalData;
    }
    
}
