
package jsat.classifiers;

import jsat.linear.Vec;

/**
 * 
 * @author Edward Raff
 */
public class DataPoint
{
    private double weight;
    protected Vec numericalValues;
    protected int[] categoricalValues;
    protected CategoricalData[] categoricalData;

    public DataPoint(Vec numericalValues, int[] categoricalValues, CategoricalData[] categoricalData)
    {
        this(numericalValues, categoricalValues, categoricalData, 1);
    }
    
    public DataPoint(Vec numericalValues, int[] categoricalValues, CategoricalData[] categoricalData, double weight)
    {
        this.numericalValues = numericalValues;
        this.categoricalValues = categoricalValues;
        this.categoricalData = categoricalData;
        this.weight = weight;
    }

    public double getWeight()
    {
        return weight;
    }

    public void setWeight(double weight)
    {
        this.weight = weight;
    }
    
    public boolean containsCategoricalData()
    {
        return categoricalValues.length > 0;
    }

    public Vec getNumericalValues()
    {
        return numericalValues;
    }
    
    public boolean containsNumericalData()
    {
        return numericalValues != null && numericalValues.length() > 0;
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
    
    /**
     * 
     * @param i the i'th categorical variable
     * @return the value of the i'th categorical variable
     * 
     */
    public int getCategoricalValue(int i)
    {
        return categoricalValues[i];
    }
    
    public CategoricalData[] getCategoricalData()
    {
        return categoricalData;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder("Numerical: ");
        sb.append(numericalValues.toString());
        
        sb.append(" Categorical: ");
        
        for(int i  = 0; i < categoricalValues.length; i++)
        {
            sb.append(categoricalData[i].catName(categoricalValues[i]));
            sb.append(",");
        }
            
        
        return sb.toString();
    }
    
    
//    public DataPoint copy()
//    {
//        
//    }
    
    
}
