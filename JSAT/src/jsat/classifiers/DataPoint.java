
package jsat.classifiers;

import java.io.Serializable;
import java.util.Arrays;
import jsat.linear.Vec;

/**
 * This is the general class object for representing a singular data point in a data set. 
 * Every data point is made up of either categorical variables, numerical variables, 
 * or a combination of the two. 
 * 
 * @author Edward Raff
 */
public class DataPoint implements Cloneable, Serializable
{

	private static final long serialVersionUID = -1363327591317639955L;
	private double weight;
    protected Vec numericalValues;
    protected int[] categoricalValues;
    protected CategoricalData[] categoricalData;
    private static final int[] emptyInt = new int[0];
    private static final CategoricalData[] emptyData = new CategoricalData[0];

    /**
     *  Creates a new data point with the default weight of 1.0
     * 
     * @param numericalValues a vector containing the numerical values for this data point
     * @param categoricalValues an array of the category values for this data point
     * @param categoricalData an array of the category information of this data point
     */
    public DataPoint(Vec numericalValues, int[] categoricalValues, CategoricalData[] categoricalData)
    {
        this(numericalValues, categoricalValues, categoricalData, 1);
    }
    
    /**
     * Creates a new data point 
     * 
     * @param numericalValues a vector containing the numerical values for this data point
     * @param categoricalValues an array of the category values for this data point
     * @param categoricalData an array of the category information of this data point
     * @param weight a double indicating how much weight this data point has, 1.0 
     * being the standard weight. Not all algorithms make use of data points that
     * have different weights. 
     */
    public DataPoint(Vec numericalValues, int[] categoricalValues, CategoricalData[] categoricalData, double weight)
    {
        this.numericalValues = numericalValues;
        this.categoricalValues = categoricalValues;
        this.categoricalData = categoricalData;
        this.weight = weight;
    }
    
    /**
     * Creates a new data point that has no categorical variables
     * 
     * @param numericalValues a vector containing the numerical values for this data point
     * @param weight a double indicating how much weight this data point has, 1.0 
     * being the standard weight. Not all algorithms make use of data points that
     * have different weights. 
     */
    public DataPoint(Vec numericalValues, double weight)
    {
        this(numericalValues, emptyInt, emptyData, weight);
    }
    
     /**
     * Creates a new data point that has no categorical variables and a weight of 1.0
     * 
     * @param numericalValues a vector containing the numerical values for this data point
     */
    public DataPoint(Vec numericalValues)
    {
        this(numericalValues, emptyInt, emptyData);
    }

    /**
     * Returns the weight that this data point carries. 
     * @return the weight that this data point carries. 
     */
    public double getWeight()
    {
        return weight;
    }

    /**
     * Set the weight that this data point should carry. The norm is 1.0
     * @param weight the new weight value
     * @throws ArithmeticException if the weight value is not a number &gt; 0
     */
    public void setWeight(double weight)
    {
        if(Double.isNaN(weight) || Double.isInfinite(weight) || weight <= 0)
            throw new ArithmeticException("Invalid weight assignment of  " + weight);
        this.weight = weight;
    }
    
    /**
     * Returns true if this data point contains any categorical variables, false otherwise.
     * @return true if this data point contains any categorical variables, false otherwise.
     */
    public boolean containsCategoricalData()
    {
        return categoricalValues.length > 0;
    }

    /**
     * Returns the vector containing the numerical values. Altering this 
     * vector will effect this data point. If changes are going to be 
     * made, a clone of the vector should be made by the caller. 
     * @return the vector containing the numerical values.
     */
    public Vec getNumericalValues()
    {
        return numericalValues;
    }
    
    /**
     * Returns true if the data point contains any numerical variables, false otherwise. 
     * @return true if the data point contains any numerical variables, false otherwise. 
     */
    public boolean containsNumericalData()
    {
        return numericalValues != null && numericalValues.length() > 0;
    }
    
    /**
     * Returns the number of numerical variables in this data point. 
     * @return the number of numerical variables in this data point.  
     */
    public int numNumericalValues()
    {
        return numericalValues == null ? 0 : numericalValues.length();
    }
    
    /**
     * Returns the array of values for each category. Altering 
     * this array will effect this data point. If changes are 
     * going to be made, a clone of the array should be made 
     * by the caller. 
     * @return the array of values for each category.  
     */
    public int[] getCategoricalValues()
    {
        return categoricalValues;
    }
    
    /**
     * Returns the number of categorical variables in this data point. 
     * @return the number of categorical variables in this data point.  
     */
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
    
    /**
     * Returns the array of Categorical Data information
     * @return the array of Categorical Data information 
     */
    public CategoricalData[] getCategoricalData()
    {
        return categoricalData;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        if(containsNumericalData())
        {
            sb.append("Numerical: ");
            sb.append(numericalValues.toString());
        }
        
        if(containsCategoricalData())
        {
            sb.append(" Categorical: ");
            for(int i  = 0; i < categoricalValues.length; i++)
            {
                sb.append(categoricalData[i].getOptionName(categoricalValues[i]));
                sb.append(",");
            }
        }
            
        
        return sb.toString();
    }
    
    /**
     * Creates a deep clone of this data point, such that altering either data point does not effect the other one. 
     * @return a deep clone of this data point. 
     */
    public DataPoint clone()
    {
        return new DataPoint(numericalValues.clone(), 
                Arrays.copyOf(categoricalValues, categoricalValues.length),
                CategoricalData.copyOf(categoricalData),
                weight);
    }
}
