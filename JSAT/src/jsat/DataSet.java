
package jsat;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.math.OnLineStatistics;

/**
 * This is the base class for representing a data set. A data set contains multiple samples,
 * each of which should have the same number of attributes. Conceptually, each 
 * {@link DataPoint} represents a row in the data set, and the attributes form the columns. 
 * 
 * @author Edward Raff
 */
public abstract class DataSet<D extends DataSet>
{
    /**
     * The number of numerical values each data point must have
     */
    protected int numNumerVals;
    /**
     * Contains the categories for each of the categorical variables
     */
    protected CategoricalData[] categories;
    /**
     * The list, in order, of the names of the numeric variables. 
     * This should be filled with default values on construction,
     * that can then be changed later. 
     */
    protected List<String> numericalVariableNames;
    
    /**
     * Sets the unique name associated with the <tt>i</tt>'th numeric attribute. All strings will be converted to lower case first. 
     * 
     * @param name the name to use
     * @param i the <tt>i</tt>th attribute. 
     * @return <tt>true</tt> if the value was set, <tt>false</tt> if it was not set because an invalid index was given . 
     */
    public boolean setNumericName(String name, int i)
    {
        name = name.toLowerCase();
        
        if(numericalVariableNames.contains(name))
            return false;
        else if(i < getNumNumericalVars() && i >= 0)
            numericalVariableNames.set(i, name);
        else
            return false;
        
        return true;
    }

    /**
     * Returns the name used for the <tt>i</tt>'th numeric attribute. 
     * @param i the <tt>i</tt>th attribute. 
     * @return the name used for the <tt>i</tt>'th numeric attribute. 
     */
    public String getNumericName(int i )
    {
        if(i < getNumNumericalVars() && i >= 0)
            return numericalVariableNames.get(i);
        else
            throw new IndexOutOfBoundsException("Can not acces variable for invalid index  " + i );
    }
    
    /**
     * Returns the name used for the <tt>i</tt>'th categorical attribute. 
     * @param i the <tt>i</tt>th attribute. 
     * @return the name used for the <tt>i</tt>'th categorical attribute. 
     */
    public String getCategoryName(int i )
    {
        if(i < getNumCategoricalVars() && i >= 0)
            return categories[i].getCategoryName();
        else
            throw new IndexOutOfBoundsException("Can not acces variable for invalid index  " + i );
    }
        
    /**
     * Applies the given transformation to all points in this data set
     * @param dt the transformation to apply
     */
    public void applyTransform(DataTransform dt)
    {
        for(int i = 0; i < getSampleSize(); i++)
            setDataPoint(i, dt.transform(getDataPoint(i)));
        //TODO this should be added to DataTransform
        numNumerVals = getDataPoint(0).numNumericalValues();
        categories = getDataPoint(0).getCategoricalData();
        this.numericalVariableNames.clear();
        for(int i = 0; i < getNumNumericalVars(); i++)
            numericalVariableNames.add("Transformed Numeric Variable " + (i+1));
    }
    
    /**
     * Returns the <tt>i</tt>'th data point in this set. The order will never 
     * chance so long as no data points are added or removed from the set. 
     * 
     * @param i the <tt>i</tt>'th data point in this set
     * @return the <tt>i</tt>'th data point in this set
     */
    abstract public DataPoint getDataPoint(int i);
    
    /**
     * Replaces an already existing data point with the one given. 
     * Any values associated with the data point, but not apart of
     * it, will remain intact. 
     * 
     * @param i the <tt>i</tt>'th dataPoint to set. 
     */
    abstract public void setDataPoint(int i, DataPoint dp);
    
    /**
     * Returns summary statistics computed in an online fashion for each numeric
     * variable. This consumes less memory, but can be less numerically stable. 
     *  This method does not take into account data point weight. 
     * 
     * @return an array of summary statistics
     */
    public OnLineStatistics[] singleVarStats()
    {
        OnLineStatistics[] stats = new OnLineStatistics[numNumerVals];
        for(int i = 0; i < stats.length; i++)
            stats[i] = new OnLineStatistics();
        
        for(Iterator<DataPoint> iter = getDataPointIterator(); iter.hasNext(); )
        {
            Vec v = iter.next().getNumericalValues();
            for (int i = 0; i < numNumerVals; i++)
                stats[i].add(v.get(i));
        }
        
        return stats;
    }
    
    /**
     * Returns summary statistics computed in an online fashion for each numeric
     * variable. This consumes less memory, but can be less numerically stable.
     * This method takes into account data point weight. 
     * 
     * @return array of summary statistics
     */
    public OnLineStatistics[] getWeightedSingleVarStats()
    {
         OnLineStatistics[] stats = new OnLineStatistics[numNumerVals];
        for(int i = 0; i < stats.length; i++)
            stats[i] = new OnLineStatistics();
        
        for(Iterator<DataPoint> iter = getDataPointIterator(); iter.hasNext(); )
        {
            DataPoint dp = iter.next();
            Vec v = dp.getNumericalValues();
            for (int i = 0; i < numNumerVals; i++)
                stats[i].add(v.get(i), dp.getWeight());
        }
        
        return stats;
    }
    
    /**
     * Returns an iterator that will iterate over all data points in the set. 
     * The behavior is not defined if one attempts to modify the data set 
     * while being iterated.
     * 
     * @return an iterator for the data points
     */
    public Iterator<DataPoint> getDataPointIterator()
    {
        Iterator<DataPoint> iteData = new Iterator<DataPoint>() 
        {
            int cur = 0;
            int to = getSampleSize();

            public boolean hasNext()
            {
                return cur < to;
            }

            public DataPoint next()
            {
                return getDataPoint(cur++);
            }

            public void remove()
            {
                throw new UnsupportedOperationException("This operation is not supported for DataSet");
            }
        };
        
        return iteData;
    }
    
    /**
     * Returns the number of data points in this data set
     * @return the number of data points in this data set 
     */
    abstract public int getSampleSize();
    
    /**
     * Returns the number of categorical variables for each data point in the set
     * @return the number of categorical variables for each data point in the set
     */
    public int getNumCategoricalVars()
    {
        return categories.length;
    }
    
    /**
     * Returns the number of numerical variables for each data point in the set
     * @return the number of numerical variables for each data point in the set 
     */
    public int getNumNumericalVars()
    {
        return numNumerVals;
    }
    
    /**
     * Returns the array containing the categorical data information for this data 
     * set. Changes to this will be reflected in the data set. 
     * 
     * @return the array of {@link CategoricalData}
     */
    public CategoricalData[] getCategories()
    {
        return categories;
    }
    
    /**
     * Creates <tt>folds</tt> data sets that contain data from this data set. 
     * The data points in each set will be random. These are meant for cross 
     * validation
     * 
     * @param folds the number of cross validation sets to create. Should be greater then 1
     * @param rand the source of randomness 
     * @return the list of data sets. 
     */
    abstract public List<D> cvSet(int folds, Random rand);
    
    /**
     * Creates <tt>folds</tt> data sets that contain data from this data set. 
     * The data points in each set will be random. These are meant for cross 
     * validation
     * 
     * @param folds the number of cross validation sets to create. Should be greater then 1
     * @return the list of data sets. 
     */
    public List<D> cvSet(int folds)
    {
        return cvSet(folds, new Random());
    }
    
    /**
     * Creates a list containing the same DataPoints in this set. They are soft copies,
     * in the same order as this data set. However, altering this list will have no 
     * effect on DataSet. Altering the DataPoints in the list will effect the 
     * DataPoints in this DataSet. 
     * 
     * @return a list of the DataPoints in this DataSet.
     */
    public List<DataPoint> getDataPoints()
    {
        List<DataPoint> list = new ArrayList<DataPoint>(getSampleSize());
        for(int i = 0; i < getSampleSize(); i++)
            list.add(getDataPoint(i));
        return list;
    }
    
    /**
     * The data set can be seen as a NxM matrix, were each row is a 
     * data point, and each column the values for a particular 
     * variable. This method grabs all the numerical values for
     * a 'column' and returns it as one vector. <br>
     * This vector can be altered and will not effect any of the values in the data set
     * 
     * @param i the <tt>i</tt>'th numerical variable to obtain all values of
     * @return a Vector of length {@link #getSampleSize() }
     */
    public Vec getNumericColumn(int i )
    {
        if(i < 0 || i >= getNumNumericalVars())
            throw new IndexOutOfBoundsException("There is no index for column " + i);
        DenseVector dv = new DenseVector(getSampleSize());
        for(int j = 0; j < getSampleSize(); j++)
            dv.set(j, getDataPoint(j).getNumericalValues().get(i));
        return dv;
    }
    
    /**
     * Creates a matrix from the data set, where each row represent a data
     * point, and each column is one of the numeric example from the data set. 
     * <br>
     * This matrix can be altered and will not effect any of the values in the data set. 
     * 
     * @return a matrix of the data points. 
     */
    public Matrix getDataMatrix()
    {
        DenseMatrix matrix = new DenseMatrix(this.getSampleSize(), this.getNumNumericalVars());
        
        for(int i = 0; i < getSampleSize(); i++)
        {
            Vec row = getDataPoint(i).getNumericalValues();
            for(int j = 0; j < row.length(); j++)
                matrix.set(i, j, row.get(j));
        }
        
        return matrix;
    }
    
    /**
     * Returns the number of features in this data set, which is the sum of {@link #getNumCategoricalVars() } and {@link #getNumNumericalVars() }
     * @return the total number of features in this data set
     */
    public int getNumFeatures()
    {
        return getNumCategoricalVars() + getNumNumericalVars();
    }
}
