
package jsat;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataTransform;
import jsat.linear.Vec;
import jsat.math.OnLineStatistics;

/**
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
    
    
    public void applyTransform(DataTransform dt)
    {
        for(int i = 0; i < getSampleSize(); i++)
            setDataPoint(i, dt.transform(getDataPoint(i)));
        //TODO this should be added to DataTransform
        numNumerVals = getDataPoint(0).numNumericalValues();
        categories = getDataPoint(0).getCategoricalData();
    }
    
    /**
     * 
     * @param i the i'th data point in this set
     * @return the ith data point in this set
     */
    abstract public DataPoint getDataPoint(int i);
    
    /**
     * Replaces an already existing data point with the one given. 
     * Any values associated with the data point, but not apart of
     * it, will remain intact. 
     * 
     * @param i the i'th dataPoint to set. 
     */
    abstract public void setDataPoint(int i, DataPoint dp);
    
    /**
     * Returns summary statistics computed in an online fashion for each numeric
     * variable. This consumes less memory, but can be less numerically stable. 
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
     * 
     * @return the number of data points in this data set 
     */
    abstract public int getSampleSize();
    
    /**
     * 
     * @return the number of categorical variables for each data point in the set
     */
    public int getNumCategoricalVars()
    {
        return categories.length;
    }
    
    /**
     * 
     * @return the number of numerical variables for each data point in the set 
     */
    public int getNumNumericalVars()
    {
        return numNumerVals;
    }
    
    public CategoricalData[] getCategories()
    {
        return categories;
    }
    
    abstract public List<D> cvSet(int folds, Random rand);
    
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
}
