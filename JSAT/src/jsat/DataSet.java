
package jsat;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.datatransform.InPlaceTransform;
import jsat.linear.*;
import jsat.math.OnLineStatistics;

/**
 * This is the base class for representing a data set. A data set contains multiple samples,
 * each of which should have the same number of attributes. Conceptually, each 
 * {@link DataPoint} represents a row in the data set, and the attributes form the columns. 
 * 
 * @author Edward Raff
 */
public abstract class DataSet
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
     * Applies the given transformation to all points in this data set, 
     * replacing each data point with the new value. No mutation of the data 
     * points will occur
     * 
     * @param dt the transformation to apply
     */
    public void applyTransform(DataTransform dt)
    {
        applyTransform(dt, false);
    }

    /**
     * Applies the given transformation to all points in this data set. If the
     * transform supports mutating the original data points, this will be
     * applied if {@code mutableTransform} is set to {@code true}
     *
     * @param dt the transformation to apply
     * @param mutate {@code true} to mutableTransform the original data points,
     * {@code false} to ignore the ability to mutableTransform and replace the original
     * data points.
     */
    public void applyTransform(DataTransform dt, boolean mutate)
    {
        if (mutate && dt instanceof InPlaceTransform)
        {
            InPlaceTransform ipt = (InPlaceTransform) dt;
            for (int i = 0; i < getSampleSize(); i++)
                ipt.mutableTransform(getDataPoint(i));
        }
        else
            for (int i = 0; i < getSampleSize(); i++)
                setDataPoint(i, dt.transform(getDataPoint(i)));
        //TODO this should be added to DataTransform
        numNumerVals = getDataPoint(0).numNumericalValues();
        categories = getDataPoint(0).getCategoricalData();
        if (this.numericalVariableNames != null)
        {
            this.numericalVariableNames.clear();
            for (int i = 0; i < getNumNumericalVars(); i++)
                numericalVariableNames.add("TN" + (i + 1));
        }
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
     * variable. This returns all summary statistics, but can be less 
     * numerically stable and uses more memory. 
     * 
     * @param useWeights {@code true} to return the weighted statistics, 
     * unweighted otherwise. 
     * @return an array of summary statistics
     */
    public OnLineStatistics[] getOnlineColumnStats(boolean useWeights)
    {
        OnLineStatistics[] stats = new OnLineStatistics[numNumerVals];
        for(int i = 0; i < stats.length; i++)
            stats[i] = new OnLineStatistics();
        
        double totalSoW = 0.0;
        
        for(Iterator<DataPoint> iter = getDataPointIterator(); iter.hasNext(); )
        {
            DataPoint dp = iter.next();
            totalSoW += dp.getWeight();
            
            Vec v = dp.getNumericalValues();
            for(IndexValue iv : v)
                if(useWeights)
                    stats[iv.getIndex()].add(iv.getValue(), dp.getWeight());
                else
                    stats[iv.getIndex()].add(iv.getValue());
        }
        
        double expected = useWeights ? totalSoW : getSampleSize();
        //Add zero counts back in
        for(OnLineStatistics stat : stats)
            stat.add(0.0, expected-stat.getSumOfWeights());
        
        return stats;
    }
    
    /**
     * Returns an {@link OnLineStatistics } object that is built by observing 
     * what proportion of each data point contains non zero numerical values. 
     * A mean of 1 indicates all values were fully dense, and a mean of 0
     * indicates all values were completely sparse (all zeros). 
     * 
     * @return statistics on the percent sparseness of each data point
     */
    public OnLineStatistics getOnlineDenseStats()
    {
        OnLineStatistics stats = new OnLineStatistics();
        double N = getNumNumericalVars();;
        for(int i = 0; i < getSampleSize(); i++)
            stats.add(getDataPoint(i).getNumericalValues().nnz()/N);
        return stats;
    }
    
    /**
     * Computes the weighted mean and variance for each column of feature 
     * values. This has less overhead than 
     * {@link #getOnlineColumnStats(boolean) } but returns less information. 
     * 
     * @return an array of the vectors containing the mean and variance for 
     * each column. 
     */
    public Vec[] getColumnMeanVariance()
    {
        final int d = getNumNumericalVars();
        Vec[] vecs = new Vec[] 
        {
            new DenseVector(d),
            new DenseVector(d)
        };
        
        Vec means = vecs[0];
        Vec stdDevs = vecs[1];
        
        MatrixStatistics.meanVector(means, this);
        MatrixStatistics.covarianceDiag(means, stdDevs, this);
        
        return vecs;
    }
    
    /**
     * Creates an array of {@link #getNumNumericalVars() D} vectors, where each
     * vector has length {@link #getSampleSize() N}. These vectors are the 
     * numeric values of the data set stored in column major order. This is most
     * useful when a data set is sparse, and column strides would take <i>
     * O(log(D))</i> time. This allows traversing the non-zeros in constant time
     * as column vectors. 
     * <br><br>
     * This requires allocation of the new vectors, and will take up space 
     * comparable to the size of the original data set. 
     * @return an array of {@link #getNumNumericalVars() D} vectors, each 
     * containing the column values for that feature for every data point. 
     */
    public Vec[] getColumnMajorVecs()
    {
        final int N = getSampleSize();
        int denseCount = 0;
        int[] nnz = new int[getNumNumericalVars()];
        Vec[] columns = new Vec[nnz.length];
        //First pass to figure out nnz
        for (int i = 0; i < N; i++)
        {
            Vec x_i = getDataPoint(i).getNumericalValues();
            if (x_i.isSparse())
                for (IndexValue iv : x_i)
                    nnz[iv.getIndex()]++;
            else
                denseCount++;
        }

        //Add dense counds and determin if the column is sparse or dense
        for (int j = 0; j < nnz.length; j++)
        {
            nnz[j] += denseCount;

            if (nnz[j] > N / 2)
                columns[j] = new DenseVector(N);
            else
                columns[j] = new SparseVector(N, nnz[j]);
        }

        //Now fill the columns 
        for (int i = 0; i < N; i++)
        {
            Vec x_i = getDataPoint(i).getNumericalValues();
            for(IndexValue iv : x_i)
                columns[iv.getIndex()].set(i, iv.getValue());
        }

        return columns;
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
    abstract public List<? extends DataSet> cvSet(int folds, Random rand);
    
    /**
     * Creates <tt>folds</tt> data sets that contain data from this data set. 
     * The data points in each set will be random. These are meant for cross 
     * validation
     * 
     * @param folds the number of cross validation sets to create. Should be greater then 1
     * @return the list of data sets. 
     */
    public List<? extends DataSet> cvSet(int folds)
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
     * Creates a list of the vectors values for each data point in the correct order. 
     * @return a list of the vectors for the data points
     */
    public List<Vec> getDataVectors()
    {
        List<Vec> vecs = new ArrayList<Vec>(getSampleSize());
        for(int i = 0; i < getSampleSize(); i++)
            vecs.add(getDataPoint(i).getNumericalValues());
        return vecs;
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
    
    /**
     * Returns a new version of this data set that is of the same type, and 
     * contains a different list pointing to the same data points. 
     * @return a shallow copy of this data set
     */
    abstract public DataSet shallowClone();
    
    /**
     * Returns a new version of this data set that is of the same type, and
     * contains a different listing pointing to shallow data point copies. 
     * Because the data point object contains the weight itself, the weight 
     * is not shared - while the vector and array information is. This 
     * allows altering the weights of the data points while preserving the
     * original weights. <br>
     * Altering the list or weights of the returned data set will not be 
     * reflected in the original. Altering the feature values will. 
     * 
     * @return a shallow copy of shallow data point copies for this data set. 
     */
    public DataSet getTwiceShallowClone()
    {
        DataSet clone = shallowClone();
        for(int i = 0; i < clone.getSampleSize(); i++)
        {
            DataPoint d = getDataPoint(i);
            DataPoint sd = new DataPoint(d.getNumericalValues(), d.getCategoricalValues(), d.getCategoricalData());
            clone.setDataPoint(i, sd);
        }
        return clone;
    }
}
