
package jsat;

import java.lang.ref.SoftReference;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.datatransform.InPlaceTransform;
import jsat.linear.*;
import jsat.math.OnLineStatistics;
import jsat.utils.FakeExecutor;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * This is the base class for representing a data set. A data set contains multiple samples,
 * each of which should have the same number of attributes. Conceptually, each 
 * {@link DataPoint} represents a row in the data set, and the attributes form the columns. 
 * 
 * @author Edward Raff
 */
public abstract class DataSet<Type extends DataSet>
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
     * This cache is used to hold a reference to the column vectors that are 
     * returned. It is often the case that the column could be requested 
     * multiple times, especially if someone is doing a grid search, and there 
     * is no need to do the work over again. If the GC is low on memory it can 
     * still collect our cache since we use soft references<br>
     * <br>
     * This map should be cleared whenever the data set as a whole is mutated
     */
    protected Map<Integer, SoftReference<Vec>> columnVecCache = new HashMap<Integer, SoftReference<Vec>>();
    
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
            return numericalVariableNames == null ? null : numericalVariableNames.get(i);
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
     * Applies the given transformation to all points in this data set in 
     * parallel, replacing each data point with the new value. No mutation of 
     * the data points will occur. 
     * 
     * @param dt the transformation to apply
     * @param ex the threadpool to provide threads from. May be {@code null} to 
     * perform operations in serial 
     */
    public void applyTransform(DataTransform dt, ExecutorService ex)
    {
        if(ex == null || ex instanceof FakeExecutor)
            applyTransform(dt);
        else
            applyTransform(dt, false, ex);
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
        applyTransform(dt, mutate, new FakeExecutor());
    }
    
    /**
     * Applies the given transformation to all points in this data set in 
     * parallel. If the transform supports mutating the original data points, 
     * this will be applied if {@code mutableTransform} is set to {@code true}
     *
     * @param dt the transformation to apply
     * @param mutate {@code true} to mutableTransform the original data points,
     * {@code false} to ignore the ability to mutableTransform and replace the original
     * @param ex the threadpool to provide threads from. May be {@code null} to 
     * perform operations in serial 
     */
    public void applyTransform(final DataTransform dt, boolean mutate, ExecutorService ex)
    {
        final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
        if(ex == null)
            ex = new FakeExecutor();
        
        if (mutate && dt instanceof InPlaceTransform)
        {
            final InPlaceTransform ipt = (InPlaceTransform) dt;
            for(int id = 0; id < SystemInfo.LogicalCores; id++)
            {
                final int ID = id;
                ex.submit(new Runnable() 
                {
                    @Override
                    public void run()
                    {
                        for (int i = ID; i < getSampleSize(); i+=SystemInfo.LogicalCores)
                            ipt.mutableTransform(getDataPoint(i));
                        latch.countDown();
                    }
                });
            }
        }
        else
            for(int id = 0; id < SystemInfo.LogicalCores; id++)
            {
                final int ID = id;
                ex.submit(new Runnable() 
                {
                    @Override
                    public void run()
                    {
                        for (int i = ID; i < getSampleSize(); i+=SystemInfo.LogicalCores)
                            setDataPoint(i, dt.transform(getDataPoint(i)));
                        latch.countDown();
                    }
                });
            }
        try
        {
            latch.await();
            columnVecCache.clear();
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
        catch (InterruptedException ex1)
        {
            Logger.getLogger(DataSet.class.getName()).log(Level.SEVERE, null, ex1);
        }
    }
    
    /**
     * This method will replace every numeric feature in this dataset with a Vec
     * object from the given list. All vecs in the given list must be of the
     * same size.
     *
     * @param newNumericFeatures the list of new numeric features to use
     */
    public void replaceNumericFeatures(List<Vec> newNumericFeatures)
    {
        if(this.getSampleSize() != newNumericFeatures.size())
            throw new RuntimeException("Input list does not have the same not of dataums as the dataset");
        
        for(int i = 0; i < newNumericFeatures.size(); i++)
        {
            DataPoint dp_i = getDataPoint(i);
            setDataPoint(i, new DataPoint(newNumericFeatures.get(i), dp_i.getCategoricalValues(), dp_i.getCategoricalData(), dp_i.getWeight()));
        }
        
        this.numNumerVals = getDataPoint(0).numNumericalValues();
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
     * @param dp the data point to set at the specified index
     */
    abstract public void setDataPoint(int i, DataPoint dp);
    
    /**
     * Returns summary statistics computed in an online fashion for each numeric
     * variable. This returns all summary statistics, but can be less 
     * numerically stable and uses more memory. <br>
     * NaNs / missing values will be ignored in the statistics for each column. 
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
        
        /**
         * We got to skip nans, count their weight in each column so that we can still fast count zeros
         */
        double[] nanWeight = new double[numNumerVals];
        
        for(Iterator<DataPoint> iter = getDataPointIterator(); iter.hasNext(); )
        {
            DataPoint dp = iter.next();
            
            double weight = useWeights ? dp.getWeight() : 1;
            totalSoW += weight;

            Vec v = dp.getNumericalValues();
            for (IndexValue iv : v)
                if (Double.isNaN(iv.getValue()))//count it so we can fast count zeros right later
                    nanWeight[iv.getIndex()] += weight;
                else
                    stats[iv.getIndex()].add(iv.getValue(), weight);
        }
        
        double expected = totalSoW;
        //Add zero counts back in
        for(int i = 0; i < stats.length; i++)
            stats[i].add(0.0, expected-stats[i].getSumOfWeights()-nanWeight[i]);
        
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
     * Creates a new dataset that is a subset of this dataset. 
     * @param indicies the indices of data points to insert into the new 
     * dataset, and will be placed in the order listed. 
     * @return a new dataset that is a specified subset of this dataset, and 
     * backed by the same values
     */
    abstract protected Type getSubset(List<Integer> indicies);
    
    /**
     * This method returns a dataset that is a subset of this dataset, where
     * only the rows that have no missing values are kept. The new dataset is
     * backed by this dataset.
     *
     * @return a subset of this dataset that has all data points with missing 
     * features dropped
     */
    public Type getMissingDropped()
    {
        List<Integer> hasNoMissing = new IntList();
        for (int i = 0; i < getSampleSize(); i++)
        {
            DataPoint dp = getDataPoint(i);
            boolean missing =  dp.getNumericalValues().countNaNs() > 0;
            for(int c : dp.getCategoricalValues())
                if(c < 0)
                    missing = true;
            if(!missing)
                hasNoMissing.add(i);
        }
        return getSubset(hasNoMissing);
    }
    
    /**
     * Splits the dataset randomly into proportionally sized partitions. 
     *
     * @param rand the source of randomness for moving data around
     * @param splits any array, where the length is the number of datasets to
     * create and the value of in each index is the fraction of samples that
     * should be placed into that dataset. The sum of values must be less than
     * or equal to 1.0
     * @return a list of new datasets
     */
    public List<Type> randomSplit(Random rand, double... splits)
    {
        if(splits.length < 1)
            throw new IllegalArgumentException("Input array of split fractions must be non-empty");
        IntList randOrder = new IntList(getSampleSize());
        ListUtils.addRange(randOrder, 0, getSampleSize(), 1);
        Collections.shuffle(randOrder, rand);
        
        
        int[] stops = new int[splits.length];
        double sum = 0;
        for(int i = 0; i < splits.length; i++)
        {
            sum += splits[i];
            if(sum >= 1.001/*some flex room for numeric issues*/)
                throw new IllegalArgumentException("Input splits sum is greater than 1 by index " + i + " reaching a sum of " + sum);
            stops[i] = (int) Math.round(sum*randOrder.size());
        }
        
        List<Type> datasets = new ArrayList<Type>(splits.length);
        
        int prev = 0;
        for(int i = 0; i < stops.length; i++)
        {
            datasets.add(getSubset(randOrder.subList(prev, stops[i])));
            prev = stops[i];
        }
        
        return datasets;
    }
    
    /**
     * Splits the dataset randomly into proportionally sized partitions. 
     *
     * @param splits any array, where the length is the number of datasets to
     * create and the value of in each index is the fraction of samples that
     * should be placed into that dataset. The sum of values must be less than
     * or equal to 1.0
     * @return a list of new datasets
     */
    public List<Type> randomSplit(double... splits)
    {
        return randomSplit(RandomUtil.getRandom(), splits);
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
    public List<Type> cvSet(int folds, Random rand)
    {
        double[] splits = new double[folds];
        Arrays.fill(splits, 1.0/folds);
        return randomSplit(rand, splits);
    }
    
    /**
     * Creates <tt>folds</tt> data sets that contain data from this data set. 
     * The data points in each set will be random. These are meant for cross 
     * validation
     * 
     * @param folds the number of cross validation sets to create. Should be greater then 1
     * @return the list of data sets. 
     */
    public List<Type> cvSet(int folds)
    {
        return cvSet(folds, RandomUtil.getRandom());
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

        SoftReference<Vec> cachedRef = columnVecCache.get(i);
        if (cachedRef != null)
        {
            Vec v = cachedRef.get();
            if (v != null)
                return v;
        }
        //no cache, so make it
        DenseVector dv = new DenseVector(getSampleSize());
        for (int j = 0; j < getSampleSize(); j++)
            dv.set(j, getDataPoint(j).getNumericalValues().get(i));
        Vec toRet;
        if (getSparsityStats().getMean() < 0.6)
            toRet = new SparseVector(dv);
        else
            toRet = dv;
        columnVecCache.put(i, new SoftReference<Vec>(toRet));
        return toRet;
    }
    
    /**
     * 
     * @return the number of missing values in both numeric and categorical features
     */
    public long countMissingValues()
    {
        long missing = 0;
        for (int i = 0; i < getSampleSize(); i++)
        {
            DataPoint dp = getDataPoint(i);
            missing += dp.getNumericalValues().countNaNs();
            for(int c : dp.getCategoricalValues())
                if(c < 0)
                    missing++;
        }
        return missing;
    }
    
    /**
     * Creates an array of column vectors for every numeric variable in this 
     * data set. The index of the array corresponds to the numeric feature 
     * index. This method is faster and more efficient than calling 
     * {@link #getNumericColumn(int) } when multiple columns are needed. <br>
     * <br>
     * Note, that the columns returned by this method may be cached and re used
     * by the DataSet itself. If you need to alter the columns you should create
     * your own copy of these vectors. If you know that you will be the only 
     * person getting a column vector from this data set, then you may safely 
     * alter the columns without mutating the data points themselves. However, 
     * future callers may or may not receive the same vector objects. 
     * 
     * @return an array of the column vectors
     */
    @SuppressWarnings("unchecked")
    public Vec[] getNumericColumns()
    {
        return getNumericColumns(Collections.EMPTY_SET);
    }
    
    /**
     * Creates an array of column vectors for every numeric variable in this 
     * data set. The index of the array corresponds to the numeric feature 
     * index. This method is faster and more efficient than calling 
     * {@link #getNumericColumn(int) } when multiple columns are needed. <br>
     * <br>
     * A set of columns to skip can be provided in order to save memory if one 
     * does not need all the columns. <br>
     * <br>
     * Note, that the columns returned by this method may be cached and re used
     * by the DataSet itself. If you need to alter the columns you should create
     * your own copy of these vectors. If you know that you will be the only 
     * person getting a column vector from this data set, then you may safely 
     * alter the columns without mutating the data points themselves. However, 
     * future callers may or may not receive the same vector objects. 
     * 
     * @param skipColumns if a column's index is in this set, a {@code null} 
     * will be returned in the array at the column's index instead of a vector
     * 
     * @return an array of the column vectors
     */
    public Vec[] getNumericColumns(Set<Integer> skipColumns)
    {
        boolean sparse = getSparsityStats().getMean() < 0.6;
        Vec[] cols = new Vec[getNumNumericalVars()];
        boolean[] dontSet = new boolean [cols.length];
        Arrays.fill(dontSet, false);
        for(int i = 0; i < cols.length; i++)
            if(!skipColumns.contains(i))
            {
                SoftReference<Vec> cachedRef = columnVecCache.get(i);
                if(cachedRef != null )
                {
                    Vec v = cachedRef.get();
                    if(v != null)
                    {
                        cols[i] = v;
                        dontSet[i] = true;
                    }
                    else
                        columnVecCache.put(i, new SoftReference<Vec>(cols[i] = sparse ? new SparseVector(getSampleSize()) : new DenseVector(getSampleSize())));
                }
                else
                    columnVecCache.put(i, new SoftReference<Vec>(cols[i] = sparse ? new SparseVector(getSampleSize()) : new DenseVector(getSampleSize())));
            }
        for(int i = 0; i < getSampleSize(); i++)
        {
            Vec v = getDataPoint(i).getNumericalValues();
            
            for(IndexValue iv : v)
            {
                int col = iv.getIndex();
                if(cols[col] != null && !dontSet[col])
                    cols[col].set(i, iv.getValue());
            }
        }
            
        return cols;
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
     * Creates a matrix backed by the data set, where each row is a data point 
     * from the dataset, and each column is one of the numeric examples from the
     * data set. <br>
     * Any modifications to this matrix will be reflected in the dataset. <br>
     * This method has the advantage over {@link #getDataMatrix() } in that it 
     * does not use any additional memory and it maintains any sparsity 
     * information. 
     * @return a matrix representation of the data points 
     */
    public Matrix getDataMatrixView()
    {
        return new MatrixOfVecs(getDataVectors());
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
    abstract public DataSet<Type> shallowClone();
    
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
    
    /**
     * Returns statistics on the sparsity of the vectors in this data set. 
     * Vectors that are not considered sparse will be treated as completely 
     * dense, even if zero values exist in the data. 
     * 
     * @return an object containing the statistics of the vector sparsity
     */
    public OnLineStatistics getSparsityStats()
    {
        OnLineStatistics stats = new OnLineStatistics();
        for(int i = 0; i < getSampleSize(); i++)
        {
            Vec v = getDataPoint(i).getNumericalValues();
            if(v.isSparse())
                stats.add(v.nnz() / (double)v.length());
            else
                stats.add(1.0);
        }
        
        return stats;
    }

    /**
     * This method returns the weight of each data point in a single Vector.
     * When all data points have the same weight, this will return a vector that
     * uses fixed memory instead of allocating a full double backed array.
     *
     * @return a vector that will return the weight for each data point with the
     * same corresponding index.
     */
    public Vec getDataWeights()
    {
        final int N = this.getSampleSize();
        if(N == 0)
            return new DenseVector(0);
        //assume everyone has the same weight until proven otherwise.
        double weight = getDataPoint(0).getWeight();
        double[] weights = null;
        
        for(int i = 1; i < N; i++)
        {
            double w_i = getDataPoint(i).getWeight();
            if(weights != null || weight != w_i)
            {
                if(weights==null)//need to init storage place
                {
                    weights = new double[N];
                    Arrays.fill(weights, 0, i, weight);
                }
                weights[i] = w_i;
            }
        }
        
        if(weights == null)
            return new ConstantVector(weight, getSampleSize());
        else
            return new DenseVector(weights);
    }
}
