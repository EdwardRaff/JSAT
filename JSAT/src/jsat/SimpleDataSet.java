
package jsat;

import java.util.ArrayList;
import java.util.List;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;

/**
 * SimpleData Set is a basic implementation of a data set. Has no assumptions about the task that is going to be performed. 
 * 
 * @author Edward Raff
 */
public class SimpleDataSet extends DataSet<SimpleDataSet>
{
    protected List<DataPoint> dataPoints;

    public SimpleDataSet(List<DataPoint> dataPoints)
    {
        if(dataPoints.isEmpty())
            throw new RuntimeException("Can not create empty data set");
        this.dataPoints = dataPoints;
        this.categories =  dataPoints.get(0).getCategoricalData();
        this.numNumerVals = dataPoints.get(0).numNumericalValues();
        this.numericalVariableNames = new ArrayList<String>(this.numNumerVals);
        for(int i = 0; i < getNumNumericalVars(); i++)
            this.numericalVariableNames.add("Numeric Input " + (i+1));
    }

    public SimpleDataSet(CategoricalData[] categories, int numNumericalValues)
    {
        this.categories = categories;
        this.numNumerVals = numNumericalValues;
        this.dataPoints = new ArrayList<DataPoint>();
    }
    
    @Override
    public DataPoint getDataPoint(int i)
    {
        return dataPoints.get(i);
    }

    @Override
    public void setDataPoint(int i, DataPoint dp)
    {
        dataPoints.set(i, dp);
        columnVecCache.clear();
    }
    
    /**
     * Adds a new datapoint to this set. 
     * @param dp the datapoint to add
     */
    public void add(DataPoint dp)
    {
        dataPoints.add(dp);
        columnVecCache.clear();
    }

    @Override
    public int getSampleSize()
    {
        return dataPoints.size();
    }
    
    @Override
    protected SimpleDataSet getSubset(List<Integer> indicies)
    {
        SimpleDataSet newData = new SimpleDataSet(categories, numNumerVals);
        for(int i : indicies)
            newData.add(getDataPoint(i));
        return newData;
    }
    
    /**
     * Converts this dataset into one meant for classification problems. The 
     * given categorical feature index is removed from the data and made the
     * target variable for the classification problem.
     *
     * @param index the classification variable index, should be in the range
     * [0, {@link #getNumCategoricalVars() })
     * @return a new dataset where one categorical variable is removed and made
     * the target of a classification dataset
     */
    public ClassificationDataSet asClassificationDataSet(int index)
    {
        if(index < 0)
            throw new IllegalArgumentException("Index must be a non-negative value");
        else if(getNumCategoricalVars() == 0)
            throw new IllegalArgumentException("Dataset has no categorical variables, can not create classification dataset");
        else if(index >= getNumCategoricalVars())
            throw new IllegalArgumentException("Index " + index + " is larger than number of categorical features " + getNumCategoricalVars());
        return new ClassificationDataSet(this, index);
    }
    
    /**
     * Converts this dataset into one meant for regression problems. The 
     * given numeric feature index is removed from the data and made the
     * target variable for the regression problem.
     *
     * @param index the regression variable index, should be in the range
     * [0, {@link #getNumNumericalVars() })
     * @return a new dataset where one numeric variable is removed and made
     * the target of a regression dataset
     */
    public RegressionDataSet asRegressionDataSet(int index)
    {
        if(index < 0)
            throw new IllegalArgumentException("Index must be a non-negative value");
        else if(getNumNumericalVars()== 0)
            throw new IllegalArgumentException("Dataset has no numeric variables, can not create regression dataset");
        else if(index >= getNumNumericalVars())
            throw new IllegalArgumentException("Index " + index + " i larger than number of numeric features " + getNumNumericalVars());
        
        return new RegressionDataSet(this.dataPoints, index);
    }
    
    
    /**
     * 
     * @return direct access to the list that backs this data set. 
     */
    public List<DataPoint> getBackingList()
    {
        return dataPoints;
    }

    @Override
    public SimpleDataSet shallowClone()
    {
        return new SimpleDataSet(new ArrayList<DataPoint>(this.dataPoints));
    }

    @Override
    public SimpleDataSet getTwiceShallowClone()
    {
        return (SimpleDataSet) super.getTwiceShallowClone();
    }
    
    
}
