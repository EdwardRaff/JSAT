
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
   
    /**
     * Creates a new dataset containing the given datapoints. 
     * 
     * @param datapoints the collection of data points to create a dataset from
     */
    public SimpleDataSet(List<DataPoint> datapoints)
    {
        super(datapoints.get(0).numNumericalValues(), datapoints.get(0).getCategoricalData());
        for(DataPoint dp : datapoints)
            this.add(dp);
    }
    
    /**
     * Creates a new dataset containing the given datapoints. The number of
     * features and categorical data information will be obtained from the
     * DataStore.
     *
     * @param datapoints the collection of data points to create a dataset from
     */
    public SimpleDataSet(DataStore datapoints) 
    {
        super(datapoints);
    }

    /**
     * Creates a new empty data set
     *
     * @param numerical the number of numerical features for points in this
     * dataset
     * @param categories the information and number of categorical features in
     * this dataset
     */
    public SimpleDataSet(int numerical, CategoricalData[] categories)
    {
        super(numerical, categories);
    }
    
    /**
     * Adds a new datapoint to this set. 
     * @param dp the datapoint to add
     */
    public void add(DataPoint dp)
    {
        base_add(dp, 1.0);
    }
    
    @Override
    protected SimpleDataSet getSubset(List<Integer> indicies)
    {
        SimpleDataSet newData = new SimpleDataSet(numNumerVals, categories);
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
        
        RegressionDataSet rds = new RegressionDataSet(this.datapoints.toList(), index);
	for(int i = 0; i < size(); i++)
	    rds.setWeight(i, this.getWeight(i));
	return rds;
    }
    
        /**
     *
     * @return access to a list of the list that backs this data set. May or may
     * not be backed by the original data.
     */
    public List<DataPoint> getList() {
        return datapoints.toList();
    }

    @Override
    public SimpleDataSet shallowClone()
    {
        return new SimpleDataSet(new ArrayList<>(this.datapoints.toList()));
    }

    @Override
    public SimpleDataSet emptyClone()
    {
	SimpleDataSet sds = new SimpleDataSet(numNumerVals, categories);
	return sds;
    }

    @Override
    public SimpleDataSet getTwiceShallowClone()
    {
        return (SimpleDataSet) super.getTwiceShallowClone();
    }
    
    
}
