
package jsat.regression;

import java.util.*;
import jsat.DataSet;
import jsat.DataStore;
import jsat.RowMajorStore;
import jsat.classifiers.*;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.utils.DoubleList;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * A RegressionDataSet is a data set specifically for the task of performing regression. 
 * Each data point is paired with s double value that indicates its true regression value.
 * An example of a regression problem would be mapping the inputs of a function to its 
 * outputs, and attempting to learn the function from the samples. 
 * 
 * @author Edward Raff
 */
public class RegressionDataSet extends DataSet<RegressionDataSet>
{

    protected DoubleList targets;
    
    /**
     * Creates a new empty data set for regression 
     * 
     * @param numerical the number of numerical attributes that will be used, excluding the regression value
     * @param categories an array of length equal to the number of categorical attributes, each object describing the attribute in question
     */
    public RegressionDataSet(int numerical, CategoricalData[] categories)
    {
        super(numerical, categories);
        targets = new DoubleList();
    }
    
    /**
     * Creates a new dataset containing the given points paired with their
     * target values. Pairing is determined by the iteration order of each
     * collection.
     *
     * @param datapoints the DataStore that will back this Data Set
     * @param targets the target values to use
     */
    public RegressionDataSet(DataStore datapoints, List<Double> targets)
    {
       super(datapoints);
       this.targets = new DoubleList(targets);
    }
    
    /**
     * Creates a new data set for the given list of data points. The data 
     * points will be copied, changes in one will not effect the other. 
     * 
     * @param data the list of data point to create a data set from
     * @param predicting which of the numerical attributes is the 
     * regression target. Categorical attributes are ignored in 
     * the count of attributes for this value. 
     */
    public RegressionDataSet(List<DataPoint> data, int predicting)
    {
        super(data.get(0).numNumericalValues()-1, data.get(0).getCategoricalData());
        //Use the first data point to set up
        DataPoint tmp = data.get(0);
        categories = new CategoricalData[tmp.numCategoricalValues()];
        System.arraycopy(tmp.getCategoricalData(), 0, categories, 0, categories.length);
        targets = new DoubleList(data.size());
        
        //Fill up data
        for(DataPoint dp : data)
        {
            Vec origV = dp.getNumericalValues();
            Vec newVec;
            double target = 0;//init to zero to inplicitly handle sparse feature vector case
            if (origV.isSparse())
                newVec = new SparseVector(origV.length() - 1, origV.nnz());
            else
                newVec = new DenseVector(origV.length() - 1);

            for (IndexValue iv : origV)
                if (iv.getIndex() < predicting)
                    newVec.set(iv.getIndex(), iv.getValue());
                else if (iv.getIndex() == predicting)
                    target = iv.getValue();
                else//iv.getIndex() > index
                    newVec.set(iv.getIndex() - 1, iv.getValue());

            DataPoint newDp = new DataPoint(newVec, dp.getCategoricalValues(), categories);
            
            datapoints.addDataPoint(newDp);
            targets.add(target);
        }
    }
    
    /**
     * Creates a new regression data set by copying all the data points 
     * in the given list. Alterations to this list will not effect this DataSet. 
     * @param list source of data points to copy
     */
    public RegressionDataSet(List<DataPointPair<Double>> list)
    {
        super(list.get(0).getDataPoint().numNumericalValues(), CategoricalData.copyOf(list.get(0).getDataPoint().getCategoricalData()));
        this.datapoints = new RowMajorStore(numNumerVals, categories);
        this.targets = new DoubleList();
        for(DataPointPair<Double> dpp : list)
        {
            datapoints.addDataPoint(dpp.getDataPoint());
            targets.add(dpp.getPair());
        }
    }
    
    private RegressionDataSet()
    {
        super(new RowMajorStore(1, new CategoricalData[0]));
    }
    
    public static RegressionDataSet comineAllBut(List<RegressionDataSet> list, int exception)
    {
        int numer = list.get(exception).getNumNumericalVars();
        CategoricalData[] categories = list.get(exception).getCategories();
        
        RegressionDataSet rds = new RegressionDataSet(numer, categories);

        //The list of data sets
        for (int i = 0; i < list.size(); i++)
            if (i == exception)
                continue;
            else
                for(int j = 0; j < list.get(i).size(); j++)
                    rds.addDataPoint(list.get(i).getDataPoint(j), list.get(i).getTargetValue(j));
        
        return rds;
    }
    
    private static final int[] emptyInt = new int[0];
    /**
     * Creates a new data point with no categorical variables to be added to the
     * data set. The arguments will be used directly, modifying them after will
     * effect the data set. 
     * 
     * @param numerical the numerical values for the data point
     * @param val the taret value
     * @throws IllegalArgumentException if the given values are inconsistent with the data this class stores. 
     */
    public void addDataPoint(Vec numerical, double val)
    {
        addDataPoint(numerical, emptyInt, val);
    }
    
    /**
     * Creates a new data point to be added to the data set. The arguments will
     * be used directly, modifying them after will effect the data set. 
     * 
     * @param numerical the numerical values for the data point
     * @param categories the categorical values for the data point
     * @param val the target value to predict
     * @throws IllegalArgumentException if the given values are inconsistent with the data this class stores. 
     */
    public void addDataPoint(Vec numerical, int[] categories, double val)
    {
        if(numerical.length() != numNumerVals)
            throw new RuntimeException("Data point does not contain enough numerical data points");
        if(categories.length != categories.length)
            throw new RuntimeException("Data point does not contain enough categorical data points");
        
        for(int i = 0; i < categories.length; i++)
            if(!this.categories[i].isValidCategory(categories[i]) && categories[i] >= 0) // >= so that missing values (negative) are allowed
                throw new RuntimeException("Categoriy value given is invalid");
        
        DataPoint dp = new DataPoint(numerical, categories, this.categories);
        addDataPoint(dp, val);
    }
    
    /**
     * 
     * @param dp the data to add
     * @param val the target value for this data point
     */
    public void addDataPoint(DataPoint dp, double val)
    {
	addDataPoint(dp, val, 1.0);
    }
    
    /**
     * 
     * @param dp the data to add
     * @param val the target value for this data point
     * @param weight the weight for this data point
     */
    public void addDataPoint(DataPoint dp, double val, double weight)
    {
        if(dp.numNumericalValues() != getNumNumericalVars() || dp.numCategoricalValues() != getNumCategoricalVars())
            throw new RuntimeException("The added data point does not match the number of values and categories for the data set");
        else if(Double.isInfinite(val) || Double.isNaN(val))
            throw new ArithmeticException("Unregressiable value " + val + " given for regression");
        
        datapoints.addDataPoint(dp);
        targets.add(val);
	setWeight(size()-1, weight);
    }
    
    public void addDataPointPair(DataPointPair<Double> pair)
    {
        addDataPoint(pair.getDataPoint(), pair.getPair());
    }
    
    /**
     * Returns the i'th data point in the data set paired with its target regressor value.
     * Modifying the DataPointPair will effect the data set. 
     * 
     * @param i the index of the data point to obtain
     * @return the i'th DataPOintPair
     */
    public DataPointPair<Double> getDataPointPair(int i)
    {
        return new DataPointPair<>(getDataPoint(i), targets.get(i));
    }
    
    /**
     * Returns a new list containing copies of the data points in this data set, 
     * paired with their regression target values. MModifications to the list 
     * or data points will not effect this data set
     * 
     * @return a list of copies of the data points in this set
     */
    public List<DataPointPair<Double>> getAsDPPList()
    {
        ArrayList<DataPointPair<Double>> list = new ArrayList<>(size());
        for(int i = 0; i < size(); i++)
            list.add(new DataPointPair<>(getDataPoint(i).clone(), targets.get(i)));
        return list;
    }
    
    /**
     * Returns a new list containing the data points in this data set, paired with 
     * their regression target values. Modifications to the list will not effect 
     * the data set, but modifying the points will. For a copy of the points, use
     * the {@link #getAsDPPList() } method. 
     * 
     * @return a list of the data points in this set
     */
    public List<DataPointPair<Double>> getDPPList()
    {
        ArrayList<DataPointPair<Double>> list = new ArrayList<>(size());
        for(int i = 0; i < size(); i++)
            list.add(getDataPointPair(i));
        return list;
    }
    
    /**
     * Sets the target regression value associated with a given data point
     * @param i the index in the data set
     * @param val the new target value
     * @throws ArithmeticException if <tt>val</tt> is infinite or NaN
     */
    public void setTargetValue(int i, double val)
    {
        if(Double.isInfinite(val) || Double.isNaN(val))
            throw new ArithmeticException("Can not predict a " + val + " value");
        targets.set(i, val);
    }

    @Override
    protected RegressionDataSet getSubset(List<Integer> indicies)
    {
        RegressionDataSet newData = new RegressionDataSet(numNumerVals, categories);
        for (int i : indicies)
            newData.addDataPoint(getDataPoint(i), getTargetValue(i));
        return newData;
    }
    
    /**
     * Returns a vector containing the target regression values for each 
     * data point. The vector is a copy, and modifications to it will not
     * effect the data set. 
     * 
     * @return a vector containing the target values for each data point
     */
    public Vec getTargetValues()
    {
        DenseVector vals = new DenseVector(size());
        
        for(int i = 0; i < size(); i++)
            vals.set(i, targets.getD(i));
        
        return vals;
    }
    
    /**
     * Returns the target regression value for the <tt>i</tt>'th data point in the data set. 
     * 
     * @param i the data point to get the regression value of 
     * @return the target regression value
     */
    public double getTargetValue(int i)
    {
        return targets.getD(i);
    }
    
    /**
     * Creates a new data set that uses the given list as its backing list. 
     * No copying is done, and changes to this list will be reflected in 
     * this data set, and the other way. 
     * 
     * @param list the list of datapoint to back a new data set with
     * @return a new data set
     */
    public static RegressionDataSet usingDPPList(List<DataPointPair<Double>> list)
    {
        return new RegressionDataSet(list);
    }

    @Override
    public RegressionDataSet shallowClone()
    {
        RegressionDataSet clone = new RegressionDataSet(numNumerVals, categories);
        for(int i = 0; i < size(); i++)
            clone.addDataPointPair(getDataPointPair(i));
        return clone;
    }

    @Override
    public RegressionDataSet emptyClone()
    {
	return new RegressionDataSet(numNumerVals, categories);
    }

    @Override
    public RegressionDataSet getTwiceShallowClone()
    {
        return (RegressionDataSet) super.getTwiceShallowClone(); //To change body of generated methods, choose Tools | Templates.
    }
}
