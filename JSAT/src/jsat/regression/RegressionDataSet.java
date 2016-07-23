
package jsat.regression;

import java.util.*;
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
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

    /**
     * The list of all data points, paired with their true regression output
     */
    protected List<DataPointPair<Double>> dataPoints;
    
    /**
     * Creates a new empty data set for regression 
     * 
     * @param numerical the number of numerical attributes that will be used, excluding the regression value
     * @param categories an array of length equal to the number of categorical attributes, each object describing the attribute in question
     */
    public RegressionDataSet(int numerical, CategoricalData[] categories)
    {
        this.numNumerVals = numerical;
        this.categories = categories;
        dataPoints = new ArrayList<DataPointPair<Double>>();
        this.numericalVariableNames = new ArrayList<String>(getNumNumericalVars());
        setUpGenericNumericNames();
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
        //Use the first data point to set up
        DataPoint tmp = data.get(0);
        categories = new CategoricalData[tmp.numCategoricalValues()];
        System.arraycopy(tmp.getCategoricalData(), 0, categories, 0, categories.length);
        
        
        numNumerVals = tmp.numNumericalValues()-1;
        
        dataPoints = new ArrayList<DataPointPair<Double>>(data.size());
        
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

            DataPoint newDp = new DataPoint(newVec, dp.getCategoricalValues(), categories, dp.getWeight());
            DataPointPair<Double> dpp = new DataPointPair<Double>(newDp, target);
            
            dataPoints.add(dpp);
        }
        
        this.numericalVariableNames = new ArrayList<String>(getNumNumericalVars());
        setUpGenericNumericNames();
    }
    
    /**
     * Creates a new regression data set by copying all the data points 
     * in the given list. Alterations to this list will not effect this DataSet. 
     * @param list source of data points to copy
     */
    public RegressionDataSet(List<DataPointPair<Double>> list)
    {
        this.numNumerVals = list.get(0).getDataPoint().numNumericalValues();
        this.numericalVariableNames = new ArrayList<String>(getNumNumericalVars());
        setUpGenericNumericNames();
        this.categories = CategoricalData.copyOf(list.get(0).getDataPoint().getCategoricalData());
        this.dataPoints =new ArrayList<DataPointPair<Double>>(list.size());
        for(DataPointPair<Double> dpp : list)
            dataPoints.add(new DataPointPair<Double>(dpp.getDataPoint().clone(), dpp.getPair()));
    }

    /**
     * Sets all the names of the numeric variables 
     */
    private void setUpGenericNumericNames()
    {
        if(getNumNumericalVars() > 100)
            return;
        for(int i = 0; i < getNumNumericalVars(); i++)
            this.numericalVariableNames.add("Numeric Input " + (i+1));
    }
    
    private RegressionDataSet()
    {
        
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
                rds.dataPoints.addAll(list.get(i).dataPoints);
        
        return rds;
    }
    
    private static final int[] emptyInt = new int[0];
    /**
     * Creates a new data point with no categorical variables to be added to the
     * data set. The arguments will be used directly, modifying them after will
     * effect the data set. 
     * 
     * @param numerical the numerical values for the data point
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
    
    public void addDataPoint(DataPoint dp, double val)
    {
        if(dp.numNumericalValues() != getNumNumericalVars() || dp.numCategoricalValues() != getNumCategoricalVars())
            throw new RuntimeException("The added data point does not match the number of values and categories for the data set");
        else if(Double.isInfinite(val) || Double.isNaN(val))
            throw new ArithmeticException("Unregressiable value " + val + " given for regression");
        
        DataPointPair<Double> dpp = new DataPointPair<Double>(dp, val);
        dataPoints.add(dpp);
        columnVecCache.clear();
    }
    
    public void addDataPointPair(DataPointPair<Double> pair)
    {
        dataPoints.add(pair);
        columnVecCache.clear();
    }
    
    @Override
    public DataPoint getDataPoint(int i)
    {
        return dataPoints.get(i).getDataPoint();
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
        return dataPoints.get(i);
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
        ArrayList<DataPointPair<Double>> list = new ArrayList<DataPointPair<Double>>(dataPoints.size());
        for(DataPointPair<Double> dpp : dataPoints)
            list.add(new DataPointPair<Double>(dpp.getDataPoint().clone(), dpp.getPair()));
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
        ArrayList<DataPointPair<Double>> list = new ArrayList<DataPointPair<Double>>(dataPoints);
        
        return list;
    }
    
    @Override
    public void setDataPoint(int i, DataPoint dp)
    {
        dataPoints.get(i).setDataPoint(dp);
        columnVecCache.clear();
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
        dataPoints.get(i).setPair(val);
    }

    @Override
    protected RegressionDataSet getSubset(List<Integer> indicies)
    {
        RegressionDataSet newData = new RegressionDataSet(numNumerVals, categories);
        for (int i : indicies)
            newData.addDataPoint(getDataPoint(i), getTargetValue(i));
        return newData;
    }
    
    @Override
    public int getSampleSize()
    {
        return dataPoints.size();
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
        DenseVector vals = new DenseVector(getSampleSize());
        
        for(int i = 0; i < getSampleSize(); i++)
            vals.set(i, dataPoints.get(i).getPair());
        
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
        return dataPoints.get(i).getPair();
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
        RegressionDataSet rds = new RegressionDataSet();
        rds.dataPoints = list;
        rds.numNumerVals = list.get(0).getDataPoint().numNumericalValues();
        rds.numericalVariableNames = new ArrayList<String>(rds.getNumNumericalVars());
        for(int i = 0; i < rds.getNumNumericalVars(); i++)
            rds.numericalVariableNames.add("Numeric Input " + (i+1));
        rds.categories = CategoricalData.copyOf(list.get(0).getDataPoint().getCategoricalData());
        return rds;
    }

    @Override
    public RegressionDataSet shallowClone()
    {
        RegressionDataSet clone = new RegressionDataSet(numNumerVals, categories);
        for(DataPointPair<Double> dpp : this.dataPoints)
            clone.dataPoints.add(new DataPointPair<Double>(dpp.getDataPoint(), dpp.getPair()));
        clone.columnVecCache.putAll(this.columnVecCache);
        return clone;
    }

    @Override
    public RegressionDataSet getTwiceShallowClone()
    {
        return (RegressionDataSet) super.getTwiceShallowClone(); //To change body of generated methods, choose Tools | Templates.
    }
}
