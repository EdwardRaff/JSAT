
package jsat.classifiers;

import java.util.*;
import jsat.DataSet;
import jsat.DataStore;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * ClassificationDataSet is a data set meant specifically for classification problems. 
 * The true class of each data point is stored separately from the data point, so that
 * it can be feed into a learning algorithm and not interfere. 
 * <br>
 * Additional functionality specific to classification problems is also available. 
 * @author Edward Raff
 */
public class ClassificationDataSet extends DataSet<ClassificationDataSet>
{
    /**
     * The categories for the predicted value
     */
    protected CategoricalData predicting;
    /**
     * the target values
     */
    protected IntList targets;
    
    /**
     * Creates a new data set for classification problems. 
     * 
     * @param dataSet the source data set
     * @param predicting the categorical attribute to use as the target class
     */
    public ClassificationDataSet(DataSet dataSet, int predicting)
    {
        this(dataSet.getDataPoints(), predicting);
        //Fix up numeric names
        for(int i = 0; i < getNumNumericalVars(); i++)
            this.numericalVariableNames.put(i, dataSet.getNumericName(i));
	for(int i = 0; i < dataSet.size(); i++)
	    this.setWeight(i, dataSet.getWeight(i));
    }

    /**
     * Creates a new data set for classification problems from the given list of data points. It is assume the data points are consistent. 
     * @param data the list of data points for the problem. 
     * @param predicting the categorical attribute to use as the target class
     */
    public ClassificationDataSet(List<DataPoint> data, int predicting)
    {
        super(data.get(0).numNumericalValues(), data.get(0).getCategoricalData());//we will fix categoricl data in a sec
        //Use the first data point to set up
        DataPoint tmp = data.get(0);
        categories = new CategoricalData[tmp.numCategoricalValues()-1];
        for(int i = 0; i < categories.length; i++)
        {
            categories[i] = i >= predicting ? 
                    tmp.getCategoricalData()[i+1] : tmp.getCategoricalData()[i];
        }
        
        //re-set DataStore with fixed categories
        this.datapoints.setCategoricalDataInfo(categories);
        
        this.predicting = tmp.getCategoricalData()[predicting];
        targets = new IntList(data.size());
        
        
        //Fill up data
        for(DataPoint dp : data)
        {
            int[] newCats = new int[dp.numCategoricalValues()-1];
            int[] prevCats = dp.getCategoricalValues();
            int k = 0;//index for the newCats 
            for(int i = 0; i < prevCats.length; i++)
            {
                if(i != predicting)
                    newCats[k++] = prevCats[i];
            }
            DataPoint newPoint = new DataPoint(dp.getNumericalValues(), newCats, categories);
            datapoints.addDataPoint(newPoint);
            targets.add(prevCats[predicting]);
        }
    }
    
        
    /**
     * Creates a new dataset containing the given points paired with their
     * target values. Pairing is determined by the iteration order of each
     * collection.<br>
     * It is assumed that all options for the target variable are contained in
     * the given targets list.
     *
     *
     * @param datapoints the DataStore that will back this Data Set
     * @param targets the target values to use
     */
    public ClassificationDataSet(DataStore datapoints, List<Integer> targets)
    {
        this(datapoints, targets, new CategoricalData(targets.stream().mapToInt(i->i).max().getAsInt()+1));
    }
    
    
    /**
     * Creates a new dataset containing the given points paired with their
     * target values. Pairing is determined by the iteration order of each
     * collection.
     *
     * @param datapoints the DataStore that will back this Data Set
     * @param targets the target values to use
     * @param predicting the information about the target attribute
     */
    public ClassificationDataSet(DataStore datapoints, List<Integer> targets, CategoricalData predicting)
    {
        super(datapoints);
        this.targets = new IntList(targets);
        this.predicting = predicting;
    }
    
    /**
     * Creates a new data set for classification problems from the given list of data points. 
     * The class value is paired with each data point. 
     * 
     * @param data the list of data points, paired with their class values
     * @param predicting the information about the target class
     */
    public ClassificationDataSet(List<DataPointPair<Integer>> data, CategoricalData predicting)
    {
        super(data.get(0).getVector().length(), data.get(0).getDataPoint().getCategoricalData());
        this.predicting = predicting;
        categories = CategoricalData.copyOf(data.get(0).getDataPoint().getCategoricalData());
        targets = new IntList(data.size());
        for(DataPointPair<Integer> dpp : data)
        {
            datapoints.addDataPoint(dpp.getDataPoint());
            targets.add(dpp.getPair());
        }
    }
    
    /**
     * Creates a new, empty, data set for classification problems. 
     * 
     * @param numerical the number of numerical attributes for the problem
     * @param categories the information about each categorical variable in the problem. 
     * @param predicting the information about the target class
     */
    public ClassificationDataSet(int numerical, CategoricalData[] categories, CategoricalData predicting)
    {
        super(numerical, categories);
        this.predicting = predicting;
        targets = new IntList();
    }
    
    /**
     * Returns the number of target classes in this classification data set. This value 
     * can also be obtained by calling {@link #getPredicting() getPredicting()}.
     * {@link CategoricalData#getNumOfCategories() getNumOfCategories() }
     * @return the number of target classes for prediction
     */
    public int getClassSize()
    {
        return predicting.getNumOfCategories();
    }
        
    /**
     * A helper method meant to be used with {@link #cvSet(int) }, this combines all 
     * classification data sets in a given list, but holding out the indicated list. 
     * 
     * @param list a list of data sets
     * @param exception the one data set in the list NOT to combine into one file
     * @return a combination of all the data sets in <tt>list</tt> except the one at index <tt>exception</tt>
     */
    public static ClassificationDataSet comineAllBut(List<ClassificationDataSet> list, int exception)
    {
        int numer = list.get(exception).getNumNumericalVars();
        CategoricalData[] categories = list.get(exception).getCategories();
        CategoricalData predicting = list.get(exception).getPredicting();
        
        ClassificationDataSet cds = new ClassificationDataSet(numer, categories, predicting);
        
        //The list of data sets
        for(int i = 0; i < list.size(); i++)
        {
            if(i == exception)
                continue;
            for(int j = 0; j < list.get(i).size(); j++)
                cds.datapoints.addDataPoint(list.get(i).getDataPoint(j));
            cds.targets.addAll(list.get(i).targets);
        }
        
        return cds;
    }
    
    /**
     * Returns the i'th data point from the data set
     * @param i the i'th data point in this set
     * @return the ith data point in this set
     */
    @Override
    public DataPoint getDataPoint(int i)
    {
        return getDataPointPair(i).getDataPoint();
    }
    
    /**
     * Returns the i'th data point from the data set, paired with the integer indicating its true class
     * @param i the i'th data point in this set
     * @return the i'th data point from the data set, paired with the integer indicating its true class
     */
    public DataPointPair<Integer> getDataPointPair(int i)
    {
        if(i >= size())
            throw new IndexOutOfBoundsException("There are not that many samples in the data set");
        
        return new DataPointPair<>(datapoints.getDataPoint(i), targets.getI(i));
    }
    
    @Override
    public void setDataPoint(int i, DataPoint dp)
    {
        if(i >= size())
            throw new IndexOutOfBoundsException("There are not that many samples in the data set");
        datapoints.setDataPoint(i, dp);
    }
    
    /**
     * Returns the integer value corresponding to the true category of the <tt>i</tt>'th data point. 
     * @param i the <tt>i</tt>'th data point. 
     * @return the integer value for the category of the <tt>i</tt>'th data point. 
     * @throws IndexOutOfBoundsException if <tt>i</tt> is not a valid index into the data set. 
     */
    public int getDataPointCategory(int i)
    {
        if(i >= size())
            throw new IndexOutOfBoundsException("There are not that many samples in the data set: " + i);
        else if(i < 0)
            throw new IndexOutOfBoundsException("Can not specify negative index " + i);
        
        return targets.get(i);
    }
    
    @Override
    protected ClassificationDataSet getSubset(List<Integer> indicies)
    {
        ClassificationDataSet newData = new ClassificationDataSet(numNumerVals, categories, predicting);
        for (int i : indicies)
            newData.addDataPoint(getDataPoint(i), getDataPointCategory(i));
        return newData;
    }
    
 
    public List<ClassificationDataSet> stratSet(int folds, Random rnd)
    {
        ArrayList<ClassificationDataSet> cvList = new ArrayList<>();
        
        while (cvList.size() < folds)
        {
            ClassificationDataSet clone = new ClassificationDataSet(numNumerVals, categories, predicting.clone());
            cvList.add(clone);
        }
        
        IntList rndOrder = new IntList();
        
        int curFold = 0;
        for(int c = 0; c < getClassSize(); c++)
        {
            List<DataPoint> subPoints = getSamples(c);
            rndOrder.clear();
            ListUtils.addRange(rndOrder, 0, subPoints.size(), 1);
            Collections.shuffle(rndOrder, rnd);
            
            for(int i : rndOrder)
            {
                cvList.get(curFold).addDataPoint(subPoints.get(i), c);
                curFold = (curFold + 1) % folds;
            }
        }
        
        return cvList;
    }
    
    /**
     * Creates a new data point and adds it to this data set. 
     * @param v the numerical values for the data point
     * @param classes the categorical values for the data point
     * @param classification the true class value for the data point
     * @throws IllegalArgumentException if the given values are inconsistent with the data this class stores. 
     */
    public void addDataPoint(Vec v, int[] classes, int classification)
    {
        addDataPoint(v, classes, classification, 1.0);
    }
    
    private static final int[] emptyInt = new int[0];
    
    /**
     * Creates a new data point with no categorical variables and adds it to 
     * this data set. 
     * @param v the numerical values for the data point
     * @param classification the true class value for the data point
     * @throws IllegalArgumentException if the given values are inconsistent with the data this class stores. 
     */
    public void addDataPoint(Vec v, int classification)
    {
        addDataPoint(v, emptyInt, classification, 1.0);
    }
    
    /**
     * Creates a new data point with no categorical variables and adds it to 
     * this data set. 
     * @param v the numerical values for the data point
     * @param classification the true class value for the data point
     * @param weight the weight value to give to the data point
     * @throws IllegalArgumentException if the given values are inconsistent with the data this class stores. 
     */
    public void addDataPoint(Vec v, int classification, double weight)
    {
        addDataPoint(v, emptyInt, classification, weight);
    }

    /**
     * Creates a new data point and add its to this data set. 
     * @param v the numerical values for the data point
     * @param classes the categorical values for the data point
     * @param classification the true class value for the data point
     * @param weight the weight value to give to the data point
     * @throws IllegalArgumentException if the given values are inconsistent with the data this class stores. 
     */
    public void addDataPoint(Vec v, int[] classes, int classification, double weight)
    {
        if(v.length() != numNumerVals)
            throw new RuntimeException("Data point does not contain enough numerical data points");
        if(classes.length != categories.length)
            throw new RuntimeException("Data point does not contain enough categorical data points");
        
        for(int i = 0; i < classes.length; i++)
            if(!categories[i].isValidCategory(classes[i]) && classes[i] >= 0) // >= so that missing values (negative) are allowed
                throw new IllegalArgumentException("Categoriy value given is invalid");
        
        datapoints.addDataPointCheck(new DataPoint(v, classes, categories));
        setWeight(size()-1, weight);
        targets.add(classification);
    }
    
    /**
     * Creates a new data point and add it
     * @param dp the data point to add to this set
     * @param classification the label for this data point
     */
    public void addDataPoint(DataPoint dp, int classification)
    {
	addDataPoint(dp, classification, 1.0);
    }
    
    /**
     * Creates a new data point and add it
     * @param dp the data point to add to this set
     * @param classification the label for this data point
     * @param weight the weight for the added data point
     */
    public void addDataPoint(DataPoint dp, int classification, double weight)
    {
        if(dp.getNumericalValues().length() != numNumerVals)
            throw new RuntimeException("Data point does not contain enough numerical data points");
        if(dp.getCategoricalValues().length != categories.length)
            throw new RuntimeException("Data point does not contain enough categorical data points");
        
        for(int i = 0; i < dp.getCategoricalValues().length; i++)
        {
            int val = dp.getCategoricalValues()[i];
            if(!categories[i].isValidCategory(val) && val >= 0)
                throw new RuntimeException("Categoriy value given is invalid");
        }
        
        datapoints.addDataPointCheck(dp);
        targets.add(classification);
	setWeight(size()-1, weight);
    }
    
    /**
     * Returns the list of all examples that belong to the given category. 
     * @param category the category desired
     * @return all given examples that belong to the given category
     */
    public List<DataPoint> getSamples(int category)
    {
        ArrayList<DataPoint> subSet = new ArrayList<>();
        for(int i = 0; i < this.targets.size(); i++)
            if(this.targets.getI(i) == category)
                subSet.add(datapoints.getDataPoint(i));
        return subSet;
    }
    
    /**
     * This method is a counter part to {@link #getNumericColumn(int) }. Instead of returning all 
     * values for a given attribute, all values for the attribute that are members of a specific 
     * class are returned. 
     * 
     * @param category the category desired
     * @param n the n'th numerical variable
     * @return a vector of all the values for the n'th numerical variable for the given category
     */
    public Vec getSampleVariableVector(int category, int n)
    {
        List<DataPoint> categoryList = getSamples(category);
        DenseVector vec = new DenseVector(categoryList.size());
        
        for(int i = 0; i < vec.length(); i++)
            vec.set(i, categoryList.get(i).getNumericalValues().get(n));
        
        return vec;
    }
    
    /**
     * 
     * @return the {@link CategoricalData} object for the variable that is to be predicted
     */
    public CategoricalData getPredicting()
    {
        return predicting;
    }
    
    /**
     * Returns the data set as a list of {@link DataPointPair}. 
     * Each data point is paired with it's true class value. 
     * Altering the data points will effect the data set. 
     * Altering the list will not. <br>
     * The list of data points will come in the same order they would 
     * be retrieved in using {@link #getDataPoint(int) }
     * 
     * @return a list of each data point paired with its class value
     */
    public List<DataPointPair<Integer>> getAsDPPList()
    {
        List<DataPointPair<Integer>> dataPoints = new ArrayList<DataPointPair<Integer>>(size());
        for(int i = 0; i < size(); i++)
            dataPoints.add(new DataPointPair<Integer>(datapoints.getDataPoint(i), targets.get(i)));
        
        return dataPoints;
    }
    
    /**
     * Returns the data set as a list of {@link DataPointPair}. 
     * Each data point is paired with it's true class value, which is stored in a double. 
     * Altering the data points will effect the data set. 
     * Altering the list will not. <br>
     * The list of data points will come in the same order they would 
     * be retrieved in using {@link #getDataPoint(int) }
     * 
     * @return a list of each data point paired with its class value stored in a double 
     */
    public List<DataPointPair<Double>> getAsFloatDPPList()
    {
        List<DataPointPair<Double>> dataPoints = new ArrayList<>(size());
        for(int i = 0; i < size(); i++)
            dataPoints.add(new DataPointPair<>(datapoints.getDataPoint(i), (double) targets.getI(i)));
        
        return dataPoints;
    }
    
    /**
     * Computes the prior probabilities of each class, and returns an array containing the values. 
     * @return the array of prior probabilities
     */
    public double[] getPriors()
    {
        double[] priors = new double[getClassSize()];
        
        double sum = 0.0;
        for(int i = 0; i < size(); i++)
        {
            double w = getWeight(i);
            priors[targets.getI(i)] += w;
            sum += w;
        }
        
        for(int i = 0; i < priors.length; i++)
            priors[i] /= sum;
        
        return priors;
    }
    
    /**
     * Returns the number of data points that belong to the specified class, 
     * irrespective of the weights of the individual points. 
     * 
     * @param targetClass the target class 
     * @return how many data points belong to the given class
     */
    public int classSampleCount(int targetClass)
    {
        int count = 0;
        for(int i : targets)
            if(i == targetClass)
                count++;
        return count;
    }

    @Override
    public int size()
    {
        return datapoints.size();
    }

    @Override
    public ClassificationDataSet shallowClone()
    {
        ClassificationDataSet clone = new ClassificationDataSet(numNumerVals, categories, predicting.clone());
        for(int i = 0; i < size(); i++)
            clone.datapoints.addDataPoint(getDataPoint(i));
        clone.targets.addAll(this.targets);
	if(this.weights != null)
	    clone.weights = Arrays.copyOf(this.weights, this.weights.length);
        return clone;
    }

    @Override
    public ClassificationDataSet emptyClone()
    {
	ClassificationDataSet clone = new ClassificationDataSet(numNumerVals, categories, predicting.clone());
        return clone;
    }
    
    @Override
    public ClassificationDataSet getTwiceShallowClone()
    {
        return (ClassificationDataSet) super.getTwiceShallowClone();
    }
}
