
package jsat.classifiers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import jsat.DataSet;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

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
     * Contains a list of data points that have already been classified according to {@link #predicting}
     */
    protected List<List<DataPoint>> classifiedExamples;
    protected int numOfSamples = 0;
    
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
            this.numericalVariableNames.set(i, dataSet.getNumericName(i));
    }

    /**
     * Creates a new data set for classification problems from the given list of data points. It is assume the data points are consistent. 
     * @param data the list of data points for the problem. 
     * @param predicting the categorical attribute to use as the target class
     */
    public ClassificationDataSet(List<DataPoint> data, int predicting)
    {
        //Use the first data point to set up
        DataPoint tmp = data.get(0);
        categories = new CategoricalData[tmp.numCategoricalValues()-1];
        for(int i = 0; i < categories.length; i++)
        {
            categories[i] = i >= predicting ? 
                    tmp.getCategoricalData()[i+1] : tmp.getCategoricalData()[i];
        }
        numNumerVals = tmp.numNumericalValues();
        this.predicting = tmp.getCategoricalData()[predicting];
        
        classifiedExamples = new ArrayList<List<DataPoint>>(this.predicting.getNumOfCategories());
        for(int i = 0; i < this.predicting.getNumOfCategories(); i++)
            classifiedExamples.add(new ArrayList<DataPoint>());
        
        
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
            classifiedExamples.get(prevCats[predicting]).add(newPoint); 
        }
        
        numOfSamples = data.size();
        generateGenericNumericNames();
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
        this.predicting = predicting;
        numNumerVals = data.get(0).getVector().length();
        numOfSamples = data.size();
        categories = CategoricalData.copyOf(data.get(0).getDataPoint().getCategoricalData());
        classifiedExamples = new ArrayList<List<DataPoint>>(predicting.getNumOfCategories());
        for(int i = 0; i < predicting.getNumOfCategories(); i++)
            classifiedExamples.add( new ArrayList<DataPoint>());
        for(DataPointPair<Integer> dpp : data)
            classifiedExamples.get(dpp.getPair()).add(dpp.getDataPoint());
        generateGenericNumericNames();
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
        this.predicting = predicting;
        this.categories = categories;
        this.numNumerVals = numerical;
        
        classifiedExamples = new ArrayList<List<DataPoint>>();
        for(int i = 0; i < predicting.getNumOfCategories(); i++)
            classifiedExamples.add(new ArrayList<DataPoint>());
        generateGenericNumericNames();
    }

    private void generateGenericNumericNames()
    {
        this.numericalVariableNames = new ArrayList<String>(getNumNumericalVars());
        for(int i = 0; i < getNumNumericalVars(); i++)
            this.numericalVariableNames.add("Numeric Input " + (i+1));
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
            //each category of lists
            for(int j = 0; j < predicting.getNumOfCategories(); j++)
            {
                //each sample interface a given category
                for(DataPoint dp : list.get(i).classifiedExamples.get(j))
                    cds.addDataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), j);
            }
        }
        
        return cds;
    }
    
    /**
     * Returns the i'th data point from the data set
     * @param i the i'th data point in this set
     * @return the ith data point in this set
     */
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
        if(i >= getSampleSize())
            throw new IndexOutOfBoundsException("There are not that many samples in the data set");
        int set = 0;
        
        while(i >= classifiedExamples.get(set).size())
            i -= classifiedExamples.get(set++).size();
        
        return new DataPointPair<Integer>(classifiedExamples.get(set).get(i), set);
    }
    
    @Override
    public void setDataPoint(int i, DataPoint dp)
    {
        if(i >= getSampleSize())
            throw new IndexOutOfBoundsException("There are not that many samples in the data set");
        int set = 0;
        
        while(i >= classifiedExamples.get(set).size())
            i -= classifiedExamples.get(set++).size();
        
        classifiedExamples.get(set).set(i, dp);
    }
    
    /**
     * Returns the integer value corresponding to the true category of the <tt>i</tt>'th data point. 
     * @param i the <tt>i</tt>'th data point. 
     * @return the integer value for the category of the <tt>i</tt>'th data point. 
     * @throws IndexOutOfBoundsException if <tt>i</tt> is not a valid index into the data set. 
     */
    public int getDataPointCategory(int i)
    {
        if(i >= getSampleSize())
            throw new IndexOutOfBoundsException("There are not that many samples in the data set: " + i);
        else if(i < 0)
            throw new IndexOutOfBoundsException("Can not specify negative index " + i);
        int set = 0;
        
        while(i >= classifiedExamples.get(set).size())
            i -= classifiedExamples.get(set++).size();
        
        return set;
    }
    
    /**
     * 
     * @param folds
     * @return 
     */
    public List<ClassificationDataSet> cvSet(int folds, Random rnd)
    {
        ArrayList<ClassificationDataSet> cvList = new ArrayList<ClassificationDataSet>();
        
        
        List<DataPoint> randSmpls = new ArrayList<DataPoint>(getSampleSize());
        List<Integer> cats = new ArrayList<Integer>(getSampleSize());
        
        for(int i = 0; i < classifiedExamples.size(); i++)
            for(DataPoint dp : classifiedExamples.get(i))
            {
                randSmpls.add(dp); 
                cats.add(i);
            }
        
        
        //Shuffle them TOGETHER, we need to keep track of the category!
        for(int i = getSampleSize()-1; i > 0; i--)
        {
            int swapPos = rnd.nextInt(i);
            //Swap DataPoint
            DataPoint tmp = randSmpls.get(i);
            randSmpls.set(i, randSmpls.get(swapPos));
            randSmpls.set(swapPos, tmp);
            
            int tmpI = cats.get(i);
            cats.set(i, cats.get(swapPos));
            cats.set(swapPos, tmpI);
        }
        
        
        //Initalize all of the new ClassificationDataSets
        for(int i = 0;  i < folds; i++)
            cvList.add(new ClassificationDataSet(getNumNumericalVars(), getCategories(), getPredicting())); 
        
        
        
        int splitSize = getSampleSize()/folds;
        
        //Keep track completly
        int k = 0;
        for(int i = 0; i < folds-1; i++)
        {
            for(int j = 0; j < splitSize; j++)
            {
                DataPoint dp = randSmpls.get(k);
                cvList.get(i).addDataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), cats.get(k));
                k++;
            }
        }
        
        //Last set, contains any leftovers
        for( ; k < getSampleSize(); k++)
        {
            DataPoint dp = randSmpls.get(k);
            cvList.get(folds-1).addDataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), cats.get(k));
        }
        
        return cvList;
    }
    
    /**
     * Creates a new data point and add its to this data set. 
     * @param v the numerical values for the data point
     * @param classes the categorical values for the data point
     * @param classification the true class value for the data point
     * @throws RuntimeException if the given values are inconsistent with the data this class stores. 
     */
    public void addDataPoint(Vec v, int[] classes, int classification)
    {
        if(v.length() != numNumerVals)
            throw new RuntimeException("Data point does not contain enough numerical data points");
        if(classes.length != categories.length)
            throw new RuntimeException("Data point does not contain enough categorical data points");
        
        for(int i = 0; i < classes.length; i++)
            if(!categories[i].isValidCategory(classes[i]))
                throw new RuntimeException("Categoriy value given is invalid");
        
        classifiedExamples.get(classification).add(new DataPoint(v, classes, categories));
        
        numOfSamples++;
        
    }
    
    /**
     * Returns the list of all examples that belong to the given category. 
     * @param category the category desired
     * @return all given examples that belong to the given category
     */
    public List<DataPoint> getSamples(int category)
    {
        return new ArrayList<DataPoint>(classifiedExamples.get(category));
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
        List<DataPointPair<Integer>> dataPoints = new ArrayList<DataPointPair<Integer>>(getSampleSize());
        for(int i = 0; i < predicting.getNumOfCategories(); i++)
            for(DataPoint dp : classifiedExamples.get(i))
                dataPoints.add(new DataPointPair<Integer>(dp, i));
        
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
        List<DataPointPair<Double>> dataPoints = new ArrayList<DataPointPair<Double>>(getSampleSize());
        for(int i = 0; i < predicting.getNumOfCategories(); i++)
            for(DataPoint dp : classifiedExamples.get(i))
                dataPoints.add(new DataPointPair<Double>(dp, (double)i));
        
        return dataPoints;
    }

    @Override
    public int getSampleSize()
    {
        return numOfSamples;
    }
}
