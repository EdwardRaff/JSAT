
package jsat.classifiers.bayesian;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;

import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.utils.IntSet;

/**
 * The conditional probability table (CPT) is a classifier for categorical attributes. It builds the whole
 * conditional probability table for a data set. The size of the CPT grows exponentially with the number of dimensions and options,
 * and requires exponentially more data to get a good fit. CPTs can be useful for small data sets, or as a building 
 * block for another algorithm
 * 
 * @author Edward Raff
 */
public class ConditionalProbabilityTable implements Classifier
{

	private static final long serialVersionUID = -287709075031023626L;
	/**
     * The predicting target class
     */
    private CategoricalData predicting;
    /**
     * The flat array that stores the n dimensional table
     */
    private double[] countArray;
    /**
     * Set subset of the variables we will be using 
     */
    private Map<Integer, CategoricalData> valid;
    /**
     * Maps the index order for the CPT to the index values from the training data set
     */
    private int[] realIndexToCatIndex;
    /**
     * Maps the index values of the training set to the index values used by the CPT. 
     * A value of '-1' indicates that the feature is not in the CPT
     */
    private int[] catIndexToRealIndex;
    /**
     * the dimension size for each category, ie: the number of options for each category
     */
    private int[] dimSize;
    
    /**
     * The index of the predicting attribute, which is the number of categorical features in the training data set. 
     */
    private int predictingIndex;
    
    public CategoricalResults classify(DataPoint data)
    {
        if(catIndexToRealIndex[predictingIndex] < 0)
            throw new UntrainedModelException("CPT has not been trained for a classification problem");
        CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());
        
        int[] cord = new int[dimSize.length];
        
        dataPointToCord(new DataPointPair<Integer>(data, -1), predictingIndex, cord);
        for(int i = 0; i < cr.size(); i++)
        {
            cord[catIndexToRealIndex[predictingIndex]] = i;
            cr.setProb(i, countArray[cordToIndex(cord)]);
        }
        cr.normalize();//Turn counts into probabilityes
        
        return cr;
    }
    
    /**
     * Returns the number of dimensions in the CPT
     * @return the number of dimensions in the CPT
     */
    public int getDimensionSize()
    {
        return dimSize.length;
    }

    /**
     * Converts a data point pair into a coordinate. The paired value contains the value for the predicting index. 
     * Though this value will not be used if the predicting class of the original data set was not used to make the table. 
     * 
     * @param dataPoint the DataPointPair to convert
     * @param targetClass the index in the original data set of the category that we would like to predict
     * @param cord the array to store the coordinate in. 
     * @return the value of the target class for the given data point
     * @throws ArithmeticException if the <tt>cord</tt> array does not match the {@link #getDimensionSize() dimension} of the CPT
     */
    public int dataPointToCord(DataPointPair<Integer> dataPoint, int targetClass, int[] cord)
    {
        if(cord.length != getDimensionSize())
            throw new ArithmeticException("Storage space and CPT dimension miss match");
        DataPoint dp = dataPoint.getDataPoint();
        int skipVal = -1;
        //Set up cord
        for(int i = 0; i < dimSize.length; i++)
        {
            if(realIndexToCatIndex[i] == targetClass)
            {
                if(targetClass == dp.numCategoricalValues())
                    skipVal = dataPoint.getPair();
                else
                    skipVal = dp.getCategoricalValue(realIndexToCatIndex[i]);
            }
            if(realIndexToCatIndex[i] == predictingIndex)
                cord[i] = dataPoint.getPair();
            else
                cord[i] =  dp.getCategoricalValue(realIndexToCatIndex[i]);
        }
        return skipVal;
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        Set<Integer> all = new IntSet();
        for(int i = 0; i < dataSet.getNumCategoricalVars()+1; i++)
            all.add(i);
        trainC(dataSet, all);
    }
    
    /**
     * Creates a CPT using only a subset of the features specified by <tt>categoriesToUse</tt>. 
     * 
     * @param dataSet the data set to train from
     * @param categoriesToUse the attributes to use in training. Each value corresponds to the categorical 
     * index in <tt>dataSet</tt>, and adding the value {@link DataSet#getNumCategoricalVars() }, which is 
     * not a valid index, indicates to used the {@link ClassificationDataSet#getPredicting()  predicting class} 
     * of the data set in the CPT. 
     */
    public void trainC(ClassificationDataSet dataSet, Set<Integer> categoriesToUse)
    {
        if(categoriesToUse.size() > dataSet.getNumFeatures()+1)
            throw new FailedToFitException("CPT can not train on a number of features greater then the dataset's feature count. "
                    + "Specified " + categoriesToUse.size() + " but data set has only " + dataSet.getNumFeatures());
        CategoricalData[] tmp = dataSet.getCategories();
        predicting = dataSet.getPredicting();
        predictingIndex = dataSet.getNumCategoricalVars();
        valid = new HashMap<Integer, CategoricalData>();
        realIndexToCatIndex = new int[categoriesToUse.size()];
        catIndexToRealIndex = new int[dataSet.getNumCategoricalVars()+1];//+1 for the predicting
        Arrays.fill(catIndexToRealIndex, -1);//-1s are non existant values
        dimSize = new int[realIndexToCatIndex.length];
        int flatSize = 1;//The number of bins in the n dimensional array
        int k = 0;
        for(int i : categoriesToUse)
        {
            if(i == predictingIndex)//The predicint class is treated seperatly
                continue;
            CategoricalData dataInfo = tmp[i];
            flatSize *= dataInfo.getNumOfCategories();
            valid.put(i, dataInfo);
            realIndexToCatIndex[k] = i;
            catIndexToRealIndex[i] = k;
            dimSize[k++] = dataInfo.getNumOfCategories();
        }
        
        if(categoriesToUse.contains(predictingIndex))
        {
            //Lastly the predicing quantity
            flatSize *= predicting.getNumOfCategories();
            realIndexToCatIndex[k] = predictingIndex;
            catIndexToRealIndex[predictingIndex] = k;
            dimSize[k] = predicting.getNumOfCategories();
            valid.put(predictingIndex, predicting);
        }
        
        countArray = new double[flatSize];
        Arrays.fill(countArray, 1);//Laplace correction
        
        int[] cordinate = new int[dimSize.length];
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            DataPoint dp = dataSet.getDataPoint(i);
            for (int j = 0; j < realIndexToCatIndex.length; j++)
                if (realIndexToCatIndex[j] != predictingIndex)
                    cordinate[j] = dp.getCategoricalValue(realIndexToCatIndex[j]);
                else
                    cordinate[j] = dataSet.getDataPointCategory(i);
            countArray[cordToIndex(cordinate)]+= dp.getWeight();
        }
    }
    
    /**
     * Queries the CPT for the probability that the class value of <tt>targetClas</tt> would occur with the given DataPointPair.
     * @param targetClass the index in the original data set of the class that we want the probability of
     * @param dataPoint the data point of values paired with the value of the predicting attribute in the original training set
     * @return the probability in [0,1] of the target class value occurring with the given DataPointPair
     */
    public double query(int targetClass, DataPointPair<Integer> dataPoint)
    {
        int[] cord = new int[dimSize.length];
        
        int skipVal = dataPointToCord(dataPoint, targetClass, cord);

        return query(targetClass, skipVal, cord);
    }
    
    /**
     * Queries the CPT for the probability of the target class occurring with the specified value given the class values of the other attributes
     * 
     * @param targetClass the index in the original data set of the class that we want to probability of 
     * @param targetValue the value of the <tt>targetClass</tt> that we want to probability of occurring
     * @param cord the coordinate array that corresponds the the class values for the CPT, where the coordinate of the <tt>targetClass</tt> may contain any value. 
     * @return the probability in [0, 1] of the <tt>targetClass</tt> occurring with the <tt>targetValue</tt> given the information in <tt>cord</tt>
     * @see #dataPointToCord(jsat.classifiers.DataPointPair, int, int[]) 
     */
    public double query(int targetClass, int targetValue, int[] cord)
    {
        double sumVal = 0; 
        double targetVal = 0;
        int realTargetIndex = catIndexToRealIndex[targetClass];
        
        CategoricalData queryData = valid.get(targetClass);
        
        //Now do all other target class posibilty querys 
        for (int i = 0; i < queryData.getNumOfCategories(); i++)
        {
            
            cord[realTargetIndex] = i;
            
                
            double tmp =  countArray[cordToIndex(cord)];
            sumVal += tmp;
            if (i == targetValue)
                targetVal = tmp;
        }

        
        return targetVal/sumVal;
    }
    
    /**
     * Computes the index into the {@link #countArray} using the given coordinate
     * @param cords the coordinate value in question
     * @return the index for the given coordinate
     */
    private int cordToIndex(int... cords)
    {
        if(cords.length != realIndexToCatIndex.length)
            throw new RuntimeException("Something bad");
        int index = 0;
        for(int i = 0; i < cords.length; i++)
        {
            index = cords[i] + dimSize[i]*index;
        }
        return index;
    }
    
    /**
     * Computes the index into the {@link #countArray} using the given data point
     * @param dataPoint the data point to get the index of
     * @return the index for the given data point
     */
    @SuppressWarnings("unused")
    private int cordToIndex(DataPointPair<Integer> dataPoint)
    {
        DataPoint dp = dataPoint.getDataPoint();
        int index = 0;
        for(int i = 0; i < dimSize.length; i++)
            index = dp.getCategoricalValue(realIndexToCatIndex[i]) + dimSize[i]*index;
        return index;
    }

    public boolean supportsWeightedData()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public Classifier clone()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
}
