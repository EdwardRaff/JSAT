
package jsat.regression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class RegressionDataSet extends DataSet<RegressionDataSet>
{

    /**
     * The list of all data points, paired with their true regression output
     */
    List<DataPointPair<Double>> dataPoints;
    
    public RegressionDataSet(int numerical, CategoricalData[] categories)
    {
        this.numNumerVals = numerical;
        this.categories = categories;
        dataPoints = new ArrayList<DataPointPair<Double>>();
    }
    
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
            DenseVector newVec = new DenseVector(numNumerVals);
            Vec origVec = dp.getNumericalValues();
            //Set up the new vector
            for(int i = 0; i < origVec.length()-1; i++)
            {
                if(i >= predicting)
                    newVec.set(i+1, origVec.get(i+1));
                else
                    newVec.set(i, origVec.get(i));
            }
            
            DataPoint newDp = new DataPoint(newVec, dp.getCategoricalValues(), categories);
            DataPointPair<Double> dpp = new DataPointPair<Double>(newDp, origVec.get(predicting));
            
            dataPoints.add(dpp);
        }
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
    
    /**
     * Creates a new data point to be added to the data set
     * @param numerical the numerical values for the data point
     * @param categories the categorical values for the data point
     */
    public void addDataPoint(Vec numerical, int[] categories, double val)
    {
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
    }
    
    
    @Override
    public DataPoint getDataPoint(int i)
    {
        return dataPoints.get(i).getDataPoint();
    }
    
    public DataPointPair<Double> getDataPointPair(int i)
    {
        return dataPoints.get(i);
    }
    
    @Override
    public void setDataPoint(int i, DataPoint dp)
    {
        dataPoints.get(i).setDataPoint(dp);
    }

    @Override
    public List<RegressionDataSet> cvSet(int folds, Random rand)
    {
        
        List<DataPointPair<Double>> shuffleSet = new ArrayList<DataPointPair<Double>>(this.dataPoints);
        
        Collections.shuffle(shuffleSet, rand);
        
        List<RegressionDataSet> cvSet = new ArrayList<RegressionDataSet>(folds);
        for(int i = 0; i < folds; i++)
            cvSet.add(new RegressionDataSet(this.numNumerVals, CategoricalData.copyOf(this.categories)));
        
        for(int i = 0; i < dataPoints.size(); i++)
            cvSet.get(i%folds).dataPoints.add(this.dataPoints.get(i));
        
        return cvSet;
    }
    
    public Vec regressionValues()
    {
        DenseVector vec = new DenseVector(getSampleSize());
        for(int i = 0; i < dataPoints.size(); i++)
            vec.set(i, dataPoints.get(i).getPair());
        
        return vec;
    }

    @Override
    public int getSampleSize()
    {
        return dataPoints.size();
    }
}
