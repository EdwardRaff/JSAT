
package jsat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;

/**
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
    }

    public SimpleDataSet(CategoricalData[] categories, int numNumericalValues)
    {
        this.categories = categories;
        this.numNumerVals = numNumericalValues;
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
    }

    @Override
    public int getSampleSize()
    {
        return dataPoints.size();
    }

    @Override
    public List<SimpleDataSet> cvSet(int folds, Random rand)
    {
        List<DataPoint> shuffleSet = new ArrayList<DataPoint>(this.dataPoints);
        
        Collections.shuffle(shuffleSet, rand);
        
        List<SimpleDataSet> cvSet = new ArrayList<SimpleDataSet>(folds);
        for(int i = 0; i < folds; i++)
            cvSet.add(new SimpleDataSet(this.categories, this.getNumNumericalVars()));
        
        for(int i = 0; i < dataPoints.size(); i++)
            cvSet.get(i%folds).dataPoints.add(this.dataPoints.get(i));
        
        return cvSet;
    }

    /**
     * 
     * @return direct access to the list that backs this data set. 
     */
    public List<DataPoint> getBackingList()
    {
        return dataPoints;
    }
    
}
