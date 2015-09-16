
package jsat;

import java.util.ArrayList;
import java.util.List;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;

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
        if(dataPoints.isEmpty()) {
          throw new RuntimeException("Can not create empty data set");
        }
        this.dataPoints = dataPoints;
        this.categories =  dataPoints.get(0).getCategoricalData();
        this.numNumerVals = dataPoints.get(0).numNumericalValues();
        this.numericalVariableNames = new ArrayList<String>(this.numNumerVals);
        for(int i = 0; i < getNumNumericalVars(); i++) {
          this.numericalVariableNames.add("Numeric Input " + (i+1));
        }
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
        for(int i : indicies) {
          newData.add(getDataPoint(i));
        }
        return newData;
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
