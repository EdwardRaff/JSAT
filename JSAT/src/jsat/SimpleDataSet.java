
package jsat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * SimpleData Set is a basic implementation of a data set. Has no assumptions about the task that is going to be performed. 
 * 
 * @author Edward Raff
 */
public class SimpleDataSet extends DataSet<SimpleDataSet>
{
    protected List<DataPoint> dataPoints;

    public SimpleDataSet(final List<DataPoint> dataPoints)
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

    public SimpleDataSet(final CategoricalData[] categories, final int numNumericalValues)
    {
        this.categories = categories;
        this.numNumerVals = numNumericalValues;
        this.dataPoints = new ArrayList<DataPoint>();
    }
    
    @Override
    public DataPoint getDataPoint(final int i)
    {
        return dataPoints.get(i);
    }

    @Override
    public void setDataPoint(final int i, final DataPoint dp)
    {
        dataPoints.set(i, dp);
        columnVecCache.clear();
    }
    
    /**
     * Adds a new datapoint to this set. 
     * @param dp the datapoint to add
     */
    public void add(final DataPoint dp)
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
    protected SimpleDataSet getSubset(final List<Integer> indicies)
    {
        final SimpleDataSet newData = new SimpleDataSet(categories, numNumerVals);
        for(final int i : indicies) {
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
