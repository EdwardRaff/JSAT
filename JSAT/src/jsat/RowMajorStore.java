/*
 * Copyright (C) 2018 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package jsat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.SparseVector;
import jsat.linear.SubVector;
import jsat.linear.Vec;
import jsat.math.OnLineStatistics;

/**
 *
 * @author Edward Raff
 */
public class RowMajorStore implements DataStore
{
    protected List<DataPoint> datapoints;
    protected int num_numeric = 0;
    protected int num_cat = 0;
    protected CategoricalData[] cat_info;
    
    /**
     * Creates a new Data Store to add points to, where the number of features is not known in advance. 
     */
    public RowMajorStore() 
    {
        this(0, null);
    }

    /**
     * Creates a new Data Store with the intent for a specific number of features known ahead of time. 
     * @param numNumeric the number of numeric features to be in the data store
     * @param cat_info the information about the categorical data
     */
    public RowMajorStore(int numNumeric, CategoricalData[] cat_info) 
    {
        this.num_numeric = numNumeric;
        this.cat_info = cat_info;
        this.num_cat = cat_info == null ? 0 : cat_info.length;
        datapoints = new ArrayList<>();
    }
    
    public RowMajorStore(List<DataPoint> collection)
    {
        this(collection.get(0).numNumericalValues(), collection.get(0).getCategoricalData());
        for(DataPoint dp : collection)
            this.addDataPoint(dp);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public RowMajorStore(RowMajorStore toCopy) 
    {
        this.datapoints = new ArrayList<>(toCopy.datapoints);
        if(toCopy.cat_info != null)
            this.cat_info = CategoricalData.copyOf(toCopy.cat_info);
        this.num_cat = toCopy.num_cat;
        this.num_numeric = toCopy.numNumeric();
    }
    
    @Override
    public void addDataPoint(DataPoint dp)
    {
        datapoints.add(dp);
        num_numeric = Math.max(dp.getNumericalValues().length(), num_numeric);
        num_cat = Math.max(dp.getCategoricalValues().length, num_cat);
    }

    @Override
    public CategoricalData[] getCategoricalDataInfo() 
    {
        return cat_info;
    }
    

    @Override
    public DataPoint getDataPoint(int i)
    {
        return datapoints.get(i);
    }

    @Override
    public void setDataPoint(int i, DataPoint dp)
    {
        datapoints.set(i, dp);
    }

    @Override
    public void finishAdding() 
    {
        for(int i = 0; i < datapoints.size(); i++)
        {
            DataPoint d = datapoints.get(i);
            Vec v = d.getNumericalValues();
            Vec nv = v;
            //Check that the number of numeric values match up
            //if short, fill with zeros
            v.setLength(num_numeric);
            
            //Check that the number of categorical values match up
            //if short, fill with missing values
            int[] c = d.getCategoricalValues();
            int[] nc = c;
            if(d.numCategoricalValues() < num_cat)
            {
                nc = Arrays.copyOf(c, num_cat);
                for(int j = c.length; j < nc.length; j++)
                    nc[j] = -1;//Missing value
            }
            
            if(v != nv || c != nc)//intentionally doing equality check on objects
            {
                datapoints.set(i, new DataPoint(nv, nc, cat_info));
            }
        }
    }


    @Override
    public int size()
    {
        return datapoints.size();
    }

    @Override
    public Vec[] getNumericColumns(Set<Integer> skipColumns)
    {
        boolean sparse = getSparsityStats().getMean() < 0.6;
        Vec[] cols = new Vec[numNumeric()];
        
        for(int i = 0; i < cols.length; i++)
            if(!skipColumns.contains(i))
            {
                cols[i] = sparse ? new SparseVector(size()) : new DenseVector(size());
            }
        
        for(int i = 0; i < size(); i++)
        {
            Vec v = getDataPoint(i).getNumericalValues();
            
            for(IndexValue iv : v)
            {
                int col = iv.getIndex();
                if(cols[col] != null)
                    cols[col].set(i, iv.getValue());
            }
        }
            
        return cols;
    }

    @Override
    public void setCategoricalDataInfo(CategoricalData[] cat_info) 
    {
        this.cat_info = cat_info;
        this.num_cat = cat_info.length;
    }

    @Override
    public int numNumeric() 
    {
        return num_numeric;
    }

    @Override
    public void setNumNumeric(int d)
    {
        if(d < 0)
            throw new RuntimeException("Can not store a negative number of features (" +d + ")");
        num_numeric = d;
    }

    @Override
    public int numCategorical() 
    {
        return num_cat;
    }

    @Override
    public OnLineStatistics getSparsityStats() 
    {
        OnLineStatistics stats = new OnLineStatistics();
        for(int i = 0; i < size(); i++)
        {
            Vec v = getDataPoint(i).getNumericalValues();
            if(v.isSparse())
                stats.add(v.nnz() / (double)v.length());
            else
                stats.add(1.0);
        }
        
        return stats;
    }

    @Override
    public RowMajorStore clone() 
    {
        return new RowMajorStore(this);
    }

    @Override
    public RowMajorStore emptyClone()
    {
        return new RowMajorStore(num_numeric, cat_info);
    }

    @Override
    public int[] getCatColumn(int i)
    {
        if (i < 0 || i >= numCategorical())
            throw new IndexOutOfBoundsException("There is no index for column " + i);
        int[] toRet = new int[size()];
        for(int z = 0; z < size(); z++)
            toRet[z] = datapoints.get(z).getCategoricalValue(i);
        return toRet;
    }

}
