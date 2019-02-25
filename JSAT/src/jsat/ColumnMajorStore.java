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
import jsat.utils.DoubleList;
import jsat.utils.IntList;

/**
 *
 * @author Edward Raff
 */
public class ColumnMajorStore implements DataStore
{
    private boolean sparse;
    int size = 0;
    List<Vec> columns;
    List<IntList> cat_columns;
    CategoricalData[] cat_info;

    /**
     * Creates a new Data Store to add points to, where the number of features
     * is not known in advance.
     */
    public ColumnMajorStore(boolean sparse)
    {
        this(0, null, sparse);
    }
    
    /**
     * Creates a new Data Store to add points to, where the number of features
     * is not known in advance.
     */
    public ColumnMajorStore() 
    {
        this(0, null);
    }

    @Override
    public CategoricalData[] getCategoricalDataInfo() 
    {
        return cat_info;
    }
    
    /**
     * Creates a new Data Store with the intent for a specific number of features known ahead of time. 
     * @param numNumeric the number of numeric features to be in the data store
     * @param cat_info the information about the categorical data
     */
    public ColumnMajorStore(int numNumeric, CategoricalData[] cat_info) 
    {
        this(numNumeric, cat_info, true);
    }
    /**
     * Creates a new Data Store with the intent for a specific number of features known ahead of time. 
     * @param numNumeric the number of numeric features to be in the data store
     * @param cat_info the information about the categorical data
     * @param sparse if the columns should be stored in a sparse format
     */
    public ColumnMajorStore(int numNumeric, CategoricalData[] cat_info, boolean sparse) 
    {
        this.columns = new ArrayList<>(numNumeric);
        for(int i = 0; i < numNumeric; i++)
            this.columns.add(sparse ? new SparseVector(10) : new DenseVector(10));
        this.cat_info = cat_info;
        this.cat_columns = new ArrayList<>();
        for(int i = 0; i < (cat_info == null ? 0 : cat_info.length); i++)
            this.cat_columns.add(new IntList());
        
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public ColumnMajorStore(ColumnMajorStore toCopy) 
    {
        this.columns = new ArrayList<>(toCopy.columns.size());
        for(Vec v : toCopy.columns)
            this.columns.add(v.clone());
        this.cat_columns = new ArrayList<>(toCopy.cat_columns.size());
        for(IntList cv : toCopy.cat_columns)
            this.cat_columns.add(new IntList(cv));
                
        if(toCopy.cat_info != null)
            this.cat_info = CategoricalData.copyOf(toCopy.cat_info);
        this.size = toCopy.size;
        this.sparse = toCopy.sparse;
    }

    @Override
    public boolean rowMajor()
    {
        return false;
    }

    @Override
    public void setCategoricalDataInfo(CategoricalData[] cat_info) 
    {
        this.cat_info = cat_info;
    }
    

    @Override
    public int numCategorical() 
    {
        return cat_columns.size();
    }

    @Override
    public int numNumeric() 
    {
        return columns.size();
    }

    @Override
    public void setNumNumeric(int d)
    {
        if(d < 0)
            throw new RuntimeException("Can not store a negative number of features (" +d + ")");
        //do we need to add more?
        while(columns.size() < d)
            this.columns.add(sparse ? new SparseVector(10) : new DenseVector(10));
        //or do we need to remove?
        while(columns.size() > d)
            columns.remove(columns.size()-1);
        //now we should be the same length!
    }
    
    
    
    @Override
    public void addDataPoint(DataPoint dp)
    {
        Vec x = dp.getNumericalValues();
        int[] x_c = dp.getCategoricalValues();
        
        int pos = size++;
        //make sure we have capacity in the columns
        while(columns.size() < x.length())
            columns.add(sparse ? new SparseVector(size) : new DenseVector(size));
        while(cat_columns.size() < x_c.length)
        {
            int[] newCol = new int[size];
            Arrays.fill(newCol, -1);//All previous entries have a missing value indication  (-1) now. 
            cat_columns.add(IntList.view(newCol));
        }
        
        //add numeric features 
        for(IndexValue iv : x)
        {
            int d = iv.getIndex();
            double v = iv.getValue();
            
            Vec x_d = columns.get(d);
            if(x_d.length() <= pos)//We need to increase the size of x_d
                x_d.setLength(size*2);
            //size is now correct, we can set the values
            x_d.set(pos, v);
        }
        
        //add categorical features
        for(int d = 0; d < x_c.length; d++)
        {
            IntList col = cat_columns.get(d);
            while(col.size() <= pos)//fill missing values with missing value indicator
                col.add(-1);
            //we filled up to the current value, so now we can call set and 
            // handle 3 cases with same code. Col is correct size and col is too
            //small or too large.
            col.set(pos, x_c[d]);
        }
        
        //add missing categorical features
        for(int d = x_c.length; d < cat_columns.size(); d++)
        {
            IntList col = cat_columns.get(d);
            while(col.size() <= pos)//fill missing values with missing value indicator
                col.add(-1);
        }
    }
    
    @Override
    public DataPoint getDataPoint(int i)
    {
        if(i >= size)
            throw new IndexOutOfBoundsException("Requested datapoint " + i + " but index has only " + size + " datums");
        
        int d_n = numNumeric();
        int d_c = numCategorical();
        //best guess on sparseness 
        Vec x = sparse ? new SparseVector(d_n) : new DenseVector(d_n);
        int[] cat = new int[d_c];
        
        for(int j = 0; j < d_n; j++)
        {
            Vec col_j = columns.get(j);
            if(col_j.length() > i)
                x.set(j, col_j.get(i));
            //else, no more occurances of that featre by this point - so skip. 
        }
        for(int j = 0; j < d_c; j++)
            cat[j] = cat_columns.get(j).get(i);
        
        if(sparse && x.nnz() > d_n/2)
            x = new DenseVector(x);//denseify
        
        return new DataPoint(x, cat, cat_info);
    }

    @Override
    public void setDataPoint(int i, DataPoint dp)
    {
        if(i >= size)
            throw new IndexOutOfBoundsException("Requested datapoint " + i + " but index has only " + size + " datums");
        
        int d_n = numNumeric();
        int d_c = numCategorical();
        Vec x = dp.getNumericalValues();
        int[] cat = dp.getCategoricalValues();
        
       
        for(int j = 0; j < d_n; j++)
        {
            Vec tmp = columns.get(j);
            tmp.set(i, 0);
        }
        for(IndexValue iv : x)
            columns.get(iv.getIndex()).set(i, iv.getValue());
        for(int j = 0; j < d_c; j++)
            cat_columns.get(j).set(i, cat[j]);
    }

    @Override
    public void finishAdding() 
    {
        if(cat_info == null)
        {
            cat_info = new CategoricalData[numCategorical()];
            for(int j = 0; j < cat_info.length; j++)
            {
                int options = cat_columns.get(j).streamInts().max().orElse(1);
                cat_info[j] = new CategoricalData(Math.max(options, 1));//max incase all are missing 
            }
        }
        
        for(int i = 0; i < columns.size(); i++)
        {
            Vec v = columns.get(i);
            v.setLength(size);
        }
    }

    @Override
    public Vec[] getNumericColumns(Set<Integer> skipColumns)
    {
        Vec[] toRet = new Vec[numNumeric()];
        for(int j = 0; j < toRet.length; j++)
            if(!skipColumns.contains(j))
            {
                toRet[j] = columns.get(j);
                if(toRet[j].length() != size())
                    toRet[j] = new SubVector(0, size, toRet[j]);
            }
        return toRet;
    }

    @Override
    public int size()
    {
        return size;
    }

    @Override
    public OnLineStatistics getSparsityStats() 
    {
        OnLineStatistics stats = new OnLineStatistics();
        for(Vec v : columns)
        {
            if(v.isSparse())
                stats.add(v.nnz() / (double)size);
            else
                stats.add(1.0);
        }
        
        return stats;
    }

    @Override
    public ColumnMajorStore clone() 
    {
        return new ColumnMajorStore(this);
    }

    @Override
    public ColumnMajorStore emptyClone()
    {
        return new ColumnMajorStore(columns.size(), cat_info, sparse);
    }

    @Override
    public int[] getCatColumn(int i)
    {
        if (i < 0 || i >= numCategorical())
            throw new IndexOutOfBoundsException("There is no index for column " + i);
        return Arrays.copyOf(cat_columns.get(i).streamInts().toArray(), size());
    }

}
