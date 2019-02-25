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
import java.util.List;
import java.util.Set;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import jsat.math.OnLineStatistics;

/**
 *
 * @author Edward Raff
 */
public interface DataStore 
{   

    /**
     * This is the default data store type that will be used whenever any new
     * DataSet object is created.
     */
    public static DataStore DEFAULT_STORE = new RowMajorStore();
    
    /**
     * Sets the categorical data information used in this data store
     *
     * @param cat_info an array of the number of categorical features, with each
     * object describing their information.
     */
    public void setCategoricalDataInfo(CategoricalData[] cat_info);
    
    public CategoricalData[] getCategoricalDataInfo();
    
    /**
     * Adds the given data point to this data store. If the given data point has
     * more features (numeric or categorical) than is expected, the store will
     * automatically expand to accept the given features.<br>
     * If you want an error to be thrown on miss-match between current and given
     * data, use {@link #addDataPointCheck(jsat.classifiers.DataPoint) }.
     *
     *
     * @param dp the data point to add
     */
    public void addDataPoint(DataPoint dp);
    
    /**
     * Adds the given data point to this data store, and checks that it has the
     * expected number of numeric and categorical features to be added to this
     * data store.
     *
     * @param dp the data point to add
     */
    default public void addDataPointCheck(DataPoint dp)
    {
        //TODO improve these error messages. 
        if(dp.getNumericalValues().length() != numNumeric())
            throw new IllegalArgumentException("Input has incorrect number of numeric features");
        int[] cat_vals = dp.getCategoricalValues();
        if(cat_vals.length != numCategorical())
            throw new IllegalArgumentException("Input has the incorrect number of categorical features");
        for(int i = 0; i < cat_vals.length; i++)
            if(getCategoricalDataInfo()[i].getNumOfCategories() <= cat_vals[i])
                throw new IllegalArgumentException("Input has an invalid value for categorical feature");
    
        addDataPoint(dp);
    }
    
    /**
     * Returns the data point stored at the <tt>i</tt>'th index in this data store. 
     * @param i the index of the datum to get
     * @return the <tt>i</tt>'th data point
     */
    public DataPoint getDataPoint(int i);
    
    /**
     * This is called to indicate to the data store that we are done adding
     * values. This does not mean that values can not be added in the future,
     * but this allows the implementation to perform cleanup.
     */
    public void finishAdding();
    
    /**
     *
     * @return the number of numeric features that are contained by data in this
     * store.
     */
    public int numNumeric();
    
    /**
     * Sets the number of numerical features that will be stored in this datastore object
     * @param d the number of numerical features (i.e., dimensions) that should be stored. 
     */
    public void setNumNumeric(int d);

    /**
     *
     * @return the number of categorical features that are contained by data in
     * this store.
     */
    public int numCategorical();
    
    /**
     * Replaces the data point stored at the <tt>i</tt>'th index in this data store. 
     * @param i
     * @param dp 
     */
    public void setDataPoint(int i, DataPoint dp);
    
    /**
     * 
     * @param skipColumns if a column's index is in this set, a {@code null} 
     * will be returned in the array at the column's index instead of a vector
     * @return 
     */
    public Vec[] getNumericColumns(Set<Integer> skipColumns);
    
    default public List<DataPoint> toList()
    {
        List<DataPoint> list = new ArrayList<>();
        for(int i = 0; i < size(); i++)
            list.add(getDataPoint(i));
        return list;
    }
    
    /**
     * 
     * @return {@code true} if row-major traversal should be the preferred
     * iteration order for this data store, or {@code false} if column-major
     * should be preferred.
     */
    default public boolean rowMajor()
    {
        return true;
    }
    
    public int size();
    
    /**
     * Returns statistics on the sparsity of the vectors in this data store. 
     * Vectors that are not considered sparse will be treated as completely 
     * dense, even if zero values exist in the data. 
     * 
     * @return an object containing the statistics of the vector sparsity
     */
    public OnLineStatistics getSparsityStats();
    
    public DataStore clone();

    /**
     * Creates a new data store that is the same type as this one, but contains
     * no data points.
     *
     * @return a new data store that is the same type as this one, but contains
     * no data points.
     */
    public DataStore emptyClone();
}
