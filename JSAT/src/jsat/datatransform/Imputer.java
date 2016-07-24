/*
 * Copyright (C) 2016 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.datatransform;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.math.OnLineStatistics;
import jsat.utils.DoubleList;
import jsat.utils.IndexTable;

/**
 * Imputes missing values in a dataset by finding reasonable default values. For
 * categorical features, the mode will always be used for imputing. Numeric
 * values can change how the imputing value is selected by using the
 * {@link NumericImputionMode} enum.
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class Imputer implements InPlaceTransform
{
    private NumericImputionMode mode;
    /**
     * The values to impute for missing numeric columns
     */
    protected int[] cat_imputs;
    /**
     * The values to impute for missing numeric columns
     */
    protected double[] numeric_imputs;
    
    public static enum NumericImputionMode
    {
        MEAN,
        MEDIAN,
        //TODO, add mode
    }

    public Imputer(DataSet<?> data)
    {
        this(data, NumericImputionMode.MEAN);
    }
    
    public Imputer(DataSet<?> data, NumericImputionMode mode)
    {
        this.mode = mode;
        this.fit(data);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public Imputer(Imputer toCopy)
    {
        this.mode = toCopy.mode;
        if (toCopy.cat_imputs != null)
            this.cat_imputs = Arrays.copyOf(toCopy.cat_imputs, toCopy.cat_imputs.length);
        if (toCopy.numeric_imputs != null)
            this.numeric_imputs = Arrays.copyOf(toCopy.numeric_imputs, toCopy.numeric_imputs.length);
    }
    
    @Override
    public void fit(DataSet d)
    {
        numeric_imputs = new double[d.getNumNumericalVars()];
        cat_imputs = new int[d.getNumCategoricalVars()];
        List<List<Double>> columnCounts = null;
        List<List<Double>> columnWeights = null;
        double[] colSoW = null;
        
        switch(mode)
        {
            case MEAN:
                //lets just do this now since we are calling the function
                OnLineStatistics[] stats = d.getOnlineColumnStats(true);
                for(int i = 0; i < stats.length; i++)
                    numeric_imputs[i] = stats[i].getMean();
                break;
            case MEDIAN:
                columnCounts = new ArrayList<List<Double>>(d.getNumNumericalVars());
                columnWeights = new ArrayList<List<Double>>(d.getNumNumericalVars());
                colSoW = new double[d.getNumNumericalVars()];
                for(int i = 0; i < d.getNumNumericalVars(); i++)
                {
                    columnCounts.add(new DoubleList(d.getSampleSize()));
                    columnWeights.add(new DoubleList(d.getSampleSize()));
                }
                break;
        }
        
        //space to count how many times each cat is seen
        double[][] cat_counts = new double[d.getNumCategoricalVars()][];
        for(int i = 0; i < cat_counts.length; i++)
            cat_counts[i] = new double[d.getCategories()[i].getNumOfCategories()];
        
        
        for(int sample = 0; sample < d.getSampleSize(); sample++)
        {
            DataPoint dp = d.getDataPoint(sample);
            final double weights = dp.getWeight();
            
            int[] cats = dp.getCategoricalValues();
            for(int i = 0; i < cats.length; i++)
                if(cats[i] >= 0)//missing is < 0
                    cat_counts[i][cats[i]] += weights;
            
            
            Vec numeric = dp.getNumericalValues();
            if (mode == NumericImputionMode.MEDIAN)
            {

                for (IndexValue iv : numeric)
                    if (!Double.isNaN(iv.getValue()))
                    {
                        columnCounts.get(iv.getIndex()).add(iv.getValue());
                        columnWeights.get(iv.getIndex()).add(weights);
                        colSoW[iv.getIndex()] += weights;
                    }
            }
        }
        
        if(mode == NumericImputionMode.MEDIAN)
        {
            IndexTable it = new IndexTable(d.getNumNumericalVars());
            for (int col = 0; col < d.getNumNumericalVars(); col++)
            {
                List<Double> colVal = columnCounts.get(col);
                List<Double> colWeight = columnWeights.get(col);
                it.reset();
                it.sort(colVal);
                
                //we are going to loop through until we reach past the half weight mark, getting us the weighted median
                double goal = colSoW[col]/2;
                double lastSeen = 0;
                double curWeight = 0;
                //loop breaks one we pass the median, so last seen is the median
                for(int i = 0; i < it.length() && curWeight < goal; i++)
                {
                    int indx = it.index(i);
                    lastSeen = colVal.get(indx);
                    curWeight += colWeight.get(indx);
                }
                
                numeric_imputs[col] = lastSeen;
            }
        }
        
        //last, determine mode for cats
        for(int col = 0; col < cat_counts.length; col++)
        {
            int col_mode = 0;
            for(int j = 1; j < cat_counts[col].length; j++)
                if(cat_counts[col][j] > cat_counts[col][col_mode])
                    col_mode = j;
            cat_imputs[col] = col_mode;
        }
    }

    @Override
    public void mutableTransform(DataPoint dp)
    {
        Vec vec = dp.getNumericalValues();
        for(IndexValue iv : vec)
            if(Double.isNaN(iv.getValue()))
                vec.set(iv.getIndex(), numeric_imputs[iv.getIndex()]);
        int[] cats = dp.getCategoricalValues();
        for(int i = 0; i < cats.length; i++)
            if(cats[i] < 0)
                cats[i] = cat_imputs[i];
                
    }

    @Override
    public boolean mutatesNominal()
    {
        return true;
    }

    @Override
    public DataPoint transform(DataPoint dp)
    {
        DataPoint toRet = dp.clone();
        //TODO, sparse vec case can be handled better by making a and setting it seperatly
        mutableTransform(toRet);
        return toRet;
    }

    @Override
    public Imputer clone()
    {
        return new Imputer(this);
    }
    
}
