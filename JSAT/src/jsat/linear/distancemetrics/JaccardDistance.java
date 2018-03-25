/*
 * Copyright (C) 2017 Edward Raff
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
package jsat.linear.distancemetrics;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.distributions.kernels.KernelTrick;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import static java.lang.Math.*;
import java.util.Collections;

/**
 * This class implements both the weighted Jaccard Distance and the standard
 * Jaccard distance. If a input is given with only binary 0 or 1 values, the
 * weighted Jaccard is equivalent to the un-weighted version.<br>
 * For the weighted Jaccard version, all values less than or equal to zero will
 * be treated as zero. For the unweighted versions, all non-zero values will
 * behave as if their value is 1.0.
 * <br>
 * The Jaccard Distance and similarity are intertwined, and so this method is
 * both a distance metric and kernel trick.
 *
 * @author Edward Raff
 */
public class JaccardDistance implements DistanceMetric, KernelTrick
{
    private boolean weighted;

    /**
     * Creates a new Jaccard similarity, which can be weighted or unweighted.
     *
     * @param weighted {@code true} to use the weighted Jaccard, {@code false}
     * otherwise.
     */
    public JaccardDistance(boolean weighted)
    {
        this.weighted = weighted;
    }

    /**
     * Creates a new Weighted Jaccard distance / similarity 
     */
    public JaccardDistance()
    {
        this(true);
    }

    @Override
    public double dist(Vec a, Vec b)
    {
        return 1-eval(a, b);
    }

    @Override
    public boolean isSymmetric()
    {
        return true;
    }

    @Override
    public boolean isSubadditive()
    {
        return true;
    }

    @Override
    public boolean isIndiscemible()
    {
        return true;
    }

    @Override
    public double metricBound()
    {
        return 1.0;
    }

    @Override
    public boolean supportsAcceleration()
    {
        return false;
    }

    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        return null;
    }

    @Override
    public double eval(Vec a, Vec b)
    {
        
        double numer = 0, denom = 0;
        Iterator<IndexValue> a_iter = a.getNonZeroIterator();
        Iterator<IndexValue> b_iter = b.getNonZeroIterator();

        IndexValue a_val = a_iter.hasNext() ? a_iter.next() : null;
        IndexValue b_val = b_iter.hasNext() ? b_iter.next() : null;

        while (a_val != null && b_val != null)
        {
            if (weighted)
            {
                if (a_val.getIndex() == b_val.getIndex())
                {
                    numer += max(min(a_val.getValue(), b_val.getValue()), 0.0);
                    denom += max(max(a_val.getValue(), b_val.getValue()), 0.0);
                    
                    a_val = a_iter.hasNext() ? a_iter.next() : null;
                    b_val = b_iter.hasNext() ? b_iter.next() : null;
                }
                else if(a_val.getIndex() < b_val.getIndex())
                {
                    denom += max(a_val.getValue(), 0.0);
                    a_val = a_iter.hasNext() ? a_iter.next() : null;
                }
                else//b had a lower index
                {
                    denom += max(b_val.getValue(), 0.0);
                    b_val = b_iter.hasNext() ? b_iter.next() : null;
                }
            }
            else//unweighted variant
            {
                if (a_val.getIndex() == b_val.getIndex())
                {
                    numer++;
                    denom++;
                    
                    a_val = a_iter.hasNext() ? a_iter.next() : null;
                    b_val = b_iter.hasNext() ? b_iter.next() : null;
                }
                else if(a_val.getIndex() < b_val.getIndex())
                {
                    denom++;
                    a_val = a_iter.hasNext() ? a_iter.next() : null;
                }
                else//b had a lower index
                {
                    denom++;
                    b_val = b_iter.hasNext() ? b_iter.next() : null;
                }
            }
        }
        //catch straglers
        Iterator<IndexValue> finalIter = a_val != null ? a_iter : b_iter;
        IndexValue finalVal =            a_val != null ? a_val : b_val;
        
        while(finalVal != null)
        {
            if(weighted)
                denom += max(finalVal.getValue(), 0.0);
            else
                denom++;
            finalVal =  finalIter.hasNext() ?  finalIter.next() : null;
        }       

        return numer / denom;
    }

    @Override
    public JaccardDistance clone()
    {
        return new JaccardDistance(weighted);
    }

    @Override
    public void addToCache(Vec newVec, List<Double> cache)
    {
        //NOP, nothing to do 
    }

    @Override
    public double eval(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        return eval(vecs.get(a), b);
    }

    @Override
    public double eval(int a, int b, List<? extends Vec> trainingSet, List<Double> cache)
    {
        return eval(trainingSet.get(a), trainingSet.get(b));
    }

    @Override
    public double evalSum(List<? extends Vec> finalSet, List<Double> cache, double[] alpha, Vec y, int start, int end)
    {
        return evalSum(finalSet, cache, alpha, y, getQueryInfo(y), start, end);
    }

    @Override
    public double evalSum(List<? extends Vec> finalSet, List<Double> cache, double[] alpha, Vec y, List<Double> qi, int start, int end)
    {
        double sum = 0;
        for(int i = start; i < end; i++)
            if(alpha[i] != 0)
                sum += alpha[i] * eval(i, y, qi, finalSet, cache);
        return sum;
    }

    @Override
    public boolean normalized()
    {
        return true;
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> trainingSet)
    {
        return null;
    }
}
