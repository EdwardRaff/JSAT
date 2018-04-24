/*
 * Copyright (C) 2018 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.outlier;

import java.io.Serializable;
import jsat.DataSet;
import jsat.classifiers.DataPoint;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public interface Outlier extends Serializable
{
    default public void fit(DataSet d)
    {
        fit(d, false);
    }
    
    public void fit(DataSet d, boolean parallel);
            
    /**
     * Returns an unbounded anomaly/outlier score. Negative values indicate the
     * input is likely to be an outlier, and positive values that the input is
     * likely to be an inlier.
     *
     * @param x
     * @return 
     */
    public double score(DataPoint x);
    
    default public boolean isOutlier(DataPoint x)
    {
        return score(x) < 0 ;
    }
}
