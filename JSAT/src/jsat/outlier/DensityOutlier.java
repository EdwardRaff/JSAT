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
package jsat.outlier;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.distributions.multivariate.MultivariateDistribution;

/**
 * This class provides an outlier detector based upon density estimation. 
 * @author Edward Raff
 */
public class DensityOutlier implements Outlier
{
    double outlireFraction;
    MultivariateDistribution density;
    
    

    @Override
    public void fit(DataSet d, boolean parallel)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double score(DataPoint x)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
