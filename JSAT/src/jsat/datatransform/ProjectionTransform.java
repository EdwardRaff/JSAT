/*
 * Copyright (C) 2021 Edward Raff
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

import java.util.Arrays;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * This class is used as a base class for simple linear projections of a
 * dataset. You must pass in the projection you want to use at construction. It
 * should be used only if you need a temporary transform object (will not be
 * saved) and know how you want to transform it, or to extend for use by another
 * class.
 *
 *
 * @author Edward Raff
 */
public class ProjectionTransform implements DataTransform
{
    protected Matrix P;
    protected Vec b;

    /**
     * 
     * @param P the projection matrix
     * @param b an offset to apply after projection (i.e., bias terms)
     */
    public ProjectionTransform(Matrix P, Vec b)
    {
	this.P = P;
	this.b = b;
    }

    public ProjectionTransform(ProjectionTransform toClone)
    {
	this(toClone.P.clone(), toClone.b.clone());
    }

    @Override
    public DataPoint transform(DataPoint dp)
    {
	Vec x_new = P.multiply(dp.getNumericalValues());
	x_new.mutableAdd(b);
	
	DataPoint newDP = new DataPoint(
		x_new,
		Arrays.copyOf(dp.getCategoricalValues(), dp.numCategoricalValues()),
		CategoricalData.copyOf(dp.getCategoricalData()));
	return newDP;
    }

    @Override
    public void fit(DataSet data)
    {
	//NOP
    }

    @Override
    public ProjectionTransform clone()
    {
	return new ProjectionTransform(this);
    }
    
}
