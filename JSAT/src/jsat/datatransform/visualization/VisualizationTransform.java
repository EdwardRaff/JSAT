/*
 * Copyright (C) 2015 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.datatransform.visualization;

import java.io.Serializable;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.datatransform.DataTransform;

/**
 * Visualization Transform is similar to the {@link DataTransform} interface,
 * except it can not necessarily be applied to new datapoints. Classes
 * implementing this interface are intended to create 2D or 3D versions of a
 * dataset that can be visualized easily.<br>
 * <br>
 * By default, all implementations will create a 2D projection of the data if
 * supported.
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public interface VisualizationTransform extends Cloneable, Serializable
{
    /**
     * 
     * @return the number of dimensions that a dataset will be embedded down to
     */
    public int getTargetDimension();
    
    
    /**
     * Sets the target dimension to embed new dataset to. Many visualization
     * methods may only support a target of 2 or 3 dimensions, or only one of
     * those options. For that reason a boolean value will be returned
     * indicating if the target size was acceptable. If not, no change to the
     * object will occur.
     *
     * @param target the new target dimension size when {@link #transform(jsat.DataSet)
     * } is called.
     * @return {@code true} if this transform supports that dimension and it was
     * set, {@code false} if the target dimension is unsupported and the
     * previous value will be used instead.
     */
    public boolean setTargetDimension(int target);
            
    
    /**
     * Transforms the given data set, returning a dataset of the same type.
     *
     * @param <Type> the dataset type
     * @param d the data set to transform
     * @return the lower dimension dataset for visualization.
     */
    public <Type extends DataSet> Type transform(DataSet<Type> d);

    /**
     * Transforms the given data set, returning a dataset of the same type.
     *
     * @param <Type> the dataset type
     * @param d the data set to transform
     * @param ex the source of threads for parallel computation
     * @return the lower dimension dataset for visualization.
     */
    public <Type extends DataSet> Type transform(DataSet<Type> d, ExecutorService ex);
}
