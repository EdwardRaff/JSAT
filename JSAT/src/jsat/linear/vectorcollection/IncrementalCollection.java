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
package jsat.linear.vectorcollection;

import jsat.linear.Vec;

/**
 * This interface is for Vector Collections that support incremental
 * construction. If all data is available at the onset, it is recommended to use
 * the appropriate constructor / bulk insertion as they may be more compute
 * efficient or produce better indexes. The incremental insertion of points is
 * not guaranteed to result in a collection that is equally as performant in
 * either construction or querying. However, it does allow for additions to the
 * collection without needing to re-build the entire collection. Efficiency and
 * performance of incremental additions will depend on the base implementation.
 *
 * @author Edward Raff
 * @param <V> The type of vectors stored in this collection
 */
public interface IncrementalCollection<V extends Vec> extends VectorCollection<V>
{
    /**
     * Incrementally adds the given datapoint into the collection 
     * @param x the vector to add to the collection 
     */
    public void insert(V x);

    @Override
    public IncrementalCollection<V> clone();
}
