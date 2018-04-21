/*
 * Copyright (C) 2018 edwardraff
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

import java.util.List;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * This class is a generic wrapper for the Vector Collection objects within
 * JSAT. It will attempt to select a good choice for any given dataset and
 * distance metric combination at runtime.
 *
 * @author Edward Raff
 */
public class DefaultVectorCollection<V extends Vec> implements VectorCollection<V>
{
    private DistanceMetric dm;
    VectorCollection<V> base;

    public DefaultVectorCollection()
    {
        this(new EuclideanDistance());
    }

    public DefaultVectorCollection(DistanceMetric dm)
    {
        setDistanceMetric(dm);
    }
    
    public DefaultVectorCollection(DistanceMetric dm, List<V> vecs)
    {
        this(dm, vecs, false);
    }
    
    public DefaultVectorCollection(DistanceMetric dm, List<V> vecs, boolean parallel)
    {
        setDistanceMetric(dm);
        build(parallel, vecs, dm);
    }

    public DefaultVectorCollection(DefaultVectorCollection toCopy)
    {
        this.dm = toCopy.dm.clone();
        if (toCopy.base != null)
            this.base = toCopy.base.clone();
    }

    @Override
    public void build(boolean parallel, List<V> collection, DistanceMetric dm)
    {
        int N = collection.size();
        if(N <= 20 || !dm.isValidMetric())
            base = new VectorArray<>();
        else
            base = new VPTreeMV<>();
        base.build(parallel, collection, dm);
    }
    
    @Override
    public List<Double> getAccelerationCache()
    {
        return base.getAccelerationCache();
    }

    @Override
    public void setDistanceMetric(DistanceMetric dm)
    {
        this.dm = dm;
    }

    @Override
    public DistanceMetric getDistanceMetric()
    {
        return dm;
    }

    @Override
    public void search(Vec query, double range, List<Integer> neighbors, List<Double> distances)
    {
        base.search(query, range, neighbors, distances);
    }

    @Override
    public void search(Vec query, int numNeighbors, List<Integer> neighbors, List<Double> distances)
    {
        base.search(query, numNeighbors, neighbors, distances);
    }

    @Override
    public V get(int indx)
    {
        return base.get(indx);
    }

    @Override
    public int size()
    {
        return base.size();
    }

    @Override
    public DefaultVectorCollection<V> clone()
    {
        return new DefaultVectorCollection<>(this);
    }
    
}
