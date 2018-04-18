/*
 * Copyright (C) 2017 Edward Raff <Raff.Edward@gmail.com>
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

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import jsat.linear.Vec;

/**
 * This class exists primarily as a sanity/benchmarking utility. It takes a
 * given base distance metric, which will be used as the actual method of
 * measuring distances. This class will count how many times a distance
 * calculation was queried. This class is thread safe. <br>
 * NOTE: all clones of this object will share the same counter. 
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class DistanceCounter implements DistanceMetric
{
    private DistanceMetric base;
    private AtomicLong counter;

    /**
     * Creates a new distance counter to wrap the given base metric
     * @param base the base distance measure to use
     */
    public DistanceCounter(DistanceMetric base)
    {
        this.base = base;
        this.counter = new AtomicLong();
    }

    /**
     * Copies the given distance counter, while sharing the same underlying
     * counter between the original and this new object.
     *
     * @param toCopy the object to get a copy of
     */
    public DistanceCounter(DistanceCounter toCopy)
    {
        this.base = toCopy.base.clone();
        this.counter = toCopy.counter;
    }
    
    /**
     * 
     * @return the number of distance calls that have occurred
     */
    public long getCallCount()
    {
        return counter.get();
    }
    
    /**
     * Resets the distance counter calls to zero. 
     */
    public void resetCounter()
    {
        counter.set(0);
    }

    @Override
    public double dist(Vec a, Vec b)
    {
        counter.incrementAndGet();
        return base.dist(a, b);
    }

    @Override
    public boolean isSymmetric()
    {
        return base.isSymmetric();
    }

    @Override
    public boolean isSubadditive()
    {
        return base.isSubadditive();
    }

    @Override
    public boolean isIndiscemible()
    {
        return base.isIndiscemible();
    }

    @Override
    public double metricBound()
    {
        return base.metricBound();
    }

    @Override
    public boolean supportsAcceleration()
    {
        return base.supportsAcceleration();
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> vecs, boolean parallel)
    {
        return base.getAccelerationCache(vecs, parallel);
    }

    @Override
    public double dist(int a, int b, List<? extends Vec> vecs, List<Double> cache)
    {
        counter.incrementAndGet();
        return base.dist(a, b, vecs, cache);
    }

    @Override
    public double dist(int a, Vec b, List<? extends Vec> vecs, List<Double> cache)
    {
        counter.incrementAndGet();
        return base.dist(a, b, vecs, cache);
    }

    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        return base.getQueryInfo(q);
    }

    @Override
    public double dist(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        counter.incrementAndGet();
        return base.dist(a, b, qi, vecs, cache);
    }

    @Override
    public DistanceCounter clone()
    {
        return new DistanceCounter(this);
    }

}
