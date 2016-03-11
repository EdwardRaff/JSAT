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
package jsat.clustering.dissimilarity;

import jsat.linear.distancemetrics.DistanceMetric;

/**
 * Median link dissimilarity, also called WPGMC. When two points are merged
 * under the Median dissimilarity, the weighting to all points in every
 * clustering is distributed evenly.
 *
 * @author Edward Raff
 */
public class MedianDissimilarity extends LanceWilliamsDissimilarity
{
    public MedianDissimilarity(DistanceMetric dm)
    {
        super(dm);
    }
    
    public MedianDissimilarity(MedianDissimilarity toCopy)
    {
        super(toCopy);
    }

    @Override
    protected double aConst(boolean iFlag, int ni, int nj, int nk)
    {
        return 0.5;
    }

    @Override
    protected double bConst(int ni, int nj, int nk)
    {
        return -0.25;
    }

    @Override
    protected double cConst(int ni, int nj, int nk)
    {
        return 0;
    }

    @Override
    public MedianDissimilarity clone()
    {
        return new MedianDissimilarity(this);
    }
    
}
