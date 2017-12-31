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

/**
 * This class exists as a helper method for use with nearest neighbor
 * implementations. It stores an integer to represent the index of a vector, and
 * a double to store the distance of the index to a given query.
 *
 * @author Edward Raff
 */
public class IndexDistPair implements Comparable<IndexDistPair>
{
    /**
     * the index of a vector
     */
    protected int indx;
    
    /**
     * the distance of this index to a query vector
     */
    protected double dist;

    public IndexDistPair(int indx, double dist)
    {
        this.indx = indx;
        this.dist = dist;
    }
    
    public int getIndex()
    {
        return indx;
    }
    
    public void setIndex(int indx)
    {
        this.indx = indx;
    }
    
    public double getDist()
    {
        return dist;
    }
    
    public void setDist(double dist)
    {
        this.dist = dist;
    }

    @Override
    public int compareTo(IndexDistPair o)
    {
        return Double.compare(this.dist, o.dist);
    }
}
