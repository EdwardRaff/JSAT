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

package jsat.linear.vectorcollection;

/**
 *
 * @author Edward Raff
 */
public class IndexTuple implements Comparable<IndexTuple>
{
    public IndexNode a;
    public IndexNode b;
    double priority;

    public IndexTuple(IndexNode a, IndexNode b, double priority)
    {
        this.a = a;
        this.b = b;
        this.priority = priority;
    }
    
    

    @Override
    public int compareTo(IndexTuple o)
    {
        return Double.compare(this.priority, o.priority);
    }
    
}
