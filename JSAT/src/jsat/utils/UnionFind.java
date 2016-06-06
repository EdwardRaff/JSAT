/*
 * Copyright (C) 2016 Edward Raff
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
package jsat.utils;

/**
 *
 * @author Edward Raff
 */
public class UnionFind<X>
{
    protected UnionFind<X> parent;
    /**
     * Really the depth of the tree, but terminology usually is rank
     */
    protected int rank;
    protected X item;

    public UnionFind()
    {
        this(null);
    }

    public X getItem()
    {
        return item;
    }
    
    public UnionFind(X item)
    {
        this.parent = this;//yes, this is intentional. 
        this.rank = 0;
        this.item = item;
    }
    
    public UnionFind<X> find()
    {
        if(this.parent != this)
            this.parent = this.parent.find();
        return this.parent;
    }

    public void union(UnionFind<X> y)
    {
        UnionFind<X> xRoot = this.find();
        UnionFind<X> yRoot = y.find();
        if (xRoot == yRoot)
            return; // x and y are not already in same set. Merge them.
        
        if (xRoot.rank < yRoot.rank)
            xRoot.parent = yRoot;
        else if (xRoot.rank > yRoot.rank)
            yRoot.parent = xRoot;
        else
        {
            yRoot.parent = xRoot;
            xRoot.rank = xRoot.rank + 1;
        }
    }
    
}
