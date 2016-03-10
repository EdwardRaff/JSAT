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
package jsat.utils;

import java.util.AbstractMap;
import java.util.AbstractSet;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Set;

/**
 * Provides a map from integers to doubles backed by index into an array.
 * {@link Double#NaN} may not be inserted and negative numbers may not be used
 * for the key. This is meant as a convenience class, and should be used
 * sparingly.
 * 
 * @author Edward Raff
 */
public class IntDoubleMapArray extends AbstractMap<Integer, Double>
{
    
    private int used = 0;
    private double[] table;
    
    public IntDoubleMapArray()
    {
        this(1);
    }
    
    public IntDoubleMapArray(int max_index)
    {
        table = new double[max_index];
        Arrays.fill(table, Double.NaN);
    }
    
    @Override
    public Double put(Integer key, Double value)
    {
        double prev = put(key.intValue(), value.doubleValue());
        if(Double.isNaN(prev))
            return null;
        else
            return prev;
    }
    
    public double put(int key, double value)
    {
        if(Double.isNaN(value))
            throw new IllegalArgumentException("NaN is not an allowable value");
        
        //epand with NaNs to fit index
        if(table.length <= key)
        {
            int oldLen = table.length;
            table = Arrays.copyOf(table, key+1);
            Arrays.fill(table, oldLen, table.length, Double.NaN);
        }
        
        double prev = table[key];
        table[key] = value;
        
        if(Double.isNaN(prev))
            used++;
        
        return prev;
    }

    /**
     * 
     * @param key the key whose associated value is to be incremented. All
     * non-present keys behave as having an implicit value of zero, in which
     * case the delta value is directly inserted into the map.
     * @param delta the amount by which to increment the key's stored value.
     * @return the new value stored for the given key
     */
    public double increment(int key, double delta)
    {
        if(Double.isNaN(delta))
            throw new IllegalArgumentException("NaN is not an allowable value");
        
        if(table.length <= key)
            put(key, delta);
        else if(Double.isNaN(table[key]))
        {
            table[key] = delta;
            used++;
        }
        else
            table[key] += delta;
        
        return table[key];
    }
    
    /**
     * Returns the value to which the specified key is mapped, or
     * {@link Double#NaN} if this map contains no mapping for the key.
     *
     * @param key the key whose associated value is to be returned
     * @return the value to which the specified key is mapped, or
     * {@link Double#NaN} if this map contains no mapping for the key
     */
    public double get(int key)
    {
        if(table.length <= key)
            return Double.NaN;
        return table[key];
    }

    @Override
    public Double get(Object key)
    {
        if(key == null)
            return null;
        
        if(key instanceof Integer)
        {
            double d = get( ((Integer)key).intValue());
            if(Double.isNaN(d))
                return null;
            return d;
        }
        else
            throw new ClassCastException("Key not of integer type");
    }
    
    @Override
    public Double remove(Object key)
    {
        if(key instanceof Integer)
        {
            double oldValue = remove(((Integer)key).intValue());
            if(Double.isNaN(oldValue))
                return null;
            else
                return oldValue;
        }
        
        return null;
    }
    
    /**
     * 
     * @param key
     * @return the old value stored for this key, or {@link Double#NaN} if the
     * key was not present in the map
     */
    public double remove(int key)
    {
        if(table.length <= key)
            return Double.NaN;
        double toRet = table[key];
        table[key] = Double.NaN;
        if(!Double.isNaN(toRet))
            used--;
        return toRet;
    }
    
    @Override
    public void clear()
    {
        used = 0;
        Arrays.fill(table, Double.NaN);
    }
    
    @Override
    public boolean containsKey(Object key)
    {
        if(key instanceof Integer)
            return containsKey( ((Integer)key).intValue());
        else
            return false;
    }
    
    public boolean containsKey(int key)
    {
        if(table.length <= key)
            return false;
        return !Double.isNaN(table[key]);
    }
    

    @Override
    public Set<Entry<Integer, Double>> entrySet()
    {
        return new EntrySet(this);
    }

    
    /**
     * EntrySet class supports remove operations
     */
    private final class EntrySet extends AbstractSet<Entry<Integer, Double>>
    {
        final IntDoubleMapArray parentRef;
        
        public EntrySet(IntDoubleMapArray parentRef)
        {
            this.parentRef = parentRef;
        }

        @Override
        public Iterator<Entry<Integer, Double>> iterator()
        {
            return new Iterator<Entry<Integer, Double>>()
            {
                int curPos = -1;
                int nexPos = curPos;
                @Override
                public boolean hasNext()
                {
                    if(nexPos < curPos)//indicates we are out
                        return false;
                    else if(nexPos > curPos && nexPos < table.length)
                        return true;
                    //else, not sure yet - lets find the next pos
                    nexPos = curPos+1;
                    while(nexPos < table.length && Double.isNaN(table[nexPos]))
                        nexPos++;
                    if(nexPos >= table.length)
                        return false;
                    return true;
                }

                @Override
                public Entry<Integer, Double> next()
                {
                    if(!hasNext())
                        throw new RuntimeException();
                    curPos = nexPos;
                    return new Entry<Integer, Double>()
                    {
                        @Override
                        public Integer getKey()
                        {
                            return curPos;
                        }

                        @Override
                        public Double getValue()
                        {
                            return table[curPos];
                        }

                        @Override
                        public Double setValue(Double value)
                        {
                            double old = table[curPos];
                            table[curPos] = value;
                            return old;
                        }
                    };
                }

                @Override
                public void remove()
                {
                    table[curPos] = Double.NaN;
                    used--;
                }

            };
        }

        @Override
        public int size()
        {
            return used;
        }
        
    }
}
