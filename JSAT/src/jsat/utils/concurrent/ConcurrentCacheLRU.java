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
package jsat.utils.concurrent;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import jsat.utils.Pair;

/**
 * This class defines a Concurrent LRU cache. The current implementation may not
 * be the most consistent, as it uses a time stamp from last access to determine
 * LRU items. It is also designed for read-heavy work loads
 *
 * @author Edward Raff
 */
public class ConcurrentCacheLRU<K, V>
{
    private final ConcurrentHashMap<K, Pair<V, AtomicLong>> cache;
    private final int maxEntries;

    public ConcurrentCacheLRU(int max_entries)
    {
        this.maxEntries = max_entries;
        cache = new ConcurrentHashMap<K, Pair<V, AtomicLong>>(max_entries);
    }

    public V putIfAbsentAndGet(K key, V value)
    {
        Pair<V, AtomicLong> pair = cache.putIfAbsent(key, new Pair<V, AtomicLong>(value, new AtomicLong(System.currentTimeMillis())));
        
        evictOld();
        
        if(pair == null)
            return null;
        return pair.getFirstItem();
    }

    private void evictOld()
    {
        while(cache.size() > maxEntries)
        {
            K oldest_key = null;
            long oldest_time = Long.MAX_VALUE;
            for(Map.Entry<K, Pair<V, AtomicLong>> entry : cache.entrySet())
                if(entry.getValue().getSecondItem().get() < oldest_time)
                {
                    oldest_time = entry.getValue().getSecondItem().get();
                    oldest_key = entry.getKey();
                }
            if(cache.size() > maxEntries)//anotehr thread may have already evicted things
                cache.remove(oldest_key);
        }
    }
    
    public void put(K key, V value)
    {
        cache.put(key, new Pair<V, AtomicLong>(value, new AtomicLong(System.currentTimeMillis())));

        evictOld();
    }

    public V get(K key)
    {
        Pair<V, AtomicLong> pair = cache.get(key);
        if(pair == null)
            return null;
        pair.getSecondItem().set(System.currentTimeMillis());
        return pair.getFirstItem();
    }
    
}
