package jsat.utils;

import java.util.AbstractMap;
import java.util.AbstractSet;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import static jsat.utils.ClosedHashingUtil.*;

/**
 * A hash map for storing the primitive types of integers (as keys) to doubles 
 * (as table). The implementation is based on Algorithm D (Open addressing with
 * double hashing) from Knuth's TAOCP page 528. 
 * 
 * @author Edward Raff
 */
public final class IntDoubleMap extends AbstractMap<Integer, Double>
{
    
    private float loadFactor;

    private int used = 0;
    private byte[] status;
    private int[] keys;
    private double[] table;
    
    public IntDoubleMap()
    {
        this(32);
    }
    
    public IntDoubleMap(int capacity)
    {
        this(capacity, 0.75f);
    }
    
    public IntDoubleMap(Map<Integer, Double> collection)
    {
        this(Math.max(1, collection.size()));
        for(Entry<Integer, Double> entry : collection.entrySet())
            put(entry.getKey(), entry.getValue());
    }
    
    public IntDoubleMap(int capacity, float loadFactor)
    {
        if(capacity < 1)
            throw new IllegalArgumentException("Capacity must be a positive value, not " + capacity);
        if(loadFactor <= 0 || loadFactor >= 1 || Float.isNaN(loadFactor))
            throw new IllegalArgumentException("loadFactor must be in (0, 1), not " + loadFactor);
        this.loadFactor = loadFactor;
        
        int size = getNextPow2TwinPrime((int) Math.max(capacity/loadFactor, 4));
        status = new byte[size];
        keys = new int[size];
        table = new double[size];
    }

    @Override
    public int size()
    {
        return used;
    }
    
    public int[] getRawKeyTable()
    {
        return keys;
    }
    
    public double[] getRawValueTable()
    {
        return table;
    }
    
    public byte[] getRawStatusTable()
    {
        return status;
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
        long pair_index = getIndex(key);
        int deletedIndex = (int) (pair_index >>> 32);
        int valOrFreeIndex = (int) (pair_index & INT_MASK);
        
        double prev;
        if(status[valOrFreeIndex] == OCCUPIED)//easy case
        {
            prev = table[valOrFreeIndex];
            table[valOrFreeIndex] = value;
            return prev;
        }
        //else, not present
        prev = Double.NaN;
        int i = valOrFreeIndex;
        if(deletedIndex >= 0)//use occupied spot instead
            i = deletedIndex;
        
        status[i] = OCCUPIED;
        keys[i] = key;
        table[i] = value;
        used++;
        
        enlargeIfNeeded();
        
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
        
        long pair_index = getIndex(key);
        int deletedIndex = (int) (pair_index >>> 32);
        int valOrFreeIndex = (int) (pair_index & INT_MASK);
        
        if(status[valOrFreeIndex] == OCCUPIED)//easy case
            return (table[valOrFreeIndex] += delta);
        //else, not present
        double toReturn;
        int i = valOrFreeIndex;
        if(deletedIndex >= 0)//use occupied spot instead
            i = deletedIndex;
        
        status[i] = OCCUPIED;
        keys[i] = key;
        toReturn = table[i] = delta;
        used++;
        
        enlargeIfNeeded();
        
        return toReturn;
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
        long pair_index = getIndex(key);
        int valOrFreeIndex = (int) (pair_index & INT_MASK);
        
        if(status[valOrFreeIndex] == OCCUPIED)//easy case
            return table[valOrFreeIndex];
        //else, return NaN for missing
        return Double.NaN;
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
        long pair_index = getIndex(key);
        int valOrFreeIndex = (int) (pair_index & INT_MASK);
        if(status[valOrFreeIndex] == EMPTY)//ret index is always EMPTY or OCCUPIED
            return Double.NaN;
        //else
        double toRet = table[valOrFreeIndex];
        status[valOrFreeIndex] = DELETED;
        used--;
        return toRet;
    }
    
    @Override
    public void clear()
    {
        used = 0;
        Arrays.fill(status, EMPTY);
    }
    
    private void enlargeIfNeeded()
    {
        if(used < keys.length*loadFactor)
            return;
        //enlarge
        final byte[] oldSatus = status;
        final int[] oldKeys = keys;
        final double[] oldTable = table;
        
        int newSize = getNextPow2TwinPrime(status.length*3/2);//it will actually end up doubling in size since we have twin primes spaced that was
        status = new byte[newSize];
        keys = new int[newSize];
        table = new double[newSize];
        
        used = 0;
        for(int oldIndex = 0; oldIndex < oldSatus.length; oldIndex++)
            if(oldSatus[oldIndex] == OCCUPIED)
                put(oldKeys[oldIndex], oldTable[oldIndex]);
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
        int index = (int) (getIndex(key) & INT_MASK);
        return status[index] == OCCUPIED;//would be FREE if we didn't have the key
    }
    
    
    private final static long EXTRA_INDEX_INFO = ((long)Integer.MIN_VALUE) << 32;
    
    /**
     * Gets the index of the given key. Based on that {@link #status} variable,
     * the index is either the location to insert OR the location of the key.
     * 
     * This method returns 2 integer table in the long. The lower 32 bits are
     * the index that either contains the key, or is the first empty index. 
     * 
     * The upper 32 bits is the index of the first position marked as
     * {@link #DELETED} either {@link Integer#MIN_VALUE} if no position was
     * marked as DELETED while searching.
     *
     * @param key they key to search for
     * @return the mixed long containing the index of the first DELETED position
     * and the position that the key is in or the first EMPTY position found
     */
    private long getIndex(int key)
    {
        long extraInfo = EXTRA_INDEX_INFO;
        //D1 
        final int hash = h(key) & INT_MASK;
        int i = hash % keys.length;
        
        //D2
        int satus_i = status[i];
        if((keys[i] == key && satus_i != DELETED) || satus_i == EMPTY)
            return extraInfo | i;
        if(extraInfo == EXTRA_INDEX_INFO && satus_i == DELETED)
            extraInfo = ((long)i) << 32;
        
        //D3
        final int c = 1 + (hash % (keys.length -2));
        
        while(true)//this loop will terminate
        {
            //D4
            i -= c;
            if(i < 0)
                i += keys.length;
            //D5
            satus_i = status[i];
            if( (keys[i] == key && satus_i != DELETED) || satus_i == EMPTY)
                return extraInfo | i;
            if(extraInfo == EXTRA_INDEX_INFO && satus_i == DELETED)
                extraInfo = ((long)i) << 32;
        }
    }
    
    /**
     * Returns a non-negative hash value 
     * @param key
     * @return 
     */
    public static int h(int key)
    {
        //http://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
        key = ((key >>> 16) ^ key ) & 0x45f9f3b;
        key = ((key >>> 16) ^ key ) & 0x45f9f3b;
        key = ((key >>> 16) ^ key ) ;
        return key;
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
        final IntDoubleMap parentRef;
        
        public EntrySet(IntDoubleMap parent)
        {
            this.parentRef = parent;
        }

        @Override
        public Iterator<Entry<Integer, Double>> iterator()
        {
            //find the first starting inded
            int START = 0;
            while(START < status.length && status[START] != OCCUPIED)
                START++;
            if(START == status.length)
                return Collections.emptyIterator();
            final int startPos = START;
            
            return new Iterator<Entry<Integer, Double>>()
            {
                int pos = startPos;
                int prevPos = -1;

                @Override
                public boolean hasNext()
                {
                    return pos < status.length;
                }

                @Override
                public Entry<Integer, Double> next()
                {
                    //final int make so that object remains good after we call next again
                    final int oldPos = prevPos = pos++;
                    //find next
                    while (pos < status.length && status[pos] != OCCUPIED)
                        pos++;
                    //and return new object
                    return new Entry<Integer, Double>()
                    {

                        @Override
                        public Integer getKey()
                        {
                            return keys[oldPos];
                        }

                        @Override
                        public Double getValue()
                        {
                            return table[oldPos];
                        }

                        @Override
                        public Double setValue(Double value)
                        {
                            double old = table[oldPos];
                            table[oldPos] = value;
                            return old;
                        }
                    };
                }

                @Override
                public void remove()
                {
                    //its ok to just call remove b/c nothing is re-ordered when we remove an element, we just set the status to DELETED
                    parentRef.remove(keys[prevPos]);
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
