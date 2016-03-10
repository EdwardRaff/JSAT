
package jsat.utils;

import java.io.Serializable;
import java.util.*;

/**
 * Provides a modifiable implementation of a List using an array of longs. This 
 * provides considerable memory efficency improvements over using an 
 * {@link ArrayList} to store longs. 
 * Null is not allowed into the list. 
 * 
 * @author Edward Raff
 */
public class LongList extends AbstractList<Long> implements Serializable
{

	private static final long serialVersionUID = 3060216677615816178L;
	private long[] array;
    private int end;
    
    private LongList(long[] array, int end)
    {
        this.array = array;
        this.end = end;
    }

    /**
     * Creates a new LongList
     */
    public LongList()
    {
        this(10);
    }

    @Override
    public void clear()
    {
        end = 0;
    }

    /**
     * Creates a new LongList with the given initial capacity 
     * @param capacity the starting internal storage space size 
     */
    public LongList(int capacity)
    {
        array = new long[capacity];
        end = 0;
    }
    
    public LongList(Collection<Long> c)
    {
        this(c.size());
        addAll(c);
    }

    /**
     * Operates exactly as {@link #set(int, java.lang.Long) }
     * @param index the index to set
     * @param element the value to set
     * @return the previous value at the index
     */
    public long set(int index, long element)
    {
        boundsCheck(index);
        long prev = array[index];
        array[index] = element;
        return prev;
    }
    
    @Override
    public Long set(int index, Long element)
    {
        return set(index, element.longValue());
    }
    
    /**
     * Operates exactly as {@link #add(int, java.lang.Long) } 
     * @param index the index to add the value into
     * @param element the value to add
     */
    public void add(int index, long element)
    {
        if (index == size())//special case, just appending
        {
            add(element);
        }
        else
        {
            boundsCheck(index);
            enlargeIfNeeded(1);
            System.arraycopy(array, index, array, index+1, end-index);
            array[index] = element;
            end++;
        }
    }

    @Override
    public void add(int index, Long element)
    {
        add(index, element.longValue());
    }

    /**
     * Operates exactly as {@link #add(java.lang.Long) }
     * @param e the value to add
     * @return true if it was added, false otherwise
     */
    public boolean add(long e)
    {
        enlargeIfNeeded(1);
        array[end++] = e;
        return true;
    }
    
    @Override
    public boolean add(Long e)
    {
        if(e == null)
            return false;
        return add(e.longValue());
    }
    
    /**
     * Operates exactly as {@link #get(int) }
     * @param index the index of the value to get
     * @return the value at the index
     */
    public long getL(int index)
    {
        boundsCheck(index);
        return array[index];
    }
    
    @Override
    public Long get(int index)
    {
        return getL(index);
    }

    private void boundsCheck(int index) throws IndexOutOfBoundsException
    {
        if(index >= end)
            throw new IndexOutOfBoundsException("List of of size " + size() + ", index requested was " + index);
    }

    @Override
    public int size()
    {
        return end;
    }

    @Override
    public Long remove(int index)
    {
        if(index < 0 || index > size())
            throw new IndexOutOfBoundsException("Can not remove invalid index " + index);
        long removed = array[index];
        
        for(int i = index; i < end-1; i++)
            array[i] = array[i+1];
        end--;
        return removed;
    }

    private void enlargeIfNeeded(int i)
    {
        while(end+i > array.length)
            array = Arrays.copyOf(array, Math.max(array.length*2, 8));
    }
    
    /**
     * Creates and returns a view of the given long array that 
     * requires only a small object allocation. 
     * 
     * @param array the array to wrap into a list
     * @param length the number of values of the array to use, starting from zero
     * @return a list view of the array
     */
    public static List<Long> view(long[] array, int length)
    {
        return new LongList(array, length);
    }
}
