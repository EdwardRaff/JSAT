
package jsat.utils;

import java.io.Serializable;
import java.util.*;

/**
 * Provides a modifiable implementation of a List using a int array. This provides considerable
 * memory efficency improvements over using an {@link ArrayList} to store integers. 
 * Null is not allowed into the list. 
 * 
 * @author Edward Raff
 */
public class IntList extends AbstractList<Integer> implements Serializable
{

    private static final long serialVersionUID = 8189142393005394477L;
    private int[] array;
    private int end;
    
    private IntList(int[] array, int end)
    {
        this.array = array;
        this.end = end;
    }

    /**
     * Creates a new IntList
     */
    public IntList()
    {
        this(10);
    }

    @Override
    public void clear()
    {
        end = 0;
    }

    /**
     * Creates a new IntList with the given initial capacity 
     * @param capacity the starting internal storage space size 
     */
    public IntList(int capacity)
    {
        array = new int[capacity];
        end = 0;
    }
    
    public IntList(Collection<Integer> c)
    {
        this(c.size());
        addAll(c);
    }

    /**
     * Operates exactly as {@link #set(int, java.lang.Integer) }
     * @param index the index to set
     * @param element the value to set
     * @return the previous value at the index
     */
    public int set(int index, int element)
    {
        boundsCheck(index);
        int prev = array[index];
        array[index] = element;
        return prev;
    }
    
    @Override
    public Integer set(int index, Integer element)
    {
        return set(index, element.intValue());
    }
    
    /**
     * Operates exactly as {@link #add(int, java.lang.Integer) } 
     * @param index the index to add the value into
     * @param element the value to add
     */
    public void add(int index, int element)
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
    public void add(int index, Integer element)
    {
        add(index, element.intValue());
    }

    /**
     * Operates exactly as {@link #add(java.lang.Integer) }
     * @param e the value to add
     * @return true if it was added, false otherwise
     */
    public boolean add(int e)
    {
        enlargeIfNeeded(1);
        array[end++] = e;
        return true;
    }
    
    @Override
    public boolean add(Integer e)
    {
        if(e == null)
            return false;
        return add(e.intValue());
    }
    
    /**
     * Operates exactly as {@link #get(int) }
     * @param index the index of the value to get
     * @return the value at the index
     */
    public int getI(int index)
    {
        boundsCheck(index);
        return array[index];
    }
    
    @Override
    public Integer get(int index)
    {
        return getI(index);
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
    public boolean addAll(Collection<? extends Integer> c)
    {
        int initSize = size();
        enlargeIfNeeded(c.size());
        for(int i : c)
            add(i);
        return initSize != size();
    }
    
    

    @Override
    public Integer remove(int index)
    {
        if(index < 0 || index > size())
            throw new IndexOutOfBoundsException("Can not remove invalid index " + index);
        int removed = array[index];
        
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
     * Efficiently sorts this list of integers using {@link Arrays#sort(int[])
     * }.
     */
    public void sort()
    {
        Arrays.sort(array, 0, end);
    }

    /**
     * Creates and returns an unmodifiable view of the given int array that 
     * requires only a small object allocation. 
     * 
     * @param array the array to wrap into an unmodifiable list
     * @param length the number of values of the array to use, starting from zero
     * @return an unmodifiable list view of the array
     */
    public static List<Integer> unmodifiableView(int[] array, int length)
    {
        return Collections.unmodifiableList(view(array, length));
    }
    
    /**
     * Creates and returns a view of the given int array that requires only
     * a small object allocation. Changes to the list will be reflected in the 
     * array up to a point. If the modification would require increasing the 
     * capacity of the array, a new array will be allocated - at which point 
     * operations will no longer be reflected in the original array. 
     * 
     * @param array the array to wrap by an IntList object
     * @param length the initial length of the list
     * @return an IntList backed by the given array, unless modified to the 
     * point of requiring the allocation of a new array
     */
    public static IntList view(int[] array, int length)
    {
        if(length > array.length || length < 0)
            throw new IllegalArgumentException("length must be non-negative and no more than the size of the array("+array.length+"), not " + length);
        return new IntList(array, length);
    }
}
