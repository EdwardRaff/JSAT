
package jsat.utils;

import java.io.Serializable;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * Provides a modifiable implementation of a List using a double array. This provides considerable
 * memory efficency improvements over using an {@link ArrayList} to store doubles. <br>
 * Null is not allowed into the list. 
 * 
 * @author Edward Raff
 */
public class DoubleList extends AbstractList<Double> implements Serializable
{
    private double[] array;
    //Exclusive
    private int end;

    private DoubleList(double[] array, int end)
    {
        this.array = array;
        this.end = end;
    }

    /**
     * Creates a new empty DoubleList
     */
    public DoubleList()
    {
        this(10);
    }
    
    /**
     * Creates a new empty DoubleList
     * 
     * @param capacity the starting internal capacity of the list
     */
    public DoubleList(int capacity)
    {
        this(new double[capacity], 0);
    }
    
    /**
     * Creates a new DoubleList containing the values of the given collection
     * @param c the collection of values to fill this double list with
     */
    public DoubleList(Collection<Double> c)
    {
        this(c.size());
        this.addAll(c);
    }

    @Override
    public int size()
    {
        return end;
    }

    /**
     * Performs exactly the same as {@link #add(java.lang.Double) }. 
     * @param e the value to add
     * @return true if it was added, false otherwise
     */
    public boolean add(double e)
    {
        enlageIfNeeded(1);
        array[end] = e;
        increasedSize(1);
        return true;
    }

    /**
     * Makes the changes indicating that a number of items have been removed
     * @param removed the number of items that were removed 
     */
    private void decreaseSize(int removed)
    {
        end-=removed;
    }

    /**
     * Marks the increase of size of this list, and reflects the change in the parent 
     */
    private void increasedSize(int added)
    {
        end+=added;
    }

    private void boundsCheck(int index) throws IndexOutOfBoundsException
    {
        if(index >= size())
            throw new IndexOutOfBoundsException("List is of size " + size() + ", index requested " + index);
    }

    /**
     * Enlarge the storage array if needed
     * @param i the amount of elements we will need to add
     */
    private void enlageIfNeeded(int i)
    {
        while(end+i > array.length)
            array = Arrays.copyOf(array, array.length*2);
    }
    
    @Override
    public boolean add(Double e)
    {
        if(e == null)
            return false;
        return add(e.doubleValue());
    }

    /**
     * Operates exactly as {@link #get(int) }
     * @param index the index of the value to get
     * @return the value at the given index
     */
    public double getD(int index)
    {
        boundsCheck(index);
        return array[index];
    }
    
    @Override
    public Double get(int index)
    {
        return getD(index);
    }
    
    /**
     * Operates exactly as {@link #set(int, java.lang.Double) }
     * @param index the index to set
     * @param element the value to set
     * @return the previous value at said index
     */
    public double set(int index, double element)
    {
        boundsCheck(index);
        double ret = get(index);
        array[index] = element;
        return ret;
    }

    @Override
    public Double set(int index, Double element)
    {
        return set(index, element.doubleValue());
    }

    /**
     * Operates exactly as {@link #add(int, java.lang.Double) }
     * @param index the index to add at
     * @param element the value to add
     */
    public void add(int index, double element)
    {
        boundsCheck(index);
        enlageIfNeeded(1);
        System.arraycopy(array, index, array, index+1, size()-index);
        set(index, element);
        increasedSize(1);
    }
    
    @Override
    public void add(int index, Double element)
    {
        add(index, element.doubleValue());
    }

    /**
     * Operates exactly as {@link #remove(int) }
     * @param index the index to remove
     * @return the value removed
     */
    public double removeD(int index)
    {
        boundsCheck(index);
        double ret = array[index];
        System.arraycopy(array, index+1, array, index, end-index+1);
        decreaseSize(1);
        return ret;
    }
 
    @Override
    public Double remove(int index)
    {
        return removeD(index);
    }
    
    /**
     * Returns the reference to the array that backs this list. 
     * Alterations to the array will be visible to the DoubelList
     * and vise versa. The array returned may not the the same
     * size as the value returned by {@link #size() }
     * @return the underlying array used by this DoubleList
     */
    public double[] getBackingArray()
    {
        return array;
    }
    
    /**
     * Creates an returns an unmodifiable view of the given double array that requires 
     * only a small object allocation. 
     * 
     * @param array the array to wrap into an unmodifiable list
     * @param length the number of values of the array to use, starting from zero
     * @return an unmodifiable list view of the array
     */
    public static List<Double> unmodifiableView(double[] array, int length)
    {
        return Collections.unmodifiableList(new DoubleList(array, length));
    }
}
