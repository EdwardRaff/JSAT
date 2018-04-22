
package jsat.utils;

import java.io.Serializable;
import java.util.*;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * Provides a modifiable implementation of a List using a boolean array. This provides considerable
 * memory efficency improvements over using an {@link ArrayList} to store booleans. <br>
 * Null is not allowed into the list. 
 * 
 * @author Edward Raff
 */
public class BooleanList extends AbstractList<Boolean> implements Serializable, RandomAccess
{

    private static final long serialVersionUID = 653930294509274337L;
    private boolean[] array;
    //Exclusive
    private int end;

    private BooleanList(boolean[] array, int end)
    {
        this.array = array;
        this.end = end;
    }

    @Override
    public void clear()
    {
        end = 0;
    }

    /**
     * Creates a new empty BooleanList
     */
    public BooleanList()
    {
        this(10);
    }
    
    /**
     * Creates a new empty BooealList
     * 
     * @param capacity the starting internal capacity of the list
     */
    public BooleanList(int capacity)
    {
        this(new boolean[capacity], 0);
    }
    
    /**
     * Creates a new BooleanList containing the values of the given collection
     * @param c the collection of values to fill this boolean list with
     */
    public BooleanList(Collection<Boolean> c)
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
     * Performs exactly the same as {@link #add(java.lang.Boolean) }. 
     * @param e the value to add
     * @return true if it was added, false otherwise
     */
    public boolean add(boolean e)
    {
        enlageIfNeeded(1);
        array[end] = e;
        increasedSize(1);
        return true;
    }
    
    /**
     * This method treats the underlying list as a stack. 
     * Pushes an item onto the top of this "stack". 
     * @param e the item to push onto the stack
     * @return the value added to the stack
     */
    public boolean push(boolean e)
    {
        add(e);
        return e;
    }
    
    /**
     * This method treats the underlying list as a stack. Removes the item at
     * the top of this "stack" and returns that item as the value.
     *
     * @return the item at the top of this stack (the last item pushed onto it)
     */
    public boolean pop()
    {
        if(isEmpty())
            throw new EmptyStackException();
        return removeB(size()-1);
    }

    /**
     * This method treats the underlying list as a stack. Gets the item at the
     * top of this "stack" and returns that item as the value, but leaves it on
     * the stack.
     *
     * @return the item at the top of this stack (the last item pushed onto it)
     */
    public boolean peek()
    {
        if(isEmpty())
            throw new EmptyStackException();
        return get(size()-1);
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
            array = Arrays.copyOf(array, Math.max(array.length*2, 8));
    }
    
    @Override
    public boolean add(Boolean e)
    {
        if(e == null)
            return false;
        return add(e.booleanValue());
    }

    /**
     * Operates exactly as {@link #get(int) }
     * @param index the index of the value to get
     * @return the value at the given index
     */
    public boolean getB(int index)
    {
        boundsCheck(index);
        return array[index];
    }
    
    @Override
    public Boolean get(int index)
    {
        return getB(index);
    }
    
    /**
     * Operates exactly as {@link #set(int, java.lang.Boolean) }
     * @param index the index to set
     * @param element the value to set
     * @return the previous value at said index
     */
    public boolean set(int index, boolean element)
    {
        boundsCheck(index);
        boolean ret = get(index);
        array[index] = element;
        return ret;
    }

    @Override
    public Boolean set(int index, Boolean element)
    {
        return set(index, element.booleanValue());
    }

    /**
     * Operates exactly as {@link #add(int, java.lang.Boolean) }
     * @param index the index to add at
     * @param element the value to add
     */
    public void add(int index, boolean element)
    {
        if(index == size())//special case, just appending
        {
            add(element);
        }
        else
        {
            boundsCheck(index);
            enlageIfNeeded(1);
            System.arraycopy(array, index, array, index+1, size()-index);
            set(index, element);
            increasedSize(1);
        }
    }
    
    @Override
    public void add(int index, Boolean element)
    {
        add(index, element.booleanValue());
    }

    /**
     * Operates exactly as {@link #remove(int) }
     * @param index the index to remove
     * @return the value removed
     */
    public boolean removeB(int index)
    {
        boundsCheck(index);
        boolean ret = array[index];
        for(int i = index; i < end-1; i++)
            array[i] = array[i+1];
        decreaseSize(1);
        return ret;
    }
 
    @Override
    public Boolean remove(int index)
    {
        return removeB(index);
    }
    
    /**
     * Returns the reference to the array that backs this list. 
     * Alterations to the array will be visible to the DoubelList
     * and vise versa. The array returned may not the the same
     * size as the value returned by {@link #size() }
     * @return the underlying array used by this BooleanList
     */
    public boolean[] getBackingArray()
    {
        return array;
    }
    
    
    /**
     * Creates an returns an unmodifiable view of the given boolean array that requires 
     * only a small object allocation. 
     * 
     * @param array the array to wrap into an unmodifiable list
     * @param length the number of values of the array to use, starting from zero
     * @return an unmodifiable list view of the array
     */
    public static List<Boolean> unmodifiableView(boolean[] array, int length)
    {
        return Collections.unmodifiableList(view(array, length));
    }
    
    /**
     * Creates and returns a view of the given boolean array that requires only
     * a small object allocation. Changes to the list will be reflected in the 
     * array up to a point. If the modification would require increasing the 
     * capacity of the array, a new array will be allocated - at which point 
     * operations will no longer be reflected in the original array. 
     * 
     * @param array the array to wrap by a BooleanList object
     * @param length the initial length of the list
     * @return a BoolaenList backed by the given array, unless modified to the 
     * point of requiring the allocation of a new array
     */
    public static BooleanList view(boolean[] array, int length)
    {
        if(length > array.length || length < 0)
            throw new IllegalArgumentException("length must be non-negative and no more than the size of the array("+array.length+"), not " + length);
        return new BooleanList(array, length);
    }
}
