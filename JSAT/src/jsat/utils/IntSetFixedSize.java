package jsat.utils;

import java.io.Serializable;
import java.util.AbstractSet;
import java.util.Iterator;

/**
 * A set for integers that is of a fixed initial size, and can only accept 
 * integers in the range [0, size). Insertions, removals, and checks are all 
 * constant time with a fast iterator. <br>
 * Null values are not supported
 * 
 * @author Edward Raff
 */
public class IntSetFixedSize extends AbstractSet<Integer> implements Serializable
{

	private static final long serialVersionUID = 7743166074116253587L;
	private static final int STOP = -1;
    private int nnz = 0;
    private int first = -1;
    private boolean[] has;
    //Use as a linked list
    private int[] prev;
    private int[] next;
    
    /**
     * Creates a new fixed size int set
     * @param size the size of the int set
     */
    public IntSetFixedSize(int size)
    {
        has = new boolean[size];
        prev = new int[size];
        next = new int[size];
        first = STOP;
    }

    @Override
    public boolean add(Integer e)
    {
        return add(e.intValue());
    }
    
    /**
     * Adds a new integer into the set
     * @param e the value to add into the set
     * @return {@code true} if the operation modified the set, {@code false} 
     * otherwise. 
     */
    public boolean add(int e)
    {
        if(e < 0 || e >= has.length)
            throw new IllegalArgumentException("Input must be in range [0, " + has.length + ") not " + e);
        else if(contains(e) )
            return false;
        else
        {
            if (nnz == 0)
            {
                first = e;
                next[e] = prev[e] = STOP;
            }
            else
            {
                prev[first] = e;
                next[e] = first;
                prev[e] = STOP;
                first = e;
            }
            nnz++;
            return has[e] = true;
        }
    }

    @Override
    public boolean remove(Object o)
    {
        if(o instanceof Integer)
            return remove_int((Integer)o);
        return super.remove(o);
    }
    
    /**
     * Removes the specified integer from the set
     * @param val the value to remove 
     * @return {@code true} if the set was modified by this operation, 
     * {@code false} if it was not.
     */
    public boolean remove(int val)
    {
        return remove_int(val);
    }

    @Override
    public boolean contains(Object o)
    {
        if(o instanceof Integer)
        {
            int val = (Integer)o;
            return contains(val);
        }
        else
            return false;
    }
    
    /**
     * Checks if the given value is contained in this set
     * @param val the value to check for 
     * @return {@code true} if the value is in the set, {@code false} otherwise.
     */
    public boolean contains(int val)
    {
        if(val < 0 || val >= has.length)
            return false;
        return has[val];
    }
    
    
    private boolean remove_int(int index)
    {
        if (contains(index))
        {
            if (first == index)
                first = next[index];
            else
                next[prev[index]] = next[index];
            
            if (next[index] != STOP)
                prev[next[index]] = prev[index];
            next[index] = STOP;
            has[index] = false;
            nnz--;
            return true;
        }
        else
            return false;
    }
    

    @Override
    public Iterator<Integer> iterator()
    {
        final Iterator<Integer> iterator = new Iterator<Integer>() 
        {
            int prev = STOP;
            int runner = first;
            @Override
            public boolean hasNext()
            {
                return runner != STOP;
            }

            @Override
            public Integer next()
            {
                prev = runner;
                runner = next[runner];
                return prev;
            }

            @Override
            public void remove()
            {
                remove_int(prev);
            }
        };
        return iterator;
    }

    @Override
    public int size()
    {
        return nnz;
    }
    
}
