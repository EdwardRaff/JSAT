package jsat.utils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

/**
 *
 * @author Edward Raff
 */
public class BoundedSortedList<E extends Comparable<E>> extends ArrayList<E>
{
    int maxSize;

    public BoundedSortedList(int maxSize, int initialCapacity)
    {
        super(initialCapacity);
        if(maxSize < 1)
            throw new RuntimeException("Invalid max size");
        this.maxSize = maxSize;
    }

    public BoundedSortedList(int maxSize)
    {
        if(maxSize < 1)
            throw new RuntimeException("Invalid max size");
        this.maxSize = maxSize;
    }
        

    @Override
    public boolean add(E e)
    {
        if(isEmpty())
        {
            super.add(e);
            return true;
        }
        else
        {
            int ind = Collections.binarySearch(this, e);
            if (ind >= 0)//it is already in the list, 
            {
                if (size() == maxSize)//pop the last, put this 
                {
                    this.remove(maxSize - 1);
                    super.add(ind, e);
                }
                else//not full yet, can jsut add
                    super.add(e);
                return true;
            }
            else
            {
                ind = -(ind + 1);//Now it is the point where it should be inserted
                if (size() < maxSize)
                    super.add(ind, e);
                else if (ind < maxSize)
                {
                    this.remove(maxSize - 1);
                    super.add(ind, e);
                }
                else
                    return false;

                return true;
            }
        }
    }


    public E first()
    {
        if(isEmpty())
            return null;
        return get(0);
    }

    @Override
    public void add(int index, E element)
    {
        add(element);
    }
    
}
