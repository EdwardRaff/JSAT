
package jsat.utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

/**
 *
 * @author Edward Raff
 */
public class SortedArrayList<T extends Comparable<T>> extends ArrayList<T> implements Serializable
{


	private static final long serialVersionUID = -8728381865616791954L;

	public SortedArrayList(final Collection<? extends T> c)
    {
        super(c);
        Collections.sort(this);
    }

    public SortedArrayList(final int initialCapacity)
    {
        super(initialCapacity);
    }

    public SortedArrayList()
    {
        super();
    }

    @Override
    public boolean add(final T e)
    {
        if(isEmpty())
        {
            return super.add(e);
        }
        else
        {
            int ind = Collections.binarySearch(this, e);
            if (ind < 0) {
              ind = -(ind + 1);//Now it is the point where it should be inserted
            }

            if (ind > size()) {
              super.add(e);
            } else {
              super.add(ind, e);
            }
            return true;
        }
    }

    @Override
    public void add(final int index, final T element)
    {
        this.add(element);
    }
    
    public T first()
    {
        if(isEmpty()) {
          return null;
        }
        return get(0);
    }
    
    public T last()
    {
        if(isEmpty()) {
          return null;
        }
        return get(size()-1);
    }

    @Override
    public boolean addAll(final Collection<? extends T> c)
    {
        if(c.isEmpty()) {
          return false;
        } else if(c.size() > this.size()*3/2)//heuristic when is it faster to just add them all and sort the whole thing?
        {
            final boolean did = super.addAll(c);
            if(did) {
              Collections.sort(this);
          }
            return did;
        }
        else
        {
            for(final T t : c) {
              this.add(t);
          }
            return true;
        }
    }

    @Override
    public boolean addAll(final int index, final Collection<? extends T> c)
    {
        return this.addAll(c);
    }
}
