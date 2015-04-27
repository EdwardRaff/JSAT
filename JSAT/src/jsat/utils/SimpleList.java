package jsat.utils;

import java.io.Serializable;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.ConcurrentModificationException;

/**
 * An alternative implementation of an {@link ArrayList}. The behavior is the 
 * same and performance is almost exactly the same. The major difference is 
 * SimpleList does not perform the same checks on modification as ArrayList, and
 * will never throw a {@link ConcurrentModificationException}. This means 
 * SimpleList can have unexpected behavior when asked in a multi threaded 
 * environment, but can also be used by multiple threads where an ArrayList 
 * could not be. 
 * 
 * @author Edward Raff
 */
public class SimpleList<E> extends AbstractList<E> implements Serializable
{

	private static final long serialVersionUID = -1641584937585415217L;
	private Object[] source;
    private int size;
    
    /**
     * Creates a new SimpleList 
     * @param initialCapacity the initial storage size of the list
     */
    public SimpleList(int initialCapacity)
    {
        source = new Object[initialCapacity];
        size = 0;
    }
    
    /**
     * Creates a new SimpleList
     */
    public SimpleList()
    {
        this(50);
    }
    
    /**
     * Creates a new SimpleList 
     * @param c the collection of elements to place into the list. 
     */
    public SimpleList(Collection<? extends E> c)
    {
        this(c.size());
        this.addAll(c);
    }

    @Override
    public E get(int index)
    {
        if(index >= size())
            throw new IndexOutOfBoundsException();
        return (E) source[index];
    }

    @Override
    public int size()
    {
        return size;
    }

    @Override
    public void add(int index, E element)
    {
        if(index > size())
            throw new IndexOutOfBoundsException();
        if(size == source.length)
            source = Arrays.copyOf(source, size*2);
        
        if(index == size)
            source[size++] = element;
        else
        {
            System.arraycopy(source, index, source, index+1, size-index);
            source[index] = element;
            size++;
        }
    }

    @Override
    public E remove(int index)
    {
        if(index >= size())
            throw new IndexOutOfBoundsException();
        E removed = (E) source[index];
        System.arraycopy(source, index+1, source, index, size-index-1);
        size--;
        return removed;
    }

    @Override
    public E set(int index, E element)
    {
        if(index >= size())
            throw new IndexOutOfBoundsException();
        E prev = (E) source[index];
        source[index] = element;
        return prev;
    }
}
