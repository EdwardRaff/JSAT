package jsat.utils;

import java.util.Iterator;

/**
 * Convenience object for being able to use the for each loop on an iterator. 
 * 
 * @author Edward Raff
 */
public final class IterableIterator<T> implements Iterable<T>
{
    private final Iterator<T> iterator;

    public IterableIterator(Iterator<T> iterator)
    {
        this.iterator = iterator;
    }

    @Override
    public Iterator<T> iterator()
    {
        return iterator;
    }
    
}
