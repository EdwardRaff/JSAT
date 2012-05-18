
package jsat.utils;

import java.util.*;

/**
 * A utility class for efficiently storing a set of integers. In order to do 
 * this, the integers are stored in their natural order. 
 * 
 * @author Edward Raff
 */
public class IntSet extends AbstractSet<Integer>
{
    private static final int defaultSize = 8;
     
    private int[] store;
    private int size;
    
    public IntSet(int initialSize)
    {
        store = new int[initialSize];
        size = 0;
    }
    
    public IntSet(SortedSet<Integer> sortedSet)
    {
        this();
        for(Integer integer : sortedSet)
            this.add(integer);
    }
    
    public IntSet(Collection<Integer> collection)
    {
        this();
        for(Integer integer : collection)
            this.add(integer);
    }
    
    public IntSet()
    {
        this(defaultSize);
    }

    @Override
    public boolean add(Integer e)
    {
        if(e == null)
            return false;
        int insertionPoint = Arrays.binarySearch(store, 0, size, e);
        if(insertionPoint >= 0 )
            return false;//Already in the set
        //Fix up to where we would like to place it
        insertionPoint = (-(insertionPoint) - 1);
        
        //Increase store size if needed
        if(size >= store.length)
            store = Arrays.copyOf(store, store.length*2);
        
        for(int i = size; i > insertionPoint; i--)
            store[i] = store[i-1];
        store[insertionPoint] = e;
        size++;
        
        return true;
    }
    
    

    @Override
    public Iterator<Integer> iterator()
    {
        final Iterator<Integer> iterator = new Iterator<Integer>() 
        {
            int index = 0;
            boolean canRemove = false;

            @Override
            public boolean hasNext()
            {
                return index < size;
            }

            @Override
            public Integer next()
            {
                if(!hasNext())
                    throw new NoSuchElementException("The whole set has been iterated");
                canRemove = true;
                return store[index++];
            }

            @Override
            public void remove()
            {
                if(!canRemove)
                    throw new IllegalStateException("Can not remove, remove can only occur after a successful call to next");
                
                for(int i = index; i < size; i++ )
                    store[i-1] = store[i];
                
                index--;
                size--;
                canRemove = false;
            }
        };
        return iterator;
    }

    @Override
    public int size()
    {
        return size;
    }
    
}
