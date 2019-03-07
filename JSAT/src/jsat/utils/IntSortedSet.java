
package jsat.utils;

import java.io.Serializable;
import java.util.*;

/**
 * A utility class for efficiently storing a set of integers. In order to do 
 * this, the integers are stored in their natural order. 
 * 
 * @author Edward Raff
 */
public class IntSortedSet extends AbstractSet<Integer> implements Serializable, SortedSet<Integer>
{

    private static final long serialVersionUID = -2675363824037596497L;

    private static final int defaultSize = 8;

    private int[] store;
    private int size;
    
    public IntSortedSet(int initialSize)
    {
        store = new int[initialSize];
        size = 0;
    }
    
    /**
     * Creates a new set of integers from the given set 
     * @param set the set of integers to create a copy of
     */
    public IntSortedSet(Set<Integer> set)
    {
        this(set.size());
        batch_insert(set, false);
    }
    
    
    /**
     * Creates a new set of integers from the given set
     *
     * @param set the set of integers to create a copy of
     * @param parallel {@code true} if sorting should be done in parallel, or
     * {@code false} for serial sorting.
     */
    public IntSortedSet(Collection<Integer> set, boolean parallel)
    {
        this(set.size());
        batch_insert(set, parallel);
    }
    
    /**
     * Creates a set of integers from the given collection
     * @param collection a collection of integers to create a set from
     */
    public IntSortedSet(Collection<Integer> collection)
    {
        this(collection.size());
        batch_insert(collection, false);
    }
    

    /**
     * more efficient insertion of many items by placing them all into the
     * backing store, and then doing one large sort.
     *
     * @param set
     * @param parallel
     */
    private void batch_insert(Collection<Integer> set, boolean parallel)
    {
        for(int i : set)
            store[size++] = i;
        if(parallel)
            Arrays.parallelSort(store, 0, size);
        else
            Arrays.sort(store, 0, size);
    }
    
    /**
     * Creates a set of integers from the given list of integers. 
     * @param ints a list of integers to create a set from
     * @return a set of integers of all the unique integers in the given list
     */
    public static IntSortedSet from(int... ints)
    {
        return new IntSortedSet(IntList.view(ints, ints.length));
    }
    
    /**
     * This method provides direct access to the integer array used to store all
     * ints in this object. Altering this will alter the set, and may cause
     * incorrect behavior if the sorted order is violated.
     *
     * @return the sorted int array used to back this current object. 
     */
    public int[] getBackingStore()
    {
        return store;
    }
    
    public IntSortedSet()
    {
        this(defaultSize);
    }

    @Override
    public boolean add(Integer e)
    {
        if(e == null)
            return false;
        
        //Increase store size if needed
        //checking here means we will actually expand 1 value early, but avoids a double check later
        //so small cost, I'll take it to simplify code
        if(size >= store.length)
            store = Arrays.copyOf(store, Math.max(1, store.length)*2);
        
        //fast path for when we are filling a set, and iterating from smallest to largest index. 
        //avoids needlessly expensive binary search for common insertion pattern
        if(size > 0 && store[size-1] < e)
        {
            store[size++] = e;
            return true;
        }
        //else, do a search to find where we should be placing this value
        int insertionPoint = Arrays.binarySearch(store, 0, size, e);
        if(insertionPoint >= 0 )
            return false;//Already in the set
        //Fix up to where we would like to place it
        insertionPoint = (-(insertionPoint) - 1);
        
        for(int i = size; i > insertionPoint; i--)
            store[i] = store[i-1];
        store[insertionPoint] = e;
        size++;
        
        return true;
    }

    public boolean contains(int o)
    {
        int insertionPoint = Arrays.binarySearch(store, 0, size, o);
        if(insertionPoint >= 0 )
            return true;//Already in the set
        else
            return false;
    }
    
    @Override
    public boolean contains(Object o)
    {
        if(o != null && o instanceof Integer)
            return contains(((Integer)o).intValue());
        else
            return false;
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

    @Override
    public Comparator<? super Integer> comparator()
    {
        return new Comparator<Integer>()
        {
            @Override
            public int compare(Integer o1, Integer o2)
            {
                return o1.compareTo(o2);
            }
        };
    }

    @Override
    public SortedSet<Integer> subSet(Integer fromElement, Integer toElement)
    {
        if(fromElement > toElement)
            throw new IllegalArgumentException("fromKey was > toKey");
        return new IntSortedSubSet(fromElement, toElement-1, this);
    }

    @Override
    public SortedSet<Integer> headSet(Integer toElement)
    {
        return new IntSortedSubSet(Integer.MIN_VALUE, toElement-1, this);
    }

    @Override
    public SortedSet<Integer> tailSet(Integer fromElement)
    {
        return new IntSortedSubSet(fromElement, Integer.MAX_VALUE-1, this);
    }

    @Override
    public Integer first()
    {
        return store[0];
    }

    @Override
    public Integer last()
    {
        return store[size-1];
    }
    
    private class IntSortedSubSet extends AbstractSet<Integer> implements Serializable, SortedSet<Integer>
    {
        int minValidValue;
        int maxValidValue;
        IntSortedSet parent;
        
        public IntSortedSubSet(int minValidValue, int maxValidValue, IntSortedSet parent)
        {
            this.minValidValue = minValidValue;
            this.maxValidValue = maxValidValue;
            this.parent = parent;
        }
        
        @Override
        public boolean add(Integer e)
        {
            if(e == null)
                return false;
            if(e < minValidValue || e > maxValidValue)
                throw new IllegalArgumentException("You can not add to a sub-set view outside of the constructed range");
            return parent.add(e);
        }

        public boolean contains(int o)
        {
            if(o < minValidValue || o > maxValidValue)
                return false;
            return parent.contains(o);
        }

        @Override
        public boolean contains(Object o)
        {
            if(o != null && o instanceof Integer)
                return contains(((Integer)o).intValue());
            else
                return false;
        }

        @Override
        public Iterator<Integer> iterator()
        {
            int start = Arrays.binarySearch(store, 0, size, minValidValue);
            if (start < 0)
                start = (-(start) - 1);
            final int start_pos = start;
            final Iterator<Integer> iterator = new Iterator<Integer>() 
            {
                int index = start_pos;
                boolean canRemove = false;

                @Override
                public boolean hasNext()
                {
                    return index < size && store[index] <= maxValidValue;
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
            int start = Arrays.binarySearch(store, 0, size, minValidValue);
            if (start < 0)
                start = (-(start) - 1);
            int end = Arrays.binarySearch(store, 0, size, maxValidValue);
            if (end < 0)
                end = (-(end) - 1);
            if(end < size && store[end] == maxValidValue)//should only happen once b/c we are a set
                end++;
            return end-start;
        }

        @Override
        public Comparator<? super Integer> comparator()
        {
            return parent.comparator();
        }

        @Override
        public SortedSet<Integer> subSet(Integer fromElement, Integer toElement)
        {
            if(fromElement > toElement)
                throw new IllegalArgumentException("fromKey was > toKey");
            return parent.subSet(Math.max(minValidValue, fromElement), Math.min(maxValidValue+1, toElement));
        }

        @Override
        public SortedSet<Integer> headSet(Integer toElement)
        {
            return new IntSortedSubSet(Math.max(Integer.MIN_VALUE, minValidValue), Math.min(toElement-1, maxValidValue), parent);
//            return parent.headSet(Math.min(toElement, maxValidValue+1));
        }

        @Override
        public SortedSet<Integer> tailSet(Integer fromElement)
        {
            return new IntSortedSubSet(Math.max(fromElement, minValidValue), Math.min(Integer.MAX_VALUE-1, maxValidValue), parent);
//            return parent.tailSet(Math.max(fromElement, minValidValue));
        }

        @Override
        public Integer first()
        {
            int start = Arrays.binarySearch(store, 0, size, minValidValue);
            if (start < 0)
                start = (-(start) - 1);
            return store[start];
        }

        @Override
        public Integer last()
        {
            int pos = Arrays.binarySearch(store, 0, size, maxValidValue);
            if (pos < 0)
                pos = (-(pos) - 1);
            //can end up at a pos out of valid range, so back up if needed
            while(pos >= 0 && store[pos] > maxValidValue)
                pos--;
            return store[pos];
        }
        
    }
    
}
