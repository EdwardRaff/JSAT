
package jsat.utils;

import java.util.*;

/**
 *
 * A Sorted set that has a maximum number of values it will hold.
 * 
 * @author Edward Raff
 */
public class BoundedSortedSet<V> extends TreeSet<V>
{

	private static final long serialVersionUID = -4774987058243433217L;
	private final int maxSize;

    public BoundedSortedSet(int max)
    {
        super();
        this.maxSize = max;
    }

    public BoundedSortedSet(int max, Comparator<? super  V> cmp)
    {
        super(cmp);
        this.maxSize = max;
    }
    
    @Override
    public boolean add(V e)
    {
        super.add(e);
        
        if(size() > maxSize)
            remove(last());
        return true;
    }

    @Override
    public boolean addAll(Collection<? extends V> clctn)
    {
        super.addAll(clctn);
        while (size() > maxSize)
            remove(last());
        return true;
    }

    /**
     * Returns the maximum size allowed for the bounded set
     * @return the maximum size allowed for the bounded set
     */
    public int getMaxSize()
    {
        return maxSize;
    }
    
}
