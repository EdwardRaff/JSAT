
package jsat.utils;

import java.util.Collection;
import java.util.TreeSet;

/**
 *
 * A Sorted set that has a maximum number of values it will hold.
 * 
 * @author Edward Raff
 */
public class BoundedSortedSet<V> extends TreeSet<V>
{
    int max;

    public BoundedSortedSet(int max)
    {
        super();
        this.max = max;
    }
    
    

    @Override
    public boolean add(V e)
    {
        super.add(e);
        
        if(size() > max)
            remove(last());
        return true;
    }

    @Override
    public boolean addAll(Collection<? extends V> clctn)
    {
        super.addAll(clctn);
        while (size() > max)
            remove(last());
        return true;
    }

    
    
}
