
package jsat.utils;

/**
 *
 * Utility class that allows the returning of 2 different objects as one. 
 * 
 * @author Edwartd Raff
 */
public class PairedReturn<T, V>
{
    private final T firstItem;
    private final V secondItem;

    public PairedReturn(T t, V v)
    {
        this.firstItem = t;
        this.secondItem = v;
    }

    public T getFirstItem()
    {
        return firstItem;
    }

    public V getSecondItem()
    {
        return secondItem;
    }
}
