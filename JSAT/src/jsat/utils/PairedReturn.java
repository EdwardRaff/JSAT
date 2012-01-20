
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

    /**
     * Returns the first object stored.
     * @return the first object stored
     */
    public T getFirstItem()
    {
        return firstItem;
    }

    /**
     * Returns the second object stored
     * @return the second object stored
     */
    public V getSecondItem()
    {
        return secondItem;
    }

    @Override
    public String toString()
    {
        return "(" + getFirstItem() + ", " + getSecondItem() + ")";
    }
}
