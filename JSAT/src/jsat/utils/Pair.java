package jsat.utils;

/**
 * A simple object to hold a pair of values
 * @author Edward Raff
 */
public class Pair<X, Y>
{
    private X firstItem;
    private Y secondItem;

    public Pair(final X firstItem, final Y secondItem)
    {
        setFirstItem(firstItem);
        setSecondItem(secondItem);
    }

    public void setFirstItem(final X firstItem)
    {
        this.firstItem = firstItem;
    }

    public X getFirstItem()
    {
        return firstItem;
    }

    public void setSecondItem(final Y secondItem)
    {
        this.secondItem = secondItem;
    }

    public Y getSecondItem()
    {
        return secondItem;
    }
}
