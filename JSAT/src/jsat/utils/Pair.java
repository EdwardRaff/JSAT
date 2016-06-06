package jsat.utils;

/**
 * A simple object to hold a pair of values
 * @author Edward Raff
 */
public class Pair<X, Y>
{
    private X firstItem;
    private Y secondItem;

    public Pair(X firstItem, Y secondItem)
    {
        setFirstItem(firstItem);
        setSecondItem(secondItem);
    }

    public void setFirstItem(X firstItem)
    {
        this.firstItem = firstItem;
    }

    public X getFirstItem()
    {
        return firstItem;
    }

    public void setSecondItem(Y secondItem)
    {
        this.secondItem = secondItem;
    }

    public Y getSecondItem()
    {
        return secondItem;
    }

    @Override
    public String toString()
    {
        return "(" + firstItem + ", " + secondItem + ")";
    }
}
