package jsat.utils;

import java.util.Objects;

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

    @Override
    public boolean equals(Object obj)
    {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        final Pair<?, ?> other = (Pair<?, ?>) obj;
        if (!Objects.equals(this.firstItem, other.firstItem))
            return false;
        if (!Objects.equals(this.secondItem, other.secondItem))
            return false;
        return true;
    }

    @Override
    public int hashCode()
    {
        int hash = 3;
        hash = 41 * hash + Objects.hashCode(this.firstItem);
        hash = 41 * hash + Objects.hashCode(this.secondItem);
        return hash;
    }
    
    
}
