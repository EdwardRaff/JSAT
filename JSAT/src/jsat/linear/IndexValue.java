
package jsat.linear;

/**
 * The value at a specified index for one dimension. This is a tool mean for use with sparce data structures. 
 * The values should not be backed by any list, and changes to the IndexValue should not alter any data 
 * structures. This class is mean to be returned by an iterator, and the iterator may reuse the same
 * IndexValue object for efficency. 
 * 
 * @author Edward Raff
 */
public class IndexValue
{
    private int index;
    private double value;

    /**
     * Creates a new IndexValue
     * @param index the index for the given value
     * @param value the value at the specified index
     */
    public IndexValue(int index, double value)
    {
        this.index = index;
        this.value = value;
    }

    /**
     * Sets the index associated with the value. 
     * @param index the new index
     */
    public void setIndex(int index)
    {
        this.index = index;
    }

    /**
     * Sets the value associated with the index
     * @param value the new value
     */
    public void setValue(double value)
    {
        this.value = value;
    }

    /**
     * Returns the index of the stored value
     * @return the index of the stored value
     */
    public int getIndex()
    {
        return index;
    }

    /**
     * Returns the value of the stored index
     * @return the value of the stored index
     */
    public double getValue()
    {
        return value;
    }
    
}
