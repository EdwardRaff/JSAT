
package jsat.linear;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * SubVector takes an already existing vector and creates a new one that is a 
 * subset of and backed by the original one. Altering the sub vector will effect 
 * the original and vise versa. 
 * 
 * @author Edward Raff
 */
public class SubVector extends Vec
{

	private static final long serialVersionUID = -873882618035700676L;
	private int startPosition;
    private int length;
    private Vec vec;

    /**
     * Creates a new sub vector of the input vector
     * 
     * @param startPosition the starting index (inclusive) or the original 
     * vector
     * @param length the length of the new sub vector
     * @param vec the original vector to back this sub vector. 
     */
    public SubVector(int startPosition, int length, Vec vec)
    {
        if(startPosition < 0 || startPosition >= vec.length())
            throw new IndexOutOfBoundsException("Start position out of bounds for input vector");
        else if(length+startPosition > vec.length())
            throw new IndexOutOfBoundsException("Length too long for start position for the given vector");
        
        this.startPosition = startPosition;
        this.length = length;
        this.vec = vec;
    }

    @Override
    public int length()
    {
        return length;
    }

    @Override
    public double get(int index)
    {
        if(index >= length)
            throw new IndexOutOfBoundsException("Index of " + index + " can not be accessed for length of " + length);
        return vec.get(startPosition+index);
    }

    @Override
    public void set(int index, double val)
    {
        if(index >= length)
            throw new IndexOutOfBoundsException("Index of " + index + " can not be accessed for length of " + length);
        vec.set(startPosition+index, val);
    }

    @Override
    public boolean isSparse()
    {
        return vec.isSparse();
    }

    @Override
    public Iterator<IndexValue> getNonZeroIterator(int start)
    {
        final Iterator<IndexValue> origIter = vec.getNonZeroIterator(startPosition+start);

        Iterator<IndexValue> newIter = new Iterator<IndexValue>()
        {
            IndexValue nextVal = origIter.hasNext() ? origIter.next() : new IndexValue(Integer.MAX_VALUE, Double.NaN);
            IndexValue curVal = new IndexValue(-1, Double.NaN);
            @Override
            public boolean hasNext()
            {
                return nextVal.getIndex() < length+startPosition;
            }

            @Override
            public IndexValue next()
            {
                if(!hasNext())
                    throw new NoSuchElementException();
                curVal.setIndex(nextVal.getIndex()-startPosition);
                curVal.setValue(nextVal.getValue());
                if(origIter.hasNext())
                    nextVal = origIter.next();
                else
                    nextVal.setIndex(Integer.MAX_VALUE);
                
                return curVal;
            }

            @Override
            public void remove()
            {
                throw new UnsupportedOperationException("Not supported yet.");
            }
        };
        
        return newIter;
    }

    @Override
    public Vec clone()
    {
        if(vec.isSparse())
            return new SparseVector(this);
        else
            return new DenseVector(this);
    }

}
