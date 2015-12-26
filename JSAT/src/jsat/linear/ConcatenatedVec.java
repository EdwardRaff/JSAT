package jsat.linear;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * ConcatenatedVec provides a light wrapper around a list of vectors to provide 
 * a view of one single vector that's length is the sum of the lengths of the 
 * inputs. 
 * 
 * @author Edward Raff
 */
public class ConcatenatedVec extends Vec
{

    private static final long serialVersionUID = -1412322616974470550L;
    private Vec[] vecs;
    private int[] lengthSums;
    private int totalLength;

    /**
     * Creates a new Vector that is the concatenation of the given vectors in 
     * the given order. The vector created is backed by the ones provided, and 
     * any mutation to one is visible in the others. 
     * 
     * @param vecs the list of vectors to concatenate
     */
    public ConcatenatedVec(List<Vec> vecs)
    {
        this.vecs = new Vec[vecs.size()];
        lengthSums = new int[vecs.size()];
        totalLength = 0;
        for(int i = 0; i < vecs.size(); i++)
        {
            lengthSums[i] = totalLength;
            this.vecs[i] = vecs.get(i);
            totalLength += vecs.get(i).length();
        }
    }
    
    /**
     * Creates a new Vector that is the concatenation of the given vectors in 
     * the given order. The vector created is backed by the ones provided, and 
     * any mutation to one is visible in the others. 
     * 
     * @param vecs the array of vectors to concatenate
     */
    public ConcatenatedVec(Vec... vecs)
    {
        this(Arrays.asList(vecs));
    }

    @Override
    public int length()
    {
        return totalLength;
    }

    @Override
    public double get(int index)
    {
        int baseIndex = getBaseIndex(index);
        return vecs[baseIndex].get(index-lengthSums[baseIndex]);
    }

    @Override
    public void set(int index, double val)
    {
        int baseIndex = getBaseIndex(index);
        vecs[baseIndex].set(index-lengthSums[baseIndex], val);
    }
    
    //The following are implemented only for performance reasons
    
    @Override
    public void increment(int index, double val)
    {
        int baseIndex = getBaseIndex(index);
        vecs[baseIndex].increment(index-lengthSums[baseIndex], val);
    }

    @Override
    public int nnz()
    {
        int nnz = 0;
        for(Vec v : vecs)
            nnz += v.nnz();
        return nnz;
    }

    @Override
    public void mutableAdd(double c, Vec b)
    {
        for(int i = 0; i < vecs.length; i++)
        {
            vecs[i].mutableAdd(c, new SubVector(lengthSums[i], vecs[i].length(), b));
        }
    }

    @Override
    public Iterator<IndexValue> getNonZeroIterator(final int start)
    {
        return new Iterator<IndexValue>()
        {
            int baseIndex = -1;
            IndexValue valToSend = new IndexValue(0, 0);
            Iterator<IndexValue> curIter = null;
            IndexValue nextValue = null;
            
            
            @Override
            public boolean hasNext()
            {
                if(baseIndex == -1)//initialize everything
                {
                    baseIndex = getBaseIndex(start);
                    int curIndexConsidering = start;
                    //Keep moving till we
                    while(baseIndex < vecs.length && !vecs[baseIndex].getNonZeroIterator(curIndexConsidering-lengthSums[baseIndex]).hasNext())
                    {
                        baseIndex++;
                        if(baseIndex < vecs.length)
                            curIndexConsidering = lengthSums[baseIndex];
                        
                    }
                    if(baseIndex >= vecs.length)
                        return false;//All zeros beyond this point
                    curIter = vecs[baseIndex].getNonZeroIterator(curIndexConsidering-lengthSums[baseIndex]);
                    nextValue = curIter.next();
                    return true;
                }
                else
                    return nextValue != null;
            }

            @Override
            public IndexValue next()
            {
                if(nextValue == null)
                    throw new NoSuchElementException();
                valToSend.setIndex(nextValue.getIndex()+lengthSums[baseIndex]);
                valToSend.setValue(nextValue.getValue());
                
                if(curIter.hasNext())
                    nextValue = curIter.next();
                else
                {
                    baseIndex++;
                    while(baseIndex < vecs.length && !(curIter = vecs[baseIndex].getNonZeroIterator()).hasNext())//keep moving till with find a non empty vec
                        baseIndex++;
                    if(baseIndex >= vecs.length)//we have run out
                    {
                        nextValue = null;
                        curIter = null;
                    }
                    else
                    {
                        nextValue = curIter.next();
                    }
                }
                
                return valToSend;
            }

            @Override
            public void remove()
            {
                throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
            }
        };
    }
    
    @Override
    public boolean isSparse()
    {
        for(Vec v : vecs)
            if(v.isSparse())
                return true;
        return false;
    }

    @Override
    public ConcatenatedVec clone()
    {
        Vec[] newVecs = new Vec[vecs.length];
        for(int i = 0; i < vecs.length; i++)
            newVecs[i] = vecs[i].clone();
        return new ConcatenatedVec(Arrays.asList(newVecs));
    }

    private int getBaseIndex(int index)
    {
        int basIndex = Arrays.binarySearch(lengthSums, index);
        if(basIndex < 0)
            basIndex = (-(basIndex)-2);//-1 extra b/c we want to be on the lesser side
        return basIndex;
    }
    
}
