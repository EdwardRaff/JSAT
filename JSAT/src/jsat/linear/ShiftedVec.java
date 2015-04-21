
package jsat.linear;

import static java.lang.Math.abs;
import static java.lang.Math.pow;
import java.util.Iterator;

/**
 * A wrapper for a vector that represents the vector added with a scalar value.
 * This allows for using and altering the value added by a 
 * constant factor quickly. Mutable operations will alter the underling vector, 
 * and all operations will automatically be adjusted on a per element basis as 
 * needed. The most significant performance difference is in representing a 
 * sparse input sifted by a constant factor. Note, that when using a sparse base
 * vector - the ShiftedVec will always return {@code true} for 
 * {@link #isSparse() }, even if shifting makes all values non-zero
 * <br>
 * 
 * @author Edward Raff
 */
public class ShiftedVec extends Vec
{

	private static final long serialVersionUID = -8318033099234181766L;
	private Vec base;
    private double shift;

    /**
     * Creates a new vector represented as <code><b>base</b>+shift</code>
     * @param base the base vector to represent
     * @param shift the scalar shift to add to the base vector
     */
    public ShiftedVec(Vec base, double shift)
    {
        this.base = base;
        this.shift = shift;
    }

    /**
     * @return the base vector used by this object
     */
    public Vec getBase()
    {
        return base;
    }

    /**
     * Directly alters the shift value used for this vector. The old sift is 
     * "forgotten" immediately. 
     * @param shift the new value to use for the additive scalar
     */
    public void setShift(double shift)
    {
        this.shift = shift;
    }

    /**
     * 
     * @return the additive scalar used to shift over the {@link #getBase() 
     * base} value
     */
    public double getShift()
    {
        return shift;
    }
    
    /**
     * Embeds the current shift scalar into the base vector, modifying it and 
     * then setting the new shit to zero. This makes the base vector have the 
     * value represented by this ShiftedVec 
     */
    public void embedShift()
    {
        base.mutableAdd(shift);
        shift = 0;
    }
    
    @Override
    public int length()
    {
        return base.length();
    }

    @Override
    public double get(int index)
    {
        return base.get(index)+shift;
    }

    @Override
    public void set(int index, double val)
    {
        base.set(index, val-shift);
    }

    @Override
    public void increment(int index, double val)
    {
        base.increment(index, val);
    }

    @Override
    public void mutableAdd(Vec b)
    {
        if(b instanceof ShiftedVec)
        {
            ShiftedVec other = (ShiftedVec) b;
            base.mutableAdd(other.base);
            shift += other.shift;
        }
        else
            base.mutableAdd(b);
    }

    @Override
    public void mutableAdd(double c)
    {
        shift += c;
    }

    @Override
    public void mutableAdd(double c, Vec b)
    {
        if(b instanceof ShiftedVec)
        {
            ShiftedVec other = (ShiftedVec) b;
            base.mutableAdd(c, other.base);
            shift += other.shift*c;
        }
        else
            base.mutableAdd(c, b); 
    }

    @Override
    public void mutableDivide(double c)
    {
        base.mutableDivide(c);
        shift /= c;
        if(Double.isNaN(shift))
            shift = 0;
    }

    @Override
    public void mutableMultiply(double c)
    {
        base.mutableMultiply(c);
        shift*=c;
    }

    @Override
    public void mutablePairwiseDivide(Vec b)
    {
        //this would require multiple different shifts, so we have to fold it back into the base vector
        base.mutableAdd(shift);
        shift = 0;
        base.mutablePairwiseDivide(b);
    }

    @Override
    public void mutablePairwiseMultiply(Vec b)
    {
        //this would require multiple different shifts, so we have to fold it back into the base vector
        base.mutableAdd(shift);
        shift = 0;
        base.mutablePairwiseMultiply(b);
    }
    
    
//    No real performance to gain by re-implementing matrix mul ops
//    @Override
//    public void multiply(double c, Matrix A, Vec b)
    
    @Override
    public double dot(Vec v)
    {
        if(v instanceof  ShiftedVec)
        {
            ShiftedVec other = (ShiftedVec) v;
            return this.base.dot(other.base) + other.base.sum()*this.shift + this.base.sum()*other.shift + this.length()*this.shift*other.shift;
        }
        return base.dot(v) + v.sum()*shift;
    }

    @Override
    public void zeroOut()
    {
        base.zeroOut();
        shift = 0;
    }

    @Override
    public double pNorm(double p)
    {
        if(!isSparse())
            return super.pNorm(p);
        //else sparse base, we can save some work
        //contributes of zero values
        double baseZeroContribs = pow(abs(shift), p)*(length()-base.nnz());
        //+ contribution of non zero values
        double baseNonZeroContribs = 0;
        for(IndexValue iv : base)
            baseNonZeroContribs += pow(abs(iv.getValue()+shift), p);
        return pow(baseNonZeroContribs+baseZeroContribs, 1/p);
    }

//    TODO: In the case of the y also being a ShiftedVec and sparse some significant performance could be saved.
//    public double pNormDist(double p, Vec y)
            
    
    @Override
    public double mean()
    {
        return base.mean()+ shift;
    }

    @Override
    public double variance()
    {
        return base.variance();
    }

    @Override
    public double standardDeviation()
    {
        return base.standardDeviation();
    }

    @Override
    public double kurtosis()
    {
        return base.kurtosis();
    }

    @Override
    public double max()
    {
        return base.max()+shift;
    }

    @Override
    public double min()
    {
        return base.min()+shift;
    }

    @Override
    public double median()
    {
        return base.median()+shift;
    }
    
    @Override
    public Iterator<IndexValue> getNonZeroIterator(final int start)
    {
        if(!isSparse())//dense case, just add the shift and use the base implemenaton since its going to do the exact same thing I would
            return super.getNonZeroIterator(start);
        final Iterator<IndexValue> baseIter = base.getNonZeroIterator(start);
        if(shift == 0)//easy case, just use the base's iterator
            return baseIter;
        
        //ugly case, sparse vec with shifted values iterating over non zeros (which should generally be all of them)
        final int lastIndx = length()-1;
        
        return new Iterator<IndexValue>()
        {
            IndexValue nextBaseVal;
            IndexValue nextVal;
            IndexValue toRet = null;
            
            //init
            {
                
                for(int effectiveStart = start; effectiveStart <= lastIndx; effectiveStart++)
                {
                    nextBaseVal = baseIter.hasNext() ? baseIter.next() : null;
                    if (nextBaseVal != null && nextBaseVal.getIndex() == effectiveStart)
                    {
                        if (nextBaseVal.getValue() + shift == 0)
                            continue;//no starting on zero!
                        else
                            nextVal = new IndexValue(effectiveStart, nextBaseVal.getValue() + shift);
                        nextBaseVal = baseIter.hasNext() ? baseIter.next() : null;
                    }
                    else//was zero + shift
                        nextVal = new IndexValue(effectiveStart, shift);
                    toRet = new IndexValue(effectiveStart, shift);
                    break;
                }
            }

            @Override
            public boolean hasNext()
            {
                return nextVal != null;
            }
            
            

            @Override
            public IndexValue next()
            {
                toRet.setIndex(nextVal.getIndex());
                toRet.setValue(nextVal.getValue());
                
                //loop to get next value b/c we may have to skip over zeros
                do
                {
                    nextVal.setIndex(nextVal.getIndex()+1);//pre-bump index 
                    //prep next value
                    if(nextVal.getIndex() == lastIndx+1)
                        nextVal = null;//done
                    else
                    {
                        if(nextBaseVal != null && nextBaseVal.getIndex() == nextVal.getIndex())//there is a base non-zero next
                        {
                            nextVal.setValue(nextBaseVal.getValue()+shift);
                            nextBaseVal = baseIter.hasNext() ? baseIter.next() : null;
                        }
                        else//a base non-zero in our imediate future
                        {
                            nextVal.setValue(shift);
                        }
                    }
                }
                while(nextVal != null && nextVal.getValue() == 0);
                
                return toRet;
            }

            @Override
            public void remove()
            {
                throw new UnsupportedOperationException("Not supported.");
            }
        };
    }

    @Override
    public boolean isSparse()
    {
        return base.isSparse();
    }

    @Override
    public ShiftedVec clone()
    {
        return new ShiftedVec(base.clone(), shift);
    }
    
}
