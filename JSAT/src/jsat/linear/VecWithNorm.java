package jsat.linear;

import java.util.Iterator;

/**
 * A wrapper for a vector that allows for transparent tracking of the 2-norm of 
 * the base vector. This class is meant primarily for use when most updates are 
 * done by sparse vectors accumulated to a single dense vector. If there are 
 * only O(s) non zero values, updating the norm can be done in O(s) time. If 
 * most updates will be done by dense vectors, this wrapper may not give any
 * performance improvements. <br>
 * The norm is obtained by calling {@link #pNorm(double) }. The original vector 
 * can be obtained by calling {@link #getBase() }. The exact values returned for
 * the norm may differ slightly due to numerical issues. 
 * 
 * @author Edward Raff
 */
public class VecWithNorm extends Vec
{

	private static final long serialVersionUID = 3888178071694466561L;
	final private Vec base;
    private double normSqrd;

    /**
     * Creates a wrapper around the base vector that will update the norm of the
     * vector 
     * @param base the vector to use as the base value
     * @param norm the initial value of the norm
     */
    public VecWithNorm(final Vec base, final double norm)
    {
        this.base = base;
        this.normSqrd = norm*norm;
    }

    /**
     * Creates a wrapper around the base vector that will update the norm of the
     * vector 
     * @param base the vector to use as the base value
     */
    public VecWithNorm(final Vec base)
    {
        this(base, base.pNorm(2));
    }

    /**
     * Return the base vector that is having its norm tracked
     * @return the base vector that is having its norm tracked
     */
    public Vec getBase()
    {
        return base;
    }

    @Override
    public double pNorm(final double p)
    {
        if(p == 2) {
          return Math.sqrt(normSqrd);
        }
        return base.pNorm(p);
    }

    @Override
    public int length()
    {
        return base.length();
    }

    @Override
    public double get(final int index)
    {
        return base.get(index);
    }

    @Override
    public void set(final int index, final double val)
    {
        final double old = base.get(index);
        
        normSqrd += -(old*old)+(val*val);
        base.set(index, val);
    }

    @Override
    public boolean isSparse()
    {
        return base.isSparse();
    }

    @Override
    public VecWithNorm clone()
    {
        return new VecWithNorm(this.base.clone(), Math.sqrt(normSqrd));
    }

    @Override
    public void mutableAdd(final double c)
    {
        //TODO this can be improved for scenarios where the base vector is sparse, but that should be uncommon 
        for(int i = 0; i < base.length(); i++)
        {
            final double old = base.get(i);
            final double toAdd = c;
            normSqrd += toAdd*(toAdd+2*old);
        }
        base.mutableAdd(c);
    }

    @Override
    public void mutableAdd(final double c, final Vec b)
    {
        for(final IndexValue iv : b)
        {
            final double old = base.get(iv.getIndex());
            final double toAdd = c*iv.getValue();
            normSqrd += toAdd*(toAdd+2*old);
        }
        base.mutableAdd(c, b);
    }

    @Override
    public void mutablePairwiseMultiply(final Vec b)
    {
        //if b is sparse or dense its going to need updates to every value.
        //migth as well jsut refresh
        base.mutablePairwiseMultiply(b);
        normSqrd = Math.pow(base.pNorm(2), 2);
    }

    @Override
    public void mutableMultiply(final double c)
    {
        normSqrd *= c*c;
        base.mutableMultiply(c);
    }

    @Override
    public void mutablePairwiseDivide(final Vec b)
    {
        //if b is sparse or dense its going to need updates to every value.
        //migth as well just refresh
        base.mutablePairwiseDivide(b);
        normSqrd = Math.pow(base.pNorm(2), 2);
    }

    @Override
    public void mutableDivide(final double c)
    {
        normSqrd /= c*c;
    }
    
    @Override
    public void zeroOut()
    {
        normSqrd = 0;
        base.zeroOut();
    }

    @Override
    public int nnz()
    {
        return base.nnz();
    }

    @Override
    public Iterator<IndexValue> getNonZeroIterator()
    {
        return base.getNonZeroIterator(); 
    }

    @Override
    public Iterator<IndexValue> getNonZeroIterator(final int start)
    {
        return base.getNonZeroIterator(start);
    }
}
