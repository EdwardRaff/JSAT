
package jsat.linear;

import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * This class is used to create an implicit representation of the degree 2
 * polynomial of an input vector, with an implicit bias term added so that the
 * original vector values are present in the implicit vector. This means no
 * extra memory will be allocated, and all values accessed will be re-computed
 * as needed. This works with sparse vectors, and work bet with algorithms that
 * iterate over the nonzero values once.
 * <br><br>
 * Any change in the base vector will change the values in this vector. Because
 * changing one value in the base effects multiple values in this one, altering
 * this vector directly is not allowed.
 * <br><br>
 * If the base vector has {@code N} non zero values, then this vec will have
 * O(N<sup>2</sup>) non zero values. (N+2)(N+1)/2 non zero values to be exact.
 *
 * @author Edward Raff
 */
public class Poly2Vec extends Vec
{

	private static final long serialVersionUID = -5653680966558726340L;

	private Vec base;
    
    /**
     * This maps values pas the original coefficients (and bias term) shifted to
     * start from zero, to the appropriate value for the fist coefficient. 
     * <br>
     * This will be created lazily as needed. Call {@link #getReverseIndex() } 
     * to access this value
     */
    private int[] reverseIndex;

    public Poly2Vec(Vec base)
    {
        setBase(base);
    }
    
    /*
     * Some math needed for this class to make sense. Given an  input we want to poly 2 form plus a bias term. So for
     * (x + y + z) 
     * we want
     * (1 + x+ y + z + x^2 + x y + x z + y^2 + y z + z^2)
     * 
     * Then for an input of size N, the poly 2 version has length (N+2)(N+1)/2
     *
     * The bias term and maintaining the original is easy. So lets assume we 
     * only want to get the value for the x^2 term and after. IE: given a term x
     * and y, give me the index of the coeff that contains their product. Let x
     * start from 0 and let x^2 also start from zero, so we map from one space
     * to the other.
     * 
     * The exact index location, when x <= y, is then x N + y -  x (x+1) / 2
     * 
     */
    
    /**
     * Creates a new vector that implicitly represents the degree 2 polynomial
     * of the base vector.
     *
     * @param base the base vector
     */
    public void setBase(Vec base)
    {
        this.base = base;
    }
    
    private int[] getReverseIndex()
    {
        if(reverseIndex != null && reverseIndex.length == base.length())
            Arrays.fill(reverseIndex, 0);
        else
            reverseIndex = new int[base.length()];
        reverseIndex[0] = base.length();
        for(int i = 1; i < reverseIndex.length; i++)
            reverseIndex[i] = reverseIndex[i-1]+(base.length()-i);
        return reverseIndex;
    }
    
    @Override
    public int length()
    {
        return (base.length()+2)*(base.length()+1)/2;
    }

    @Override
    public int nnz()
    {
        return (base.nnz()+2)*(base.nnz()+1)/2;
    }
    
    @Override
    public double get(int index)
    {
        if(index == 0)
            return 1;
        else if (index <= base.length())
            return base.get(index-1);
        else if (index >= length())
            throw new IndexOutOfBoundsException("Vector is of length " + length() +", but index "+ index + " was requested");
        int x = Arrays.binarySearch(getReverseIndex(), index-base.length()-1);
        if(x < 0)
            x = -x -1;
        else 
            x++;
        double xVal = base.get(x);
        
        int y = (x*x+x)/2 + (index-base.length()-1) - base.length()*x;//the first term is safe b/c it will always be an even number before division
        return xVal*base.get(y);
    }

    @Override
    public void set(int index, double val)
    {
        throw new UnsupportedOperationException("Poly2Vec may not be altered");
    }

    @Override
    public boolean isSparse()
    {
        return base.isSparse();
    }

    @Override
    public Vec clone()
    {
        return new Poly2Vec(base.clone());
    }

    @Override
    public Iterator<IndexValue> getNonZeroIterator(int start)
    {
        //First case: empty base vector
        if (base.nnz() == 0)
            return new Iterator<IndexValue>()
            {
                boolean hasNext = true;

                @Override
                public boolean hasNext()
                {
                    return hasNext;
                }

                @Override
                public IndexValue next()
                {
                    if (!hasNext)
                        throw new NoSuchElementException("Iterator is empty");
                    hasNext = false;
                    return new IndexValue(0, 1.0);
                }

                @Override
                public void remove()
                {
                    throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
                }
            };
        //Else, general case
        final int startStage;
        final Iterator<IndexValue> startOuterIter, startInerIter;
        boolean stage1Good = true;//fail occurs when the last index (or more) in the base vector is zero
        if(start == 0)
        {
            startStage = 0;
            startInerIter = startOuterIter = null;
        }
        else if(start <= base.length() && (stage1Good=base.getNonZeroIterator(start-1).hasNext()))
        {
            startStage = 1;
            startOuterIter = base.getNonZeroIterator(start-1);
            startInerIter = null;
        }
        else if(start >= length())
        {
            startStage = 3;
            startInerIter = startOuterIter = null;
        }
        else//where do we start?
        {
            if (!stage1Good)
                start = base.length() + 1;
            Iterator<IndexValue> candidateOuterIter, candidateInerIter;
            start--;//lazy ness so we can update first thing in each iteration (we dont actually want to change the first value in the looping
            do
            {
                start++;
                int x = Arrays.binarySearch(getReverseIndex(), start - base.length() - 1);
                if (x < 0)
                    x = -x - 1;
                else
                    x++;
                int y = (x * x + x) / 2 + (start - base.length() - 1) - base.length() * x;//the first term is safe b/c it will always be an even number before division
                candidateOuterIter = base.getNonZeroIterator(x);
                /*
                 * If the x coefficeint is zero, we will jump to the next non 
                 * zero x. This means y must change as well, so we will check if
                 * that has happened by grabbing another iterator to get the 
                 * value. If this has happened, we know that y should be set to 
                 * x's value
                 */
                int nextXIndex = candidateOuterIter.hasNext() ? base.getNonZeroIterator(x).next().getIndex() : -1;
                if(candidateOuterIter.hasNext() && nextXIndex > x)//x is at a zero, so we need to inner iter to go back to the "begining"
                    candidateInerIter = base.getNonZeroIterator(nextXIndex);//next variable starts at val^2
                else
                    candidateInerIter = base.getNonZeroIterator(y);
            }
            while ( (!candidateOuterIter.hasNext() || !candidateInerIter.hasNext()) && start < length());
            if (candidateOuterIter.hasNext() && candidateInerIter.hasNext() && start < length())
            {
                startStage = 2;
                startOuterIter = candidateOuterIter;
                startInerIter = candidateInerIter;
            }
            else
                return new Iterator<IndexValue>()
                {
                    @Override
                    public boolean hasNext()
                    {
                        return false;
                    }

                    @Override
                    public IndexValue next()
                    {
                        throw new NoSuchElementException("Iterator is empty");
                    }

                    @Override
                    public void remove()
                    {
                        throw new UnsupportedOperationException("Not supported yet.");
                    }
                };
        }

        return new Iterator<IndexValue>() 
        {
            int stage = startStage;//0 is for bias, 1 is for stanrdard values, 2 is for combinations, 3 is for empty
            
            Iterator<IndexValue> outerIter = startOuterIter, inerIter = startInerIter;
            IndexValue curOuterVal = inerIter != null ? outerIter.next() : null;
            IndexValue toReturn = new IndexValue(0, 0);
            
            @Override
            public boolean hasNext()
            {
                if(stage < 3)
                    return true;
                return false;
            }

            @Override
            public IndexValue next()
            {
                if(stage == 0)
                {
                    stage++;
                    outerIter = base.getNonZeroIterator();//we know its non empty b/c of first case 
                    return new IndexValue(0, 1.0);
                }
                else if (stage == 1)//outerIter must always have a next item if stage = 1
                {
                    IndexValue iv = outerIter.next();
                    if (!outerIter.hasNext())
                    {
                        stage++;
                        outerIter = base.getNonZeroIterator();
                        curOuterVal = outerIter.next();
                        inerIter = base.getNonZeroIterator();
                    }
                    toReturn.setIndex(1+iv.getIndex());
                    toReturn.setValue(iv.getValue());
                    return toReturn;
                }
                else if(stage == 2)
                {
                    IndexValue innerVal = inerIter.next();
                    int x = curOuterVal.getIndex();
                    int y = innerVal.getIndex();
                    int N = base.length();
                    toReturn.setIndex(1+N+x*N+y-x*(x+1)/2);
                    toReturn.setValue(curOuterVal.getValue()*innerVal.getValue());
                    
                    if(!inerIter.hasNext())
                    {
                        if(!outerIter.hasNext())//we are out!
                        {
                            stage++;
                            outerIter = inerIter = null;
                        }
                        else//Still at least one more round!
                        {
                            curOuterVal = outerIter.next();
                            //new inner itter starts at idx^2
                            inerIter = base.getNonZeroIterator(curOuterVal.getIndex());
                        }
                    }
                    
                    return toReturn;
                }
                else //stage >= 3
                    throw new NoSuchElementException("Iterator is empty");
            }

            @Override
            public void remove()
            {
                throw new UnsupportedOperationException("Not supported yet."); 
            }
        };
    }
    
}
