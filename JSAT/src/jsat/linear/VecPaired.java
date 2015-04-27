
package jsat.linear;

import java.util.Comparator;
import java.util.Iterator;

/**
 * This data structure allows to wrap a Vector so that it is 
 * associated with some object time. Note, that operations 
 * that return a vector will not be a Paired Vector, as there 
 * is no reason to associate a different vector with this 
 * vector's pair. 
 * 
 * @author Edward Raff
 */
public class VecPaired<V extends Vec, P> extends Vec
{

	private static final long serialVersionUID = 8039272826439917423L;
	private V vector;
    private P pair;

    public VecPaired(V v, P p)
    {
        this.vector = v;
        this.pair = p;
    }

    public P getPair()
    {
        return pair;
    }

    public void setPair(P pair)
    {
        this.pair = pair;
    }
    
    public V getVector()
    {
        return vector;
    }

    public void setVector(V vector)
    {
        this.vector = vector;
    }
    
    @Override
    public int length()
    {
        return vector.length();
    }

    @Override
    public int nnz()
    {
        return vector.nnz();
    }
    
    @Override
    public double get(int index)
    {
        return vector.get(index);
    }

    @Override
    public void set(int index, double val)
    {
        vector.set(index, val);
    }

    @Override
    public Vec add(double c)
    {
        return vector.add(c);
    }

    @Override
    public Vec add(Vec b)
    {
        b = extractTrueVec(b);
        return vector.add(b);
    }

    @Override
    public Vec subtract(Vec b)
    {
        b = extractTrueVec(b);
        return vector.subtract(b);
    }

    @Override
    public Vec pairwiseMultiply(Vec b)
    {
        b = extractTrueVec(b);
        return vector.pairwiseMultiply(b);
    }

    @Override
    public Vec multiply(double c)
    {
        return vector.multiply(c);
    }

    @Override
    public void multiply(double c, Matrix A, Vec b)
    {
        vector.multiply(c, A, b);
    }

    @Override
    public Vec pairwiseDivide(Vec b)
    {
        b = extractTrueVec(b);
        return vector.pairwiseDivide(b);
    }

    @Override
    public Vec divide(double c)
    {
        return vector.divide(c);
    }

    @Override
    public void mutableAdd(double c)
    {
        vector.mutableAdd(c);
    }

    @Override
    public void mutableAdd(Vec b)
    {
        b = extractTrueVec(b);
        vector.mutableAdd(b);
    }

    @Override
    public void mutableSubtract(Vec b)
    {
        b = extractTrueVec(b);
        vector.mutableSubtract(b);
    }

    @Override
    public void mutablePairwiseMultiply(Vec b)
    {
        b = extractTrueVec(b);
        vector.mutablePairwiseDivide(b);
    }

    @Override
    public void mutableMultiply(double c)
    {
        vector.mutableMultiply(c);
    }

    @Override
    public void mutablePairwiseDivide(Vec b)
    {
        b = extractTrueVec(b);
        vector.mutablePairwiseDivide(b);
    }

    @Override
    public void mutableDivide(double c)
    {
        vector.mutableDivide(c);
    }

    @Override
    public Vec sortedCopy()
    {
        return vector.sortedCopy();
    }

    @Override
    public double min()
    {
        return vector.min();
    }

    @Override
    public double max()
    {
        return vector.max();
    }

    @Override
    public double sum()
    {
        return vector.sum();
    }

    @Override
    public double mean()
    {
        return vector.mean();
    }

    @Override
    public double standardDeviation()
    {
        return vector.standardDeviation();
    }

    @Override
    public double variance()
    {
        return vector.variance();
    }

    @Override
    public double median()
    {
        return vector.median();
    }

    @Override
    public double skewness()
    {
        return vector.skewness();
    }

    @Override
    public double kurtosis()
    {
        return vector.kurtosis();
    }

    @Override
    public Vec clone()
    {
        return new VecPaired(vector.clone(), pair);
    }

    @Override
    public Vec normalized()
    {
        return vector.normalized();
    }

    @Override
    public void normalize()
    {
        vector.normalize();
    }

    @Override
    public double pNormDist(double p, Vec y)
    {
        y = extractTrueVec(y);
        return vector.pNormDist(p, y);
    }

    @Override
    public double pNorm(double p)
    {
        return vector.pNorm(p);
    }

    @Override
    public double dot(Vec v)
    {
        v = extractTrueVec(v);
        return this.vector.dot(v);
    }

    @Override
    public String toString()
    {
        return vector.toString();
    }

    @Override
    public boolean equals(Object obj)
    {
        return vector.equals(obj);
    }

    @Override
    public boolean equals(Object obj, double range)
    {
        return vector.equals(obj, range);
    }

    @Override
    public double[] arrayCopy()
    {
        return vector.arrayCopy();
    }

    @Override
    public void mutableAdd(double c, Vec b)
    {
        b = extractTrueVec(b);
       
        this.vector.mutableAdd(c, b);
    }

    @Override
    public Iterator<IndexValue> getNonZeroIterator()
    {
        if(extractTrueVec(vector) instanceof SparseVector)
            return extractTrueVec(vector).getNonZeroIterator();
        return super.getNonZeroIterator();
    }
    
    /**
     * This method is used assuming multiple VecPaired are used together. The 
     * implementation of the vector may have logic to handle the case that 
     * the other vector is of the same type. This will go through every layer 
     * of VecPaired to return the final base vector. 
     * 
     * @param b a Vec, that may or may not be an instance of {@link VecPaired}
     * @return the final Vec backing b, which may be b itself. 
     */
    public static Vec extractTrueVec(Vec b)
    {
        while(b instanceof VecPaired)
            b = ((VecPaired) b).getVector();
        return b;
    }
    
    public static <V extends Vec, P extends Comparable<P>> Comparator<VecPaired<V, P>>  vecPairedComparator()
    {
        Comparator<VecPaired<V, P>> comp = new Comparator<VecPaired<V, P>>() {

            @Override
            public int compare(VecPaired<V, P> o1, VecPaired<V, P> o2)
            {
                return o1.getPair().compareTo(o2.getPair());
            }
        };
        return comp;
    };

    @Override
    public int hashCode()
    {
        return vector.hashCode();
    }

    @Override
    public boolean isSparse()
    {
        return vector.isSparse();
    }
    
}
