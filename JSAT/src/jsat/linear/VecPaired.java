
package jsat.linear;

/**
 * This data structure allows to wrap a Vector so that it is 
 * associated with some object time. Note, that operations 
 * that return a vector will not be a Paired Vector, as there 
 * is no reason to associate a different vector with this 
 * vector's pair. 
 * 
 * @author Edward Raff
 */
public class VecPaired<P, V extends Vec> extends Vec
{
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
        return vector.add(b);
    }

    @Override
    public Vec subtract(Vec b)
    {
        return vector.subtract(b);
    }

    @Override
    public Vec pairwiseMultiply(Vec b)
    {
        return vector.pairwiseMultiply(b);
    }

    @Override
    public Vec multiply(double c)
    {
        return vector.multiply(c);
    }

    @Override
    public Vec multiply(Matrix A)
    {
        return vector.multiply(A);
    }

    @Override
    public Vec pairwiseDivide(Vec b)
    {
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
        vector.mutableAdd(b);
    }

    @Override
    public void mutableSubtract(Vec b)
    {
        vector.mutableSubtract(b);
    }

    @Override
    public void mutablePairwiseMultiply(Vec b)
    {
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
    public Vec copy()
    {
        return new VecPaired(vector, pair);
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
    
}
