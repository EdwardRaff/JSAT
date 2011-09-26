
package jsat.linear;

import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public abstract class Vec
{
    abstract public int length();

    abstract public double get(int index);

    abstract public void set(int index, double val);
    
    abstract public Vec add(double c);
    abstract public Vec add(Vec b);
    public Vec subtract(double c)
    {
        return add(-c);
    }
    abstract public Vec subtract(Vec b);
    abstract public Vec pairwiseMultiply(Vec b);
    abstract public Vec multiply(double c);
    abstract public Vec multiply(Matrix A);
    abstract public Vec pairwiseDivide(Vec b);
    abstract public Vec divide(double c);
    
    abstract public void mutableAdd(double c);
    abstract public void mutableAdd(Vec b);
    public void mutableSubtract(double c)
    {
        mutableAdd(-c);
    }
    abstract public void mutableSubtract(Vec b);
    abstract public void mutablePairwiseMultiply(Vec b);
    abstract public void mutableMultiply(double c);
    abstract public void mutablePairwiseDivide(Vec b);
    abstract public void mutableDivide(double c);

    abstract public Vec sortedCopy();

    abstract public double min();
    abstract public double max();
    abstract public double sum();
    abstract public double mean();
    abstract public double standardDeviation();
    abstract public double variance();
    abstract public double median();
    abstract public double skewness();
    abstract public double kurtosis();
    
    public void copyTo(Vec destination)
    {
        if(this.length() != destination.length())
            throw new ArithmeticException("Source and destination must be the same size");
        for(int i = 0; i < length(); i++)
            destination.set(i, this.get(i));
    }
    abstract public Vec copy();
    abstract public Vec normalized();
    abstract public void normalize();
    
    public void applyFunction(Function f)
    {
        for(int i = 0; i < length(); i++)
            set(i, f.f(get(i)));
    }
    
    /**
     * Returns the p-norm distance between this and another vector y. 
     * @param p the distance type. 2 is the common value
     * @param y the other vector to compare against
     * @return the p-norm distance
     */
    abstract public double pNormDist(double p, Vec y);
    
    abstract public double pNorm(double p);
    
    /**
     * 
     * @param v the other vector
     * @return  the dot product of this vector and another
     */
    abstract public double dot(Vec v);

    @Override
    abstract public String toString();
    
    @Override
    abstract public boolean equals(Object obj);
    
    abstract public boolean equals(Object obj, double range);
    
    abstract public double[] arrayCopy();

}
