
package jsat.linear;

import jsat.math.Function;
import jsat.math.IndexFunction;

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
    /**
     * Alters this vector such that <br>
     * <tt>this</tt> = <tt>this</tt> + <tt>c</tt> * <tt>b</tt>
     * @param c a scalar constant
     * @param b the vector to add tot his
     */
    abstract public void mutableAdd(double c, Vec b);
    public void mutableAdd(Vec b)
    {
        this.mutableAdd(1, b);
    }
    
    public void mutableSubtract(double c)
    {
        mutableAdd(-c);
    }
    
    public void mutableSubtract(double c, Vec b)
    {
        this.mutableAdd(-c, b);
    }
    
    public void mutableSubtract(Vec b)
    {
        this.mutableAdd(-1, b);
    }
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
    
    /**
     * Applies the given function to each and every value in the vector. 
     * @param f the single variable function to apply
     */
    public void applyFunction(Function f)
    {
        for(int i = 0; i < length(); i++)
            set(i, f.f(get(i)));
    }
    
    /**
     * Applies the given function to each and every value in the vector. 
     * The function takes 2 arguments, an arbitrary value, and then an 
     * index. The index passed to the function is the index in the array
     * that the value came from. 
     * <br><br>
     * <b><i>NOTE:</b></i> Because negative values are invalid indexes. 
     * The given function should return 0.0 when given a negative index,
     * if and only if, f(0,index) = 0 for any valid index. If f(0, index)
     * != 0 for even one value of index, it should return any non zero 
     * value when given a negative index. 
     * <br><br>
     * IE: f(value_i, i) = x 
     * 
     * @param f the 2 dimensional index function to apply 
     */
    public void applyIndexFunction(IndexFunction f)
    {
        for(int i = 0; i < length(); i++)
            set(i, f.indexFunc(get(i), i));
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
    
    /**
     * Zeroes out all values in this vector
     */
    public void zeroOut()
    {
        for(int i = 0; i < length(); i++)
            set(i, 0.0);
    }

    /**
     * Provides a hashcode for Vectors. All vector implementations should return the 
     * same result for cases when {@link #equals(java.lang.Object) } returns true. 
     * Below is the code used for this class<br>
     * <p><code>
     * int result = 1;<br>
     * <br>
     *   for (int i = 0; i < length(); i++) <br>
     *   {<br>
     *       double val = get(i);<br>
     *       if(val != 0)<br>
     *       {<br>
     *           long bits = Double.doubleToLongBits(val);<br>
     *           result = 31 * result + (int)(bits ^ (bits >>> 32));<br>
     *           result = 31 * result + i;<br>
     *       }<br>
     *   }<br>
     *   <br>
     *   return 31* result + length();<br>
     * </code></p>
     * @return 
     */
    @Override
    public int hashCode()
    {
        int result = 1;
        
        for (int i = 0; i < length(); i++) 
        {
            double val = get(i);
            if(val != 0)
            {
                long bits = Double.doubleToLongBits(val);
                result = 31 * result + (int)(bits ^ (bits >>> 32));
                result = 31 * result + i;
            }
        }
        
        return 31* result + length();
    }

}
