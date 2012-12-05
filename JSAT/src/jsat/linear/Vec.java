
package jsat.linear;

import java.util.Iterator;
import java.util.Random;
import jsat.math.Function;
import jsat.math.IndexFunction;

/**
 * Provides the contract for a numerical vector. 
 * 
 * @author Edward Raff
 */
public abstract class Vec implements Cloneable, Iterable<IndexValue>
{
    /**
     * Returns the length of this vector
     * @return the length of this vector
     */
    abstract public int length();

    /**
     * Gets the value stored at a specific index in the vector
     * @param index the index to access
     * @return the double value in the vector
     * @throws IndexOutOfBoundsException if the index given is greater than or 
     * equal to its {@link #length() }
     */
    abstract public double get(int index);

    /**
     * Sets the value stored at a specified index in the vector
     * @param index the index to access
     * @param val the value to store in the index
     * @throws IndexOutOfBoundsException if the index given is greater than or 
     * equal to its {@link #length() }
     */
    abstract public void set(int index, double val);
    
    /**
     * Increments the value stored at a specified index in the vector
     * @param index  the index to access
     * @param val the value to store in the index
     * @throws IndexOutOfBoundsException if the index given is greater than or 
     * equal to its {@link #length() }
     */
    public void increment(int index, double val)
    {
        set(index, val+get(index));
    }
    
    abstract public Vec add(double c);
    abstract public Vec add(Vec b);
    public Vec subtract(double c)
    {
        return add(-c);
    }
    abstract public Vec subtract(Vec b);
    abstract public Vec pairwiseMultiply(Vec b);
    abstract public Vec multiply(double c);
    
    public Vec multiply(Matrix A)
    {
        DenseVector b = new DenseVector(A.cols());
        this.multiply(A, b);
        return b;
    }
    /**
     * If this is vector <tt>a</tt>, this this computes b = b + <tt>a</tt><sup>T</sup>*<tt>A</tt>
     * @param A
     * @param b 
     */
    abstract public void multiply(Matrix A, Vec b);
    abstract public Vec pairwiseDivide(Vec b);
    abstract public Vec divide(double c);
    
    /**
     * Alters this vector such that 
     * <tt>this</tt> = <tt>this</tt> + <tt>c</tt>
     * @param c a scalar constant to add to each value in this vector
     */
    abstract public void mutableAdd(double c);
    /**
     * Alters this vector such that 
     * <tt>this</tt> = <tt>this</tt> + <tt>c</tt> * <tt>b</tt>
     * @param c a scalar constant
     * @param b the vector to add to this
     */
    abstract public void mutableAdd(double c, Vec b);
    
    /**
     * Alters this vector such that
     * <tt>this</tt> = <tt>this</tt> + <tt>b</tt>
     * @param b the vector to add to this
     * @throws ArithmeticException if the vectors do not have the same length
     */
    public void mutableAdd(Vec b)
    {
        this.mutableAdd(1, b);
    }
    
    /**
     * Alters this vector such that
     * <tt>this</tt> = <tt>this</tt> - <tt>c</tt>
     * @param c the scalar constant to subtract from all values in this vector
     */
    public void mutableSubtract(double c)
    {
        mutableAdd(-c);
    }
    
    /**
     * Alters this vector such that 
     * <tt>this</tt> = <tt>this</tt> - <tt>c</tt> * <tt>b</tt>
     * @param c a scalar constant
     * @param b the vector to subtract from this
     * @throws ArithmeticException if the vectors do not have the same length
     */
    public void mutableSubtract(double c, Vec b)
    {
        this.mutableAdd(-c, b);
    }
    
    /**
     * Alters this vector such that
     * <tt>this</tt> = <tt>this</tt> - <tt>b</tt>
     * @param b the vector to subtract from this
     * @throws ArithmeticException if the vectors are not the same length
     */
    public void mutableSubtract(Vec b)
    {
        this.mutableAdd(-1, b);
    }
    abstract public void mutablePairwiseMultiply(Vec b);
    abstract public void mutableMultiply(double c);
    abstract public void mutablePairwiseDivide(Vec b);
    abstract public void mutableDivide(double c);

    abstract public Vec sortedCopy();

    /**
     * Returns the minimum value stored in this vector
     * @return the minimum value in this vector
     */
    abstract public double min();
    /**
     * Returns the maximum value stored in this vector
     * @return the maximum value in this vector
     */
    abstract public double max();
    /**
     * Computes the sum of the values in this vector
     * @return the sum of this vector's values
     */
    abstract public double sum();
    /**
     * Computes the mean value of all values stored in this vector
     * @return the mean value
     */
    abstract public double mean();
    /**
     * Computes the standard deviation of the values in this vector
     * @return the standard deviation
     */
    abstract public double standardDeviation();
    /**
     * Computes the variance of the values in this vector
     * @return the variance 
     */
    abstract public double variance();
    /**
     * Returns the median value in this vector
     * @return the median
     */
    abstract public double median();
    /**
     * Computes the skewness of this vector, which is the 3rd moment. 
     * @return the skewness
     */
    abstract public double skewness();
    /**
     * Computes the kurtosis of this vector, which is the 4th moment. 
     * @return the kurtosis
     */
    abstract public double kurtosis();
    
    /**
     * Indicates whether or not this vector is optimized for sparce computation,
     * meaning that most values in the vector are zero - and considered 
     * implicit. Only non-zero values are stored. 
     * @return <tt>true</tt> if the vector is sparce, <tt>false</tt> otherwise. 
     */
    abstract public boolean isSparse();
    
    /**
     * Copies the values of this Vector into another vector
     * @param destination the vector to store the values in. 
     * @throws ArithmeticException if the vectors are not of the same length
     */
    public void copyTo(Vec destination)
    {
        if(this.length() != destination.length())
            throw new ArithmeticException("Source and destination must be the same size");
        for(int i = 0; i < length(); i++)
            destination.set(i, this.get(i));
    }
    
    /**
     * Copies the values of this vector into a row of another Matrix
     * @param A the matrix to store the contents of this vector in
     * @param row the row of the matrix to store the values to
     * @throws ArithmeticException if the columns of the matrix is not the same as the length of this vector. 
     */
    public void copyToRow(Matrix A, int row)
    {
        if(this.length() != A.cols())
            throw new ArithmeticException("Destination matrix does not have the same number of columns as this has rows");
        for(int i = 0; i < length(); i++)
            A.set(row, i, get(i));
    }
    
    /**
     * Copies the values of this vector into a column of another Matrix. 
     * @param A the matrix to store the contents of this vector in
     * @param col the column of the matrix to store the values to
     */
    public void copyToCol(Matrix A, int col)
    {
        if(this.length() != A.rows())
            throw new ArithmeticException("Destination matrix does not have the same number of rows as this has rows");
        for(int i = 0; i < length(); i++)
            A.set(i, col, get(i));
    }
    
    @Override
    abstract public Vec clone();
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
    public String toString()
    {
        StringBuilder sb = new StringBuilder("[");
        sb.append(get(0));
        for(int i = 1; i < length(); i++)
            sb.append(",").append(get(i));
        sb.append("]");
        return sb.toString();
    }
    
    @Override
    abstract public boolean equals(Object obj);
    
    abstract public boolean equals(Object obj, double range);
    
    abstract public double[] arrayCopy();

    @Override
    public Iterator<IndexValue> iterator()
    {
        return getNonZeroIterator();
    }
    
    /**
     * Returns an iterator that will go over the non zero values in the given vector. The iterator does not 
     * support the {@link Iterator#remove() } method. Note, that values with zero are permissible to be 
     * returned by this method. Dense structures that do not retain this information, and may have few 
     * zeros, are allowed to return them. Structures that are aware of sparseness are expected to 
     * return only the non zero values for speed efficency.
     * 
     * @return an iterator for the non zero index value pairs. 
     */
    public Iterator<IndexValue> getNonZeroIterator()
    {
        //Need a little class magic
        final Vec magic = this;
        int i;
        for(i = 0; i < magic.length(); i++)
            if(magic.get(i) != 0.0)
                break;
        final int fnz = (magic.length() == 0 || magic.get(i) == 0.0 ) ? -1 : i;
        Iterator<IndexValue> itor = new Iterator<IndexValue>() 
        {
            int curIndex = 0;
            int nextNonZero = fnz;
            
            IndexValue indexValue = new IndexValue(-1, Double.NaN);
            
            @Override
            public boolean hasNext()
            {
                return nextNonZero >= 0;
            }

            @Override
            public IndexValue next()
            {
                if(nextNonZero == -1)
                    return null;
                indexValue.setIndex(nextNonZero);
                indexValue.setValue(get(nextNonZero));
                
                int i = nextNonZero+1;
                nextNonZero = -1;
                for(; i < magic.length(); i++ )
                    if(get(i) != 0.0)
                    {
                        nextNonZero = i;
                        break;
                    }
                
                return indexValue;
            }

            @Override
            public void remove()
            {
                throw new UnsupportedOperationException("Not supported yet.");
            }
        };
        
        return itor;
    }
    
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
    
    /**
     * Creates a dense vector full of random values in the range [0, 1]
     * @param length the length of the random vector to create
     * @return a random vector of the specified length
     */
    public static Vec random(int length)
    {
        return random(length, new Random());
    }
    
    /**
     * Creates a dense vector full of random values in the range [0, 1]
     * @param length the length of the random vector to create
     * @param rand the source of randomness
     * @return a random vector of the specified length
     */
    public static Vec random(int length, Random rand)
    {
        Vec v = new DenseVector(length);
        for(int i = 0; i < length; i++)
            v.set(i, rand.nextDouble());
        return v;
    }
    
    /**
     * Creates a dense vector full of zeros. 
     * @param length the length of the vector to create
     * @return a vector of zeros
     */
    public static Vec zeros(int length)
    {
        return new DenseVector(length);
    }

}
