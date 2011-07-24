
package jsat.linear;

import java.util.concurrent.ExecutorService;

/**
 *
 * @author Edward Rafff
 */
public abstract class Matrix
{
    public Matrix add(Matrix b)
    {
        Matrix toReturn = copy();
        toReturn.mutableAdd(b);
        return toReturn;
    }
    public Matrix add(Matrix b, ExecutorService threadPool)
    {
        Matrix toReturn = copy();
        toReturn.mutableAdd(b, threadPool);
        return toReturn;
    }
    public Matrix add(double c)
    {
        Matrix toReturn = copy();
        toReturn.mutableAdd(c);
        return toReturn;
    }
    public Matrix add(double c, ExecutorService threadPool)
    {
        Matrix toReturn = copy();
        toReturn.mutableAdd(c, threadPool);
        return toReturn;
    }
    abstract public void mutableAdd(Matrix b);
    abstract public void mutableAdd(Matrix b, ExecutorService threadPool);
    abstract public void mutableAdd(double c);
    abstract public void mutableAdd(double c, ExecutorService threadPool);
    
    
    public Matrix subtract(Matrix b)
    {
        Matrix toReturn = copy();
        toReturn.mutableSubtract(b);
        return toReturn;
    }
    public Matrix subtract(Matrix b, ExecutorService threadPool)
    {
        Matrix toReturn = copy();
        toReturn.mutableSubtract(b, threadPool);
        return toReturn;
    }
    public Matrix subtract(double c)
    {
        Matrix toReturn = copy();
        toReturn.mutableSubtract(c);
        return toReturn;
    }
    public Matrix subtract(double c, ExecutorService threadPool)
    {
        Matrix toReturn = copy();
        toReturn.mutableSubtract(c, threadPool);
        return toReturn;
    }
    abstract public void mutableSubtract(Matrix b);
    abstract public void mutableSubtract(Matrix b, ExecutorService threadPool);
    public void mutableSubtract(double c)
    {
        mutableAdd(-c);
    }
    public void mutableSubtract(double c, ExecutorService threadPool)
    {
        mutableAdd(-c, threadPool);
    }
    
    /**
     * If this matrix is A_(m x n), and <tt>b</tt> has a length of n, then this will compute the result of A*b 
     * @param b the vector to be treated as a colum vector
     * @return the Vector result of the computation
     */
    abstract public Vec multiply(Vec b);
    abstract public Vec multiply(Vec b, ExecutorService threadPool);
    abstract public Matrix multiply(Matrix b);
    abstract public Matrix multiply(Matrix b, ExecutorService threadPool);
    public Matrix multiply(double c)
    {
        Matrix toReturn = copy();
        toReturn.mutableMultiply(c);
        return toReturn;
    }
    public Matrix multiply(double c, ExecutorService threadPool)
    {
        Matrix toReturn = copy();
        toReturn.mutableMultiply(c, threadPool);
        return toReturn;
    }
    abstract public void mutableMultiply(double c);
    abstract public void mutableMultiply(double c, ExecutorService threadPool);
    
    abstract public double get(int i, int j);
    abstract public void set(int i, int j, double value);
    
    abstract public int rows();
    abstract public int cols();
    
    abstract public boolean isSparce();
    abstract public long nnz();
    public boolean isSquare()
    {
        return rows() == cols();
    }
    
    abstract public Matrix copy();

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder(rows()*cols());
        
        sb.append("[");
        
        for(int i = 0; i < rows(); i++)
        {
            sb.append(get(i, 0));
            for(int j = 1; j < cols(); j++)
            {
                sb.append(", ").append(get(i, j));
            }
            sb.append(";");
        }
        
        sb.append("]");
        return sb.toString();
    }
    
    public static boolean sameDimensions(Matrix a, Matrix b)
    {
        return a.rows() == b.rows() && a.cols() == b.cols();
    }
    
    public static boolean canMultiply(Matrix a, Matrix b)
    {
        return a.cols() == b.rows();
    }
    
    
}
