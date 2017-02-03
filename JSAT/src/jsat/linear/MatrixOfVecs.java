package jsat.linear;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class provides a base mechanism to create a Matrix 'view' from a list of 
 * {@link Vec} objects. The vector objects will be used to back the rows of the
 * matrix, so changes to one are shown in the other. If the matirx is altered 
 * using {@link #changeSize(int, int) }, this may no longer be true. <br>
 * Row oriented operations are implemented to defer to the base vector objects 
 * used.
 * 
 * @author Edward Raff
 */
public class MatrixOfVecs extends GenericMatrix
{

	private static final long serialVersionUID = 6120353195388663462L;
	private List<Vec> rows;

    /**
     * Creates a new Matrix of Vecs from the given array of Vec objects. All Vec
     * objects contained should be of the same length. 
     * 
     * @param rows the rows of vecs to make this matrix. 
     */
    public MatrixOfVecs(Vec... rows)
    {
        this(Arrays.asList(rows));
    }
    
    /**
     * Creates a new Matrix of Vecs from the given list of Vec objects. All Vec
     * objects contained should be of the same length. 
     * 
     * @param rows the rows of vecs to make this matrix. 
     */
    public MatrixOfVecs(List<Vec> rows)
    {
        this.rows = new ArrayList<Vec>(rows);
        int cols = rows.get(0).length();
        for(Vec v : rows)
            if(cols != v.length())
                throw new IllegalArgumentException("Row vectors must all be of the same length");
    }

    /**
     * Creates a new Matrix of Vecs of the desired size. 
     * @param rows the number of rows in the matrix
     * @param cols the number of columns in the matrix
     * @param sparse if {@code true} {@link SparseVector} objects will be used 
     * for the rows. Else {@link DenseVector} will be used. 
     */
    public MatrixOfVecs(int rows, int cols, boolean sparse)
    {
        this.rows = new ArrayList<Vec>(rows);
        for(int i = 0; i < rows; i++)
            this.rows.add(sparse ? new SparseVector(cols) : new DenseVector(cols));
    }
    
    @Override
    protected Matrix getMatrixOfSameType(int rows, int cols)
    {
        return new MatrixOfVecs(rows, cols, isSparce());
    }

    @Override
    public void changeSize(int newRows, int newCols)
    {
        if(newRows <= 0 || newCols <= 0)
            throw new IllegalArgumentException("Rows and columns must be positive, new dimension of [" + newRows + "," + newCols + "] is invalid");
        //change cols first, add new rows of the correct size after
        if(newCols != cols())
        {
            for(int i = 0; i < rows(); i++)
            {
                Vec orig = rows.get(i);
                Vec newV = orig.isSparse() ? new SparseVector(newCols) : new DenseVector(newCols);
                if(newCols < orig.length())
                    new SubVector(0, newCols, orig).copyTo(newV);
                else
                    orig.copyTo(new SubVector(0, orig.length(), newV));
                rows.set(i, newV);
            }
        }
        
        if(newRows < rows())
            rows.subList(newRows, rows()).clear();
        else if(newRows > rows())
            while(rows.size() <  newRows)
            {
                Vec newV = rows.get(rows.size()-1).clone();
                newV.zeroOut();
                rows.add(newV);
            }
    }

    @Override
    public double get(int i, int j)
    {
        if(i >= rows() || i < 0)
            throw new IndexOutOfBoundsException("row " + i + " is not a valid index");
        else if(j >= cols() || j < 0)
            throw new IndexOutOfBoundsException("column " + j + " is not a valid index");
        return rows.get(i).get(j);
    }

    @Override
    public void set(int i, int j, double value)
    {
        if(i >= rows() || i < 0)
            throw new IndexOutOfBoundsException("row " + i + " is not a valid index");
        else if(j >= cols() || j < 0)
            throw new IndexOutOfBoundsException("column " + j + " is not a valid index");
        rows.get(i).set(j, value);
    }

    @Override
    public void increment(int i, int j, double value)
    {
        if(i >= rows() || i < 0)
            throw new IndexOutOfBoundsException("row " + i + " is not a valid index");
        else if(j >= cols() || j < 0)
            throw new IndexOutOfBoundsException("column " + j + " is not a valid index");
        rows.get(i).increment(j, value);
    }

    @Override
    public int rows()
    {
        return rows.size();
    }

    @Override
    public int cols()
    {
        return rows.get(0).length();
    }

    @Override
    public Vec getRowView(int r)
    {
        if(r >= rows() || r < 0)
            throw new IndexOutOfBoundsException("row " + r + " is not a valid index");
        return rows.get(r);
    }

    @Override
    public void updateRow(int i, double c, Vec b)
    {
        rows.get(i).mutableAdd(c, b);
    }
    
    @Override
    public void multiply(Vec b, double z, Vec c)
    {
        if(this.cols() != b.length())
            throw new ArithmeticException("Matrix dimensions do not agree, [" + rows() +"," + cols() + "] x [" + b.length() + ",1]" );
        if(this.rows() != c.length())
            throw new ArithmeticException("Target vector dimension does not agree with matrix dimensions. Matrix has " + rows() + " rows but tagert has " + c.length());

        for (int i = 0; i < rows(); i++)
        {
            double dot = this.rows.get(i).dot(b);
            c.increment(i, dot * z);
        }
    }

    @Override
    public void mutableMultiply(double c)
    {
        for(Vec row : rows)
            row.mutableMultiply(c);
    }

    @Override
    public void mutableAdd(double c)
    {
        for(Vec row : rows)
            row.mutableAdd(c);
    }

    @Override
    public void zeroOut()
    {
        for(Vec row : rows)
            row.zeroOut();
    }

    @Override
    public MatrixOfVecs clone()
    {
        MatrixOfVecs clone = new MatrixOfVecs(rows);
        for(int i = 0; i < clone.rows.size(); i++)
            clone.rows.set(i, clone.rows.get(i).clone());
        return clone;
    }
    
    
    @Override
    public boolean isSparce()
    {
        for(Vec v : rows)//TODO probably keep this in a bool
            if(v.isSparse())
                return true;
        return false;
    }
    
}
