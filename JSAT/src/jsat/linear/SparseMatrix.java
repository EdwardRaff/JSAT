package jsat.linear;

import java.util.Arrays;
import java.util.Iterator;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.utils.SystemInfo;

/**
 * Creates a new Sparse Matrix where each row is backed by a sparse vector. 
 * <br><br>
 * This implementation does not support the {@link #qr() QR} or {@link #lup() } 
 * decompositions. 
 * <br>
 * {@link #transposeMultiply(jsat.linear.Matrix, jsat.linear.Matrix, java.util.concurrent.ExecutorService) } currently does not use multiple cores. 
 * 
 * @author Edward Raff
 */
public class SparseMatrix extends Matrix
{

	private static final long serialVersionUID = -4087445771022578544L;
	private SparseVector[] rows;

    /**
     * Creates a new sparse matrix
     * @param rows the number of rows for the matrix
     * @param cols the number of columns for the matrix
     * @param rowCapacity the initial capacity for non zero values for each row
     */
    public SparseMatrix(int rows, int cols, int rowCapacity)
    {
        this.rows = new SparseVector[rows];
        for(int i = 0; i < rows; i++)
            this.rows[i] = new SparseVector(cols, rowCapacity);
    }
    
    /**
     * Creates a new Sparse Matrix backed by the given array of SpareVectors. 
     * Altering the array of any object in it will also alter the this matrix. 
     * 
     * @param rows the array to back this SparseMatrix
     */
    public SparseMatrix(SparseVector[] rows)
    {
        this.rows = rows;
        for(int i = 0; i < rows.length; i++)
            if(rows[i].length() != rows[0].length())
                throw new IllegalArgumentException("Row " + i + " has " + rows[i].length() + " columns instead of " + rows[0].length());
    }
    
    /**
     * Creates a new sparse matrix
     * @param rows the number of rows for the matrix
     * @param cols the number of columns for the matrix
     */
    public SparseMatrix(int rows, int cols)
    {
        this.rows = new SparseVector[rows];
        for(int i = 0; i < rows; i++)
            this.rows[i] = new SparseVector(cols);
    }
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    protected SparseMatrix(SparseMatrix toCopy)
    {
        this.rows = new SparseVector[toCopy.rows.length];
        for(int i = 0; i < rows.length; i++)
            this.rows[i] = toCopy.rows[i].clone();
    }

    @Override
    public void mutableAdd(double c, Matrix B)
    {
        if(!Matrix.sameDimensions(this, B))
            throw new ArithmeticException("Matrices must be the same dimension to be added");
        for( int i = 0; i < rows.length; i++)
            rows[i].mutableAdd(c, B.getRowView(i));
    }

    @Override
    public void mutableAdd(final double c, final Matrix B, ExecutorService threadPool)
    {
        if(!Matrix.sameDimensions(this, B))
            throw new ArithmeticException("Matrices must be the same dimension to be added");
        
        final CountDownLatch latch = new CountDownLatch(rows.length);
        for (int i = 0; i < rows.length; i++)
        {
            final int ii = i;
            threadPool.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    rows[ii].mutableAdd(c, B.getRowView(ii));
                    latch.countDown();
                }
            });
        }
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(SparseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public void mutableAdd(double c)
    {
        for(SparseVector row : rows)
            row.mutableAdd(c);
    }

    @Override
    public void mutableAdd(final double c, ExecutorService threadPool)
    {
        final CountDownLatch latch = new CountDownLatch(rows.length);
        for(final SparseVector row : rows)
        {
            threadPool.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    row.mutableAdd(c);
                    latch.countDown();
                }
            });
        }
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(SparseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public void multiply(Vec b, double z, Vec c)
    {
        if(this.cols() != b.length())
            throw new ArithmeticException("Matrix dimensions do not agree, [" + rows() +"," + cols() + "] x [" + b.length() + ",1]" );
        if(this.rows() != c.length())
            throw new ArithmeticException("Target vector dimension does not agree with matrix dimensions. Matrix has " + rows() + " rows but tagert has " + c.length());
        
        for(int i = 0; i < rows(); i++)
        {
            SparseVector row = rows[i];
            c.increment(i, row.dot(b)*z);
        }
    }

    @Override
    public void multiply(Matrix B, Matrix C)
    {
        if(!canMultiply(this, B))
            throw new ArithmeticException("Matrix dimensions do not agree");
        else if(this.rows() != C.rows() || B.cols() != C.cols())
            throw new ArithmeticException("Target Matrix is no the correct size");
        
        for (int i = 0; i < C.rows(); i++)
        {
            Vec Arowi = this.rows[i];
            Vec Crowi = C.getRowView(i);

            for(IndexValue iv : Arowi)
            {
                final int k = iv.getIndex();
                double a = iv.getValue();
                Vec Browk = B.getRowView(k);
                Crowi.mutableAdd(a, Browk);
            }
        }
    }

    @Override
    public void multiply(final Matrix B, Matrix C, ExecutorService threadPool)
    {
        if (!canMultiply(this, B))
            throw new ArithmeticException("Matrix dimensions do not agree");
        else if (this.rows() != C.rows() || B.cols() != C.cols())
            throw new ArithmeticException("Target Matrix is no the correct size");

        final CountDownLatch latch = new CountDownLatch(C.rows());
        for (int i = 0; i < C.rows(); i++)
        {
            final Vec Arowi = this.rows[i];
            final Vec Crowi = C.getRowView(i);

            threadPool.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    for (IndexValue iv : Arowi)
                    {
                        final int k = iv.getIndex();
                        double a = iv.getValue();
                        Vec Browk = B.getRowView(k);
                        Crowi.mutableAdd(a, Browk);
                    }
                    
                    latch.countDown();
                }
            });
        }
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(SparseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public void mutableMultiply(double c)
    {
        for(SparseVector row : rows)
            row.mutableMultiply(c);
    }

    @Override
    public void mutableMultiply(final double c, ExecutorService threadPool)
    {
        final CountDownLatch latch = new CountDownLatch(rows.length);
        for(final SparseVector row : rows)
        {
            threadPool.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    row.mutableMultiply(c);
                    latch.countDown();
                }
            });
        }
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(SparseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public Matrix[] lup()
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Matrix[] lup(ExecutorService threadPool)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Matrix[] qr()
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Matrix[] qr(ExecutorService threadPool)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void mutableTranspose()
    {
        for(int i = 0; i < rows()-1; i++)
            for(int j = i+1; j < cols(); j++)
            {
                double tmp = get(j, i);
                set(j, i, get(i, j));
                set(i, j, tmp);
            }
    }

    @Override
    public void transpose(Matrix C)
    {
        if(this.rows() != C.cols() || this.cols() != C.rows())
            throw new ArithmeticException("Target matrix does not have the correct dimensions");
        
        C.zeroOut();
        for(int row = 0; row < rows.length; row++)
            for(IndexValue iv : rows[row])
                C.set(iv.getIndex(), row, iv.getValue());
    }

    @Override
    public void transposeMultiply(Matrix B, Matrix C)
    {
        if(this.rows() != B.rows())//Normaly it is A_cols == B_rows, but we are doint A'*B, not A*B
            throw new ArithmeticException("Matrix dimensions do not agree");
        else if(this.cols() != C.rows() || B.cols() != C.cols())
            throw new ArithmeticException("Destination matrix does not have matching dimensions");
        final SparseMatrix A = this;
        ///Should choose step size such that 2*NB2^2 * dataTypeSize <= CacheSize
        
        final int kLimit = this.rows();

        for (int k = 0; k < kLimit; k++)
        {
            Vec bRow_k = B.getRowView(k);
            Vec aRow_k = A.getRowView(k);

            for (IndexValue iv : aRow_k)//iterating over "i"
            {

                Vec cRow_i = C.getRowView(iv.getIndex());
                double a = iv.getValue();//A.get(k, i);

                cRow_i.mutableAdd(a, bRow_k);
            }
        }
    }

    @Override
    public void transposeMultiply(final Matrix B, final Matrix C, ExecutorService threadPool)
    {
        transposeMultiply(B, C);//TODO use the multiple threads
    }

    @Override
    public void transposeMultiply(double c, Vec b, Vec x)
    {
        if(this.rows() != b.length())
            throw new ArithmeticException("Matrix dimensions do not agree, [" + cols() +"," + rows() + "] x [" + b.length() + ",1]" );
        else if(this.cols() != x.length())
            throw new ArithmeticException("Matrix dimensions do not agree with target vector");
        
        for(IndexValue b_iv : b)
            x.mutableAdd(c*b_iv.getValue(), rows[b_iv.getIndex()]);
    }

    @Override
    public Vec getRowView(int r)
    {
        return rows[r];
    }

    @Override
    public double get(int i, int j)
    {
        return rows[i].get(j);
    }

    @Override
    public void set(int i, int j, double value)
    {
        rows[i].set(j, value);
    }

    @Override
    public void increment(int i, int j, double value)
    {
        rows[i].increment(j, value);
    }

    @Override
    public int rows()
    {
        return rows.length;
    }

    @Override
    public int cols()
    {
        return rows[0].length();
    }

    @Override
    public boolean isSparce()
    {
        return true;
    }

    @Override
    public void swapRows(int r1, int r2)
    {
        SparseVector tmp = rows[r2];
        rows[r2] = rows[r1];
        rows[r1] = tmp;
    }

    @Override
    public void zeroOut()
    {
        for(Vec row : rows)
            row.zeroOut();
    }

    @Override
    public SparseMatrix clone()
    {
        return new SparseMatrix(this);
    }

    @Override
    public long nnz()
    {
        int nnz = 0;
        for(Vec v : rows)
            nnz += v.nnz();
        return nnz;
    }

    @Override
    public void changeSize(int newRows, int newCols)
    {
        if(newRows <= 0)
            throw new ArithmeticException("Matrix must have a positive number of rows");
        if(newCols <= 0)
            throw new ArithmeticException("Matrix must have a positive number of columns");
        final int oldRows = rows.length;
        if(newCols != cols())
        {
            for(int i = 0; i < rows.length; i++)
            {
                final SparseVector row_i = rows[i];
                while(row_i.getLastNonZeroIndex() >= newCols)
                    row_i.set(row_i.getLastNonZeroIndex(), 0);
                row_i.setLength(newCols);
            }
        }
        //update new rows
        rows = Arrays.copyOf(rows, newRows);
        for(int i = oldRows; i < newRows; i++)
            rows[i] = new SparseVector(newCols);
    }

    @Override
    public void multiplyTranspose(Matrix B, Matrix C)
    {
        if(this.cols() != B.cols())
            throw new ArithmeticException("Matrix dimensions do not agree");
        else if (this.rows() != C.rows() || B.rows() != C.cols())
            throw new ArithmeticException("Target Matrix is no the correct size");

        for (int i = 0; i < this.rows(); i++)
        {
            final SparseVector A_i = this.rows[i];
            for (int j = 0; j < B.rows(); j++)
            {
                final Vec B_j = B.getRowView(j);
                double C_ij = 0;
                
                if(!B_j.isSparse())//B is dense, lets do this the easy way
                {
                    for (IndexValue iv : A_i)
                        C_ij += iv.getValue() * B_j.get(iv.getIndex());
                    C.increment(i, j, C_ij);
                    continue;//Skip early, we did it!
                }
                //else, sparse 
                Iterator<IndexValue> A_iter = A_i.getNonZeroIterator();
                Iterator<IndexValue> B_iter = B_j.getNonZeroIterator();
                if(!B_iter.hasNext() || !A_iter.hasNext())//one is all zeros, nothing to do
                    continue;
                
                IndexValue A_val = A_iter.next();
                IndexValue B_val = B_iter.next();
                
                while(A_val != null && B_val != null)//go add everything together!
                {
                    if(A_val.getIndex() == B_val.getIndex())//inc and bump both
                    {
                        C_ij += A_val.getValue()*B_val.getValue();
                        if(A_iter.hasNext())
                            A_val = A_iter.next();
                        else
                            A_val = null;
                        if(B_iter.hasNext())
                            B_val = B_iter.next();
                        else
                            B_val = null;
                    }
                    else if(A_val.getIndex() < B_val.getIndex())//A is behind, bump it
                    {
                        if(A_iter.hasNext())
                            A_val = A_iter.next();
                        else
                            A_val = null;
                    }
                    else//B is behind, bump it
                    {
                        if(B_iter.hasNext())
                            B_val = B_iter.next();
                        else
                            B_val = null;
                    }
                }

                C.increment(i, j, C_ij);
            }
        }
    }

    @Override
    public void multiplyTranspose(final Matrix B, final Matrix C, ExecutorService threadPool)
    {
        if(this.cols() != B.cols())
            throw new ArithmeticException("Matrix dimensions do not agree");
        else if (this.rows() != C.rows() || B.rows() != C.cols())
            throw new ArithmeticException("Target Matrix is no the correct size");

        final SparseMatrix A = this;
        final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
        for(int id = 0; id < SystemInfo.LogicalCores; id++)
        {
            final int ID = id;
            threadPool.submit(new Runnable()
            {

                @Override
                public void run()
                {
                    try{
                    for (int i = ID; i < A.rows(); i += SystemInfo.LogicalCores)
                    {
                        final SparseVector A_i = A.rows[i];
                        for (int j = 0; j < B.rows(); j++)
                        {
                            final Vec B_j = B.getRowView(j);
                            double C_ij = 0;

                            if(!B_j.isSparse())//B is dense, lets do this the easy way
                            {
                                for (IndexValue iv : A_i)
                                    C_ij += iv.getValue() * B_j.get(iv.getIndex());
                                C.increment(i, j, C_ij);
                                continue;//Skip early, we did it!
                            }
                            //else, sparse 
                            Iterator<IndexValue> A_iter = A_i.getNonZeroIterator();
                            Iterator<IndexValue> B_iter = B_j.getNonZeroIterator();
                            if(!B_iter.hasNext() || !A_iter.hasNext())//one is all zeros, nothing to do
                                continue;

                            IndexValue A_val = A_iter.next();
                            IndexValue B_val = B_iter.next();

                            while(A_val != null && B_val != null)//go add everything together!
                            {
                                if(A_val.getIndex() == B_val.getIndex())//inc and bump both
                                {
                                    C_ij += A_val.getValue()*B_val.getValue();
                                    if(A_iter.hasNext())
                                        A_val = A_iter.next();
                                    else
                                        A_val = null;
                                    if(B_iter.hasNext())
                                        B_val = B_iter.next();
                                    else
                                        B_val = null;
                                }
                                else if(A_val.getIndex() < B_val.getIndex())//A is behind, bump it
                                {
                                    if(A_iter.hasNext())
                                        A_val = A_iter.next();
                                    else
                                        A_val = null;
                                }
                                else//B is behind, bump it
                                {
                                    if(B_iter.hasNext())
                                        B_val = B_iter.next();
                                    else
                                        B_val = null;
                                }
                            }

                            C.increment(i, j, C_ij);
                        }
                    }
                    
                    }
                    catch(Exception ex)
                    {
                        ex.printStackTrace();
                    }
                    System.out.println(ID + " fin");
                    latch.countDown();
                }
            });
        }
        
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(SparseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}
