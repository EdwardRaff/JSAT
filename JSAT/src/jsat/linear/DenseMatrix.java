
package jsat.linear;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public class DenseMatrix extends Matrix
{
    private final double[][] matrix;

    /**
     * Creates a new matrix based off the given vectors. 
     * @param a the first Vector, this new Matrix will have as many rows as the length of this vector
     * @param b the second Vector, this new Matrix will have as many columns as this length of this vector
     */
    public DenseMatrix(Vec a, Vec b)
    {
        matrix = new double[a.length()][b.length()];
        for(int i = 0; i < a.length(); i++)
        {
            Vec rowVals = b.multiply(a.get(i));
            for(int j = 0; j < b.length(); j++)
                matrix[i][j] = rowVals.get(j);
        }
    }
    
    public DenseMatrix(int rows, int cols)
    {
        matrix = new double[rows][cols];
    }
    
    private class MuttableAddRun implements Runnable
    {
        CountDownLatch latch;
        int row;
        Matrix otherSource;

        public MuttableAddRun(CountDownLatch latch, int row, Matrix otherSource)
        {
            this.latch = latch;
            this.row = row;
            this.otherSource = otherSource;
        }

        public void run()
        {
            for(int j = 0; j < matrix[row].length; j++)
                matrix[row][j] += otherSource.get(row, j);
            latch.countDown();
        }
    }
    
    @Override
    public void mutableAdd(Matrix b)
    {
        if(!sameDimensions(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                this.matrix[i][j] += b.get(i, j);
    }

    @Override
    public void mutableAdd(Matrix b, ExecutorService threadPool)
    {
        if(!sameDimensions(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        
        CountDownLatch latch = new CountDownLatch(rows());
        
        for(int i = 0; i < rows(); i++)
            threadPool.submit(new MuttableAddRun(latch, i, b));
        
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            //Eww, mutable failure is ugly
            Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    @Override
    public void mutableAdd(double c)
    {
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                matrix[i][j] += c;
    }
    
    private class MuttableAddConstRun implements Runnable
    {
        final CountDownLatch latch;
        final double[] row;
        final double constant;

        public MuttableAddConstRun(CountDownLatch latch, double[] row, double constant)
        {
            this.latch = latch;
            this.row = row;
            this.constant = constant;
        }

        public void run()
        {
            for(int j = 0; j < row.length; j++)
                row[j] += constant;
            latch.countDown();
        }
    }

    @Override
    public void mutableAdd(double c, ExecutorService threadPool)
    {
        CountDownLatch latch = new CountDownLatch(rows());
        
        for(int i = 0; i < rows(); i++)
            threadPool.submit(new MuttableAddConstRun(latch, matrix[i], c));
        
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    
    private class MuttableSubRun implements Runnable
    {
        CountDownLatch latch;
        int row;
        Matrix otherSource;

        public MuttableSubRun(CountDownLatch latch, int row, Matrix otherSource)
        {
            this.latch = latch;
            this.row = row;
            this.otherSource = otherSource;
        }

        public void run()
        {
            for(int j = 0; j < matrix[row].length; j++)
                matrix[row][j] -= otherSource.get(row, j);
            latch.countDown();
        }
    }

    @Override
    public void mutableSubtract(Matrix b)
    {
        if(!sameDimensions(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                this.matrix[i][j] -= b.get(i, j);
    }

    @Override
    public void mutableSubtract(Matrix b, ExecutorService threadPool)
    {
        if(!sameDimensions(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        
        CountDownLatch latch = new CountDownLatch(rows());
        
        for(int i = 0; i < rows(); i++)
            threadPool.submit(new MuttableSubRun(latch, i, b));
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public Vec multiply(Vec b)
    {
        if(this.cols() != b.length())
            throw new ArithmeticException("Matrix dimensions do not agree, [" + rows() +"," + cols() + "] x [" + b.length() + ",1]" );
        
        
        DenseVector result = new DenseVector(rows());
        for(int i = 0; i < rows(); i++)
        {
            //The Dense construcure does not copy the matrix, it just takes the refernce -making it fast
            DenseVector row = new DenseVector(matrix[i]);
            result.set(i, row.dot(b));//We use the dot product in this way so that if the incoming matrix is sparce, we can take advantage of save computaitons
        }
        
        return result;
    }
    
    private class VecMultiRun implements Callable<Double>
    {
        final Vec row;
        final Vec b;

        public VecMultiRun(Vec row, Vec b)
        {
            this.row = row;
            this.b = b;
        }

        public Double call() throws Exception
        {
            return row.dot(b);
        }
       
        
        
    }

    @Override
    public Vec multiply(Vec b, ExecutorService threadPool)
    {
        if(this.cols() != b.length())
            throw new ArithmeticException("Matrix dimensions do not agree");
        
        DenseVector result = new DenseVector(rows());
        
        List<Future<Double>> vecVals = new ArrayList<Future<Double>>(rows());
        for(int i = 0; i < rows(); i++)
        {
            DenseVector row = new DenseVector(matrix[i]);
            vecVals.add(threadPool.submit(new VecMultiRun(row, b)));
        }
        
        try
        {
            for (int i = 0; i < vecVals.size(); i++)
            {
                result.set(i, vecVals.get(i).get());
            }
        }
        catch (InterruptedException interruptedException)
        {
        }
        catch (ExecutionException executionException)
        {
            
        }
        
        return result;
    }

    /**
     * 
     * @param b
     * @return 
     */
    private Matrix pureRowOrderMultiply(Matrix b)
    {
        if(!canMultiply(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        DenseMatrix result = new DenseMatrix(this.rows(), b.cols());
        
        /*
         * In stead of row echelon order (i, j, k), we compue in "pure row oriented"
         * 
         * see
         * 
         * Data structures in Java for matrix computations
         * 
         * CONCURRENCY AND COMPUTATION: PRACTICE AND EXPERIENCE
         * Concurrency Computat.: Pract. Exper. 2004; 16:799–815 (DOI: 10.1002/cpe.793)
         * 
         */
        
        //Pull out the index operations to hand optimize for speed. 
        double[] Arowi;
        double[] Crowi;
        for(int i = 0; i < result.rows(); i++)
        {
            Arowi = this.matrix[i];
            Crowi = result.matrix[i];
            
            for(int k = 0; k < this.cols(); k++)
            {
                double a = Arowi[k];
                for(int j = 0; j < Crowi.length; j++)
                    Crowi[j] += a*b.get(k, j);
            }
        }
        
        return result;
    }
    
    public Matrix blockMultiply(Matrix b)
    {
        if(!canMultiply(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        DenseMatrix result = new DenseMatrix(this.rows(), b.cols());
        ///Should choose step size such that 2*stepSize^2 * dataTypeSize <= CacheSize
        int stepSize = 128;//value good for 8mb cache
        
        int iLimit = result.rows();
        int jLimit = result.cols();
        int kLimit = this.cols();
        
        for(int i0 = 0; i0 < iLimit; i0+=stepSize)
            for(int k0 = 0; k0 < kLimit; k0+=stepSize)
                for(int j0 = 0; j0 < jLimit; j0+=stepSize)
                {
                    for(int i = i0; i < min(i0+stepSize, iLimit); i++)
                    {
                        double[] c_row_i = result.matrix[i];
                        
                        for(int k = k0; k < min(k0+stepSize, kLimit); k++)
                        {
                            double a = this.matrix[i][k];
                            
                            for(int j = j0; j < min(j0+stepSize, jLimit); j++)
                                c_row_i[j] += a * b.get(k, j);
                        }
                    }
                }
        
        return result;
    }
    
    private class BlockMultRun implements Runnable
    {
        final CountDownLatch latch;
        final DenseMatrix result;
        final Matrix b;
        final int stepSize;
        final int kLimit, jLimit, iLimit;
        final int i0;

        public BlockMultRun(CountDownLatch latch, DenseMatrix result, Matrix b, int stepSize, int kLimit, int jLimit, int iLimit, int i0)
        {
            this.latch = latch;
            this.result = result;
            this.b = b;
            this.stepSize = stepSize;
            this.kLimit = kLimit;
            this.jLimit = jLimit;
            this.iLimit = iLimit;
            this.i0 = i0;
        }
        
        public void run()
        {
            for(int k0 = 0; k0 < kLimit; k0+=stepSize)
                for(int j0 = 0; j0 < jLimit; j0+=stepSize)
                    for(int i = i0; i < min(i0+stepSize, iLimit); i++)
                    {
                        double[] c_row_i = result.matrix[i];
                        
                        for(int k = k0; k < min(k0+stepSize, kLimit); k++)
                        {
                            double a = matrix[i][k];
                            
                            for(int j = j0; j < min(j0+stepSize, jLimit); j++)
                                c_row_i[j] += a * b.get(k, j);
                        }
                    }
            
            latch.countDown();
        }
        
    }
    
    public Matrix blockMultiply(Matrix b, ExecutorService threadPool)
    {
        if(!canMultiply(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        DenseMatrix result = new DenseMatrix(this.rows(), b.cols());
        
        ///Should choose step size such that 2*stepSize^2 * dataTypeSize <= CacheSize
        int stepSize = 128;//value good for 8mb cache
        
        
        
        int iLimit = result.rows();
        int jLimit = result.cols();
        int kLimit = this.cols();
        
        CountDownLatch latch = new CountDownLatch( (iLimit/stepSize + (iLimit%stepSize > 0 ? 1 : 0)) );
        
        for(int i0 = 0; i0 < iLimit; i0+=stepSize)
            threadPool.submit(new BlockMultRun(latch, result, b, stepSize, kLimit, jLimit, iLimit, i0));
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return result;
    }
    
    @Override
    public Matrix multiply(Matrix b)
    {
        return pureRowOrderMultiply(b);
    }
    
    /**
     * this is a direct conversion of the outer most loop of {@link #multiply(jsat.linear.Matrix) } 
     */
    private class MultRun implements Runnable
    {
        
        final CountDownLatch latch;
        final double[] Arowi;
        final double[] Crowi;
        final Matrix b;

        public MultRun(CountDownLatch latch, double[] Arowi, double[] Crowi, Matrix b)
        {
            this.latch = latch;
            this.Arowi = Arowi;
            this.Crowi = Crowi;
            this.b = b;
        }

        public void run()
        {
                for(int k = 0; k < cols(); k++)
                {
                    double a = Arowi[k];
                    for(int j = 0; j < Crowi.length; j++)
                        Crowi[j] += a*b.get(k, j);
                }
            latch.countDown();
        }
    }

    @Override
    public Matrix multiply(Matrix b, ExecutorService threadPool)
    {
        if(!canMultiply(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        DenseMatrix result = new DenseMatrix(this.rows(), b.cols());
        CountDownLatch cdl = new CountDownLatch(this.rows());
        
        for (int i = 0; i < result.rows(); i++)
        {
            double[] Arowi = this.matrix[i];
            double[] Crowi = result.matrix[i];

            threadPool.submit(new MultRun(cdl, Arowi, Crowi, b));
        }
        try
        {
            cdl.await();
        }
        catch (InterruptedException ex)
        {
            //faulre? Gah - try seriel
            return this.multiply(b);
        }
        
        return result;
    }
    
    @Override
    public void mutableMultiply(double c)
    {
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                matrix[i][j] *= c;
    }
    
    private class MultConstant implements Runnable
    {
        final CountDownLatch latch;
        final double[] row;
        final double c;

        public MultConstant(CountDownLatch latch, double[] row, double c)
        {
            this.latch = latch;
            this.row = row;
            this.c = c;
        }

        public void run()
        {
            for(int j = 0; j < row.length; j++)
                row[j] *= c;
            latch.countDown();
        }
        
    }

    @Override
    public void mutableMultiply(double c, ExecutorService threadPool)
    {
        CountDownLatch latch = new CountDownLatch(matrix.length);
        for(int i = 0; i < matrix.length; i++)
            threadPool.submit(new MultConstant(latch, matrix[i], c));
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    @Override
    public Matrix transpose()
    {
        DenseMatrix transpose = new DenseMatrix(cols(), rows());
        
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                transpose.matrix[j][i] = this.matrix[i][j];
        
        return transpose;
    }
    
    @Override
    public double get(int i, int j)
    {
        return matrix[i][j];
    }

    @Override
    public void set(int i, int j, double value)
    {
        matrix[i][j] = value;
    }

    @Override
    public int rows()
    {
        return matrix.length;
    }

    @Override
    public int cols()
    {
        return matrix[0].length;
    }

    @Override
    public boolean isSparce()
    {
        return false;
    }

    @Override
    public long nnz()
    {
        //In a dense matrix we consider all entries to be non null
        return  ((long) matrix.length )*matrix[0].length;
    }

    @Override
    public Matrix copy()
    {
        DenseMatrix copy = new DenseMatrix(rows(), cols());
        for(int i = 0; i < matrix.length; i++)
            System.arraycopy(matrix[i], 0, copy.matrix[i], 0, matrix[i].length);
        
        return copy;
    }

}
