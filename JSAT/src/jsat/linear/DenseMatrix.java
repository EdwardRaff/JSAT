
package jsat.linear;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;

import jsat.utils.FakeExecutor;
import static java.lang.Math.*;
import static jsat.linear.GenericMatrix.NB2;
import static jsat.utils.SystemInfo.*;

/**
 *
 * @author Edward Raff
 */
public class DenseMatrix extends GenericMatrix
{   

	private static final long serialVersionUID = -3112110093920307822L;
	private double[][] matrix;

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
    
    /**
     * Creates a new matrix of zeros
     * @param rows the number of rows
     * @param cols the number of columns
     */
    public DenseMatrix(int rows, int cols)
    {
        matrix = new double[rows][cols];
    }
    
    /**
     * Creates a new matrix that is a clone of the given matrix. 
     * An error will be throw if the rows of the given matrix 
     * are not all the same size
     * 
     * @param matrix the matrix to clone the values of
     */
    public DenseMatrix(double[][] matrix)
    {
        this.matrix = new double[matrix.length][matrix[0].length];
        for(int i = 0; i < this.matrix.length; i++)
            if(matrix[i].length != this.matrix[i].length)//The matrix we were given better have rows of the same length!
                throw new RuntimeException("Given matrix was not of consistent size (rows have diffrent lengths)");
            else
                System.arraycopy(matrix[i], 0, this.matrix[i], 0, this.matrix[i].length);
    }
    
    /**
     * Creates a new dense matrix that has a copy of all the same values as the 
     * given one
     * @param toCopy the matrix to copy
     */
    public DenseMatrix(Matrix toCopy)
    {
        this(toCopy.rows(), toCopy.cols());
        toCopy.copyTo(this);
    }

    @Override
    protected Matrix getMatrixOfSameType(int rows, int cols)
    {
        return new DenseMatrix(rows, cols);
    }
    
    @Override
    public void mutableAdd(double c, Matrix b)
    {
        if(!sameDimensions(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                this.matrix[i][j] += c*b.get(i, j);
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
            //The Dense construcure does not clone the matrix, it just takes the refernce -making it fast
            DenseVector row = new DenseVector(matrix[i]);
            c.increment(i, row.dot(b)*z);//We use the dot product in this way so that if the incoming matrix is sparce, we can take advantage of save computaitons
        }
    }
    
    @Override
    public void transposeMultiply(double c, Vec b, Vec x)
    {
        if(this.rows() != b.length())
            throw new ArithmeticException("Matrix dimensions do not agree, [" + cols() +"," + rows() + "] x [" + b.length() + ",1]" );
        else if(this.cols() != x.length())
            throw new ArithmeticException("Matrix dimensions do not agree with target vector");
        
        for(int i = 0; i < rows(); i++)//if b was sparce, we want to skip every time b_i = 0
        {
            double b_i = b.get(i);
            if(b_i == 0)//Skip, not quite as good as sparce handeling
                continue;//TODO handle sparce input vector better
            
            double[] A_i = this.matrix[i];
            for(int j = 0; j < cols(); j++)
                x.increment(j, c*b_i*A_i[j]);
        }
    }
    @SuppressWarnings("unused")
    private Matrix blockMultiply(Matrix b)
    {
        if(!canMultiply(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        DenseMatrix result = new DenseMatrix(this.rows(), b.cols());
        ///Should choose step size such that 2*NB2^2 * dataTypeSize <= CacheSize
        
        int iLimit = result.rows();
        int jLimit = result.cols();
        int kLimit = this.cols();
        
        for(int i0 = 0; i0 < iLimit; i0+=NB2)
            for(int k0 = 0; k0 < kLimit; k0+=NB2)
                for(int j0 = 0; j0 < jLimit; j0+=NB2)
                {
                    for(int i = i0; i < min(i0+NB2, iLimit); i++)
                    {
                        double[] c_row_i = result.matrix[i];
                        
                        for(int k = k0; k < min(k0+NB2, kLimit); k++)
                        {
                            double a = this.matrix[i][k];
                            
                            for(int j = j0; j < min(j0+NB2, jLimit); j++)
                                c_row_i[j] += a * b.get(k, j);
                            }
                        }
                    }
        
        return result;
    }

    /**
     * Copies the values from A_k to vk
     * @param k the k+1 index copying will start at
     * @param M how far to copy values
     * @param vk the array to copy into
     * @param A_k the source row of the matrix
     * @param vkNorm the initial value for vkNorm
     * @return vkNorm plus the summation of the squared values for all values copied into vk
     */
    private double initalVKNormCompute(int k, int M, double[] vk, double[] A_k)
    {
        double vkNorm = 0.0;
        for(int i = k+1; i < M; i++)
        {
            vk[i] = A_k[i];
            vkNorm += vk[i]*vk[i];
        }
        return vkNorm;
    }

    private void qrUpdateQ(DenseMatrix Q, int k, double[] vk, double TwoOverBeta)
    {
        //Computing Q

        //We are computing Q' in what we are treating as the column major order, which represents Q in row major order, which is what we want!
        for (int j = 0; j < Q.cols(); j++)
        {
            double[] Q_j = Q.matrix[j];
            double y = 0;//y = vk dot A_j
            for (int i = k; i < Q.cols(); i++)
                y += vk[i] * Q_j[i];

            y *= TwoOverBeta;
            for (int i = k; i < Q.rows(); i++)
            {
                Q_j[i] -= y * vk[i];
            }
        }

    }

    private void qrUpdateR(int k, int N, DenseMatrix A, double[] vk, double TwoOverBeta, int M)
    {
        //First run of loop removed, as it will be setting zeros. More accurate to just set them ourselves
        if(k < N)
        {
            qrUpdateRFirstIteration(A, k, vk, TwoOverBeta, M);
        }
        //The rest of the normal look
        for(int j = k+1; j < N; j++)
        {
            double[] A_j = A.matrix[j];
            double y = 0;//y = vk dot A_j
            for(int i = k; i < A.cols(); i++)
                y += vk[i]*A_j[i];
    
            y *= TwoOverBeta;
            for(int i = k; i < M; i++)
                A_j[i] -= y*vk[i];
        }
    }

    private void qrUpdateRFirstIteration(DenseMatrix A, int k, double[] vk, double TwoOverBeta, int M)
    {
        double[] A_j = A.matrix[k];
        double y = 0;//y = vk dot A_j
        for(int i = k; i < A.cols(); i++)
            y += vk[i]*A_j[i];

        y *= TwoOverBeta;
        A_j[k] -= y*vk[k];
        
        for(int i = k+1; i < M; i++)
            A_j[i] = 0.0;
    }

    @Override
    public void changeSize(int newRows, int newCols)
    {
        if(newRows <= 0)
            throw new ArithmeticException("Matrix must have a positive number of rows");
        if(newCols <= 0)
            throw new ArithmeticException("Matrix must have a positive number of columns");
        final int oldRow = matrix.length;
        //first, did the cols change? That forces a lot of allocation. 
        if(newCols != cols())
        {
            for(int i = 0; i < matrix.length; i++)
                matrix[i] = Arrays.copyOf(matrix[i], newCols);
        }
        //now cols are equal, need to add or remove rows
        matrix = Arrays.copyOf(matrix, newRows);
        for(int i = oldRow; i < newRows; i++)
            matrix[i] = new double[cols()];
    }
    
    private class BlockMultRun implements Runnable
    {
        final CountDownLatch latch;
        final DenseMatrix result;
        final DenseMatrix b;
        final int kLimit, jLimit, iLimit, threadID;
        
        public BlockMultRun(CountDownLatch latch, DenseMatrix result, DenseMatrix b, int threadID)
        {
            this.latch = latch;
            this.result = result;
            this.b = b;
            this.kLimit = cols();
            this.jLimit = result.cols();
            this.iLimit = result.rows();
            this.threadID = threadID;
        }
        
        @Override
        public void run()
        {
            for (int i0 = NB2 * threadID; i0 < iLimit; i0 += NB2 * LogicalCores)
                for (int k0 = 0; k0 < kLimit; k0 += NB2)
                    for (int j0 = 0; j0 < jLimit; j0 += NB2)
                        for (int i = i0; i < min(i0 + NB2, iLimit); i++)
                        {
                            final double[] Ci = result.matrix[i];

                            for (int k = k0; k < min(k0 + NB2, kLimit); k++)
                            {
                                double a = matrix[i][k];
                                double[] Bk = b.matrix[k];

                                for (int j = j0; j < min(j0 + NB2, jLimit); j++)
                                    Ci[j] += a * Bk[j];
                            }
                        }

            latch.countDown();
        }
        
    }
    
    private void blockMultiply(DenseMatrix b, ExecutorService threadPool, DenseMatrix C)
    {
        if(!canMultiply(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        else if(this.rows() != C.rows() || b.cols() != C.cols())
            throw new ArithmeticException("Destination matrix does not match the multiplication dimensions");
        
        CountDownLatch latch = new CountDownLatch(LogicalCores);
        
        for(int threadID = 0; threadID < LogicalCores; threadID++)
            threadPool.submit(new BlockMultRun(latch, C, b, threadID));
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
    public void transposeMultiply(final Matrix b, Matrix C)
    {
        transposeMultiply(b, C, new FakeExecutor());
    }
            
    
    @Override
    public void transposeMultiply(final Matrix b, final Matrix C, ExecutorService threadPool)
    {
        if(this.rows() != b.rows())//Normaly it is A_cols == B_rows, but we are doint A'*B, not A*B
            throw new ArithmeticException("Matrix dimensions do not agree [" + this.cols() + ", " + this.rows()+ "] * [" + b.rows() + ", " + b.cols() + "]");
        else if(this.cols() != C.rows() || b.cols() != C.cols())
            throw new ArithmeticException("Destination matrix does not have matching dimensions");
        final DenseMatrix A = this;
        
        //We only want to take care of the case where everything is of this class. Else let the generic version handle quirks
         if( !(b instanceof DenseMatrix && C instanceof  DenseMatrix) )
         {
             super.transposeMultiply(b, C, threadPool);
             return;
         }
        
        final int iLimit = C.rows();
        final int jLimit = C.cols();
        final int kLimit = this.rows();
        final int blockStep = Math.min(NB2, Math.max(iLimit/LogicalCores, 1));//reduce block size so we can use all cores if needed.
        
        final CountDownLatch cdl = new CountDownLatch(LogicalCores);
        
        for(int threadNum = 0; threadNum < LogicalCores; threadNum++)
        {
            final int threadID = threadNum;
            threadPool.submit(new Runnable() {

                public void run()
                {
                    DenseMatrix BB = (DenseMatrix) b;
                    DenseMatrix CC = (DenseMatrix) C;
                    for (int i0 = blockStep * threadID; i0 < iLimit; i0 += blockStep * LogicalCores)
                        for (int k0 = 0; k0 < kLimit; k0 += blockStep)
                            for (int j0 = 0; j0 < jLimit; j0 += blockStep)
                            {
                                for (int k = k0; k < min(k0 + blockStep, kLimit); k++)
                                {
                                    double[] A_row_k = A.matrix[k];
                                    double[] B_row_k = BB.matrix[k];

                                    for (int i = i0; i < min(i0 + blockStep, iLimit); i++)
                                    {
                                        final double a = A_row_k[i];
                                        final double[] c_row_i = CC.matrix[i];

                                        for (int j = j0; j < min(j0 + blockStep, jLimit); j++)
                                            c_row_i[j] += a * B_row_k[j];
                                    }
                                }
                            }
                    
                    cdl.countDown();
                }
            });
        }
        
        
        try
        {
            cdl.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
    
    @Override
    public void multiply(Matrix b, Matrix C)
    {
        if(!canMultiply(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        else if(this.rows() != C.rows() || b.cols() != C.cols())
            throw new ArithmeticException("Target Matrix is no the correct size");
        
        //We only want to opt the case where everyone is dense, else - let the generic version handle quierks
        if( !(C instanceof DenseMatrix && b instanceof DenseMatrix))
        {
            super.multiply(b, C);
            return;
        }

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


        DenseMatrix result = (DenseMatrix) C;
        DenseMatrix B = (DenseMatrix) b;
        //Pull out the index operations to hand optimize for speed. 
        double[] Arowi;
        double[] Browk;
        double[] Crowi;
        for (int i = 0; i < result.rows(); i++)
        {
            Arowi = this.matrix[i];
            Crowi = result.matrix[i];

            for (int k = 0; k < this.cols(); k++)
            {
                double a = Arowi[k];
                Browk = B.matrix[k];
                for (int j = 0; j < Crowi.length; j++)
                    Crowi[j] += a * Browk[j];
            }
        }

    }
    
    /**
     * this is a direct conversion of the outer most loop of {@link #multiply(jsat.linear.Matrix) } 
     */
    private class MultRun implements Runnable
    {
        
        final CountDownLatch latch;
        final DenseMatrix A;
        final DenseMatrix B, result;
        final int threadID;

        public MultRun(CountDownLatch latch, DenseMatrix A, DenseMatrix result, DenseMatrix B, int threadID)
        {
            this.latch = latch;
            this.A = A;
            this.result = result;
            this.B = B;
            this.threadID = threadID;
        }
        
        public void run()
        {

            //Pull out the index operations to hand optimize for speed. 
            double[] Ai;
            double[] Bi;
            double[] Ci;
            for (int i = 0 + threadID; i < result.rows(); i += LogicalCores)
            {
                Ai = A.matrix[i];
                Ci = result.matrix[i];

                for (int k = 0; k < A.cols(); k++)
                {
                    double a = Ai[k];
                    Bi = B.matrix[k];
                    for (int j = 0; j < Ci.length; j++)
                        Ci[j] += a * Bi[j];
                }
            }

            latch.countDown();
        }
    }

    @Override
    public void multiply(Matrix b, Matrix C, ExecutorService threadPool)
    {
        //We only care when everyone is of this class, else let the generic implementatino handle quirks
        if(!(b instanceof DenseMatrix && C instanceof DenseMatrix))
        {
            super.multiply(b, C, threadPool);
            return;
        }
        
        if(this.rows()/NB2 >= LogicalCores)//Perform block execution only when we have a large enough matrix to keep ever core busy!
        {
            blockMultiply((DenseMatrix)b, threadPool, (DenseMatrix)C);
            return;
        }
        if(!canMultiply(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        else if(this.rows() != C.rows() || b.cols() != C.cols())
            throw new ArithmeticException("Destination matrix does not match the multiplication dimensions");
        CountDownLatch cdl = new CountDownLatch(LogicalCores);
        
        for (int threadID = 0; threadID < LogicalCores; threadID++)
            threadPool.submit(new MultRun(cdl, this, (DenseMatrix)C, (DenseMatrix)b, threadID));
            
        try
        {
            cdl.await();
        }
        catch (InterruptedException ex)
        {
            //faulre? Gah - try seriel
            this.multiply(b, C);
        }
    }
    
    @Override
    public void mutableMultiply(double c)
    {
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                matrix[i][j] *= c;
    }
    
    @Override
    public void mutableTranspose()
    {
       for(int i = 0; i < rows()-1; i++)
            for(int j = i+1; j < cols(); j++)
            {
                double tmp = matrix[j][i];
                matrix[j][i] = matrix[i][j];
                matrix[i][j] = tmp;
            }
    }
    
    @Override
    public DenseMatrix transpose()
    {
        DenseMatrix toReturn = new DenseMatrix(cols(), rows());
        this.transpose(toReturn);
        return toReturn;
    }
        
    @Override
    public void transpose(Matrix C)
    {
        if(this.rows() != C.cols() || this.cols() != C.rows())
            throw new ArithmeticException("Target matrix does not have the correct dimensions");
        
        for (int i0 = 0; i0 < rows(); i0 += NB2)
            for (int j0 = 0; j0 < cols(); j0 += NB2)
                for (int i = i0; i < min(i0+NB2, rows()); i++)
                    for (int j = j0; j < min(j0+NB2, cols()); j++)
                        C.set(j, i, this.get(i, j));
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
    public void swapRows(int r1, int r2)
    {
        if(r1 >= rows() || r2 >= rows())
            throw new ArithmeticException("Can not swap row, matrix is smaller then requested");
        else if(r1 < 0 || r2 < 0)
            throw new ArithmeticException("Can not swap row, there are no negative row indices");
        double[] tmp = matrix[r1];
        matrix[r1] = matrix[r2];
        matrix[r2] = tmp;
    }
    
    @Override
    public void zeroOut()
    {
        for(int i = 0; i < rows(); i++)
            Arrays.fill(matrix[i], 0);
    }

    @Override
    public Vec getRowView(int r)
    {
        return new DenseVector(matrix[r]);
    }
    
    @Override
    public Matrix[] lup()
    {
        Matrix[] lup = new Matrix[3];
        
        Matrix P = eye(rows());
        DenseMatrix L;
        DenseMatrix U = this;
        
        //Initalization is a little wierd b/c we want to handle rectangular cases as well!
        if(rows() > cols())//In this case, we will be changing U before returning it (have to make it smaller, but we can still avoid allocating extra space
            L = new DenseMatrix(rows(), cols());
        else
            L = new DenseMatrix(rows(), rows());        
        
        for(int i = 0; i < U.rows(); i++)
        {
            //If rectangular, we still need to loop through to update ther est of L - even though we wont make many other changes
            if(i < U.cols())
            {
                //Partial pivoting, find the largest value in this colum and move it to the top! 
                //Find the largest magintude value in the colum k, row j
                int largestRow = i;
                double largestVal = Math.abs(U.matrix[i][i]);
                for (int j = i + 1; j < U.rows(); j++)
                {
                    double rowJLeadVal = Math.abs(U.matrix[j][i]);
                    if (rowJLeadVal > largestVal)
                    {
                        largestRow = j;
                        largestVal = rowJLeadVal;
                    }
                }

                //SWAP!
                U.swapRows(largestRow, i);
                P.swapRows(largestRow, i);
                L.swapRows(largestRow, i);
                
                L.matrix[i][i] = 1;
            }   

            //Seting up L 
            for(int k = 0; k < Math.min(i, U.cols()); k++)
            {
                double tmp = U.matrix[i][k]/U.matrix[k][k];
                L.matrix[i][k] = Double.isNaN(tmp) ? 0.0 : tmp;
                U.matrix[i][k] = 0;

                for(int j = k+1; j < U.cols(); j++)
                {
                    U.matrix[i][j] -= L.matrix[i][k]*U.matrix[k][j];
                }
            }
        }
        
        
        if(rows() > cols())//Clean up!
        {
            //We need to change U to a square nxn matrix in this case, we can safely drop the last 2 rows!
            double[][] newU = new double[cols()][];
            System.arraycopy(U.matrix, 0, newU, 0, newU.length);
            U = new DenseMatrix(newU);//We have made U point at a new object, but the array is still pointing at the same rows! 
        }
        
        lup[0] = L;
        lup[1] = U;
        lup[2] = P;
        
        return lup;
    }
    
    private class LUProwRun implements Callable<Integer>
    {
        final DenseMatrix L;
        final DenseMatrix U;
        final int k, threadNumber;
        double largestSeen = Double.MIN_VALUE;
        int largestIndex ;

        public LUProwRun(DenseMatrix L, DenseMatrix U, int k, int threadNumber)
        {
            this.L = L;
            this.U = U;
            this.k = k;
            largestIndex = k+1;
            this.threadNumber = threadNumber;
        }
       
        /**
         * Returns the index of the row with the largest absolute value we ever saw in column k+1
         */
        public Integer call() throws Exception
        {
            for(int i = k+1+threadNumber; i < U.rows(); i+=LogicalCores)
            {
                double tmp = U.matrix[i][k]/U.matrix[k][k];
                L.matrix[i][k] = Double.isNaN(tmp) ? 0.0 : tmp;

                //We perform the first iteration of the loop outside, as we want to cache its value for searching later
                U.matrix[i][k+1] -= L.matrix[i][k]*U.matrix[k][k+1];
                if(Math.abs(U.matrix[i][k+1]) > largestSeen)
                {
                    largestSeen = Math.abs(U.matrix[i][k+1]);
                    largestIndex = i;
                }
                for(int j = k+2; j < U.cols(); j++)
                {
                    U.matrix[i][j] -= L.matrix[i][k]*U.matrix[k][j];
                }
            }
            
            return largestIndex;
        }
        
    }
    
    @Override
    public Matrix[] lup(ExecutorService threadPool)
    {
        Matrix[] lup = new Matrix[3];
        
        Matrix P = eye(rows());
        DenseMatrix L;
        DenseMatrix U = this;
        
        //Initalization is a little wierd b/c we want to handle rectangular cases as well!
        if(rows() > cols())//In this case, we will be changing U before returning it (have to make it smaller, but we can still avoid allocating extra space
            L = new DenseMatrix(rows(), cols());
        else
            L = new DenseMatrix(rows(), rows());
        
        List<Future<Integer>> bigIndecies = new ArrayList<Future<Integer>>(LogicalCores);
        for(int k = 0; k < Math.min(rows(), cols()); k++)
        {
            //Partial pivoting, find the largest value in this colum and move it to the top! 
            //Find the largest magintude value in the colum k, row j
            int largestRow = k;
            double largestVal = Math.abs(U.matrix[k][k]);
            if(bigIndecies.isEmpty())
                for(int j = k+1; j < U.rows(); j++)
                {
                    double rowJLeadVal = Math.abs(U.matrix[j][k]);
                    if(rowJLeadVal > largestVal)
                    {
                        largestRow = j;
                        largestVal = rowJLeadVal;
                    }
                }
            else
            {
                for(Future<Integer> fut : bigIndecies)
                {
                    try
                    {
                        int j = fut.get();
                        double rowJLeadVal = Math.abs(U.matrix[j][k]);
                        if(rowJLeadVal > largestVal)
                        {
                            largestRow = j;
                            largestVal = rowJLeadVal;
                        }
                    }
                    catch (InterruptedException ex)
                    {
                        Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
                    }
                    catch (ExecutionException ex)
                    {
                        Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
                    }
                    
                }
                
                bigIndecies.clear();
            }
            
            //SWAP!
            U.swapRows(largestRow, k);
            P.swapRows(largestRow, k);
            L.swapRows(largestRow, k);
            
            L.matrix[k][k] = 1;
            //Seting up L 
            for(int threadNumber = 0; threadNumber < LogicalCores; threadNumber++)
                bigIndecies.add(threadPool.submit(new LUProwRun(L, U, k, threadNumber)));
        }
        
        
        //Zero out the bottom rows
        for(int k = 0; k < Math.min(rows(), cols()); k++)
            for(int j = 0; j < k; j++)
                U.matrix[k][j] = 0;
        
        
        if(rows() > cols())//Clean up!
        {
            //We need to change U to a square nxn matrix in this case, we can safely drop the last 2 columns!
            double[][] newU = new double[cols()][];
            System.arraycopy(U.matrix, 0, newU, 0, newU.length);
            U = new DenseMatrix(newU);//We have made U point at a new object, but the array is still pointing at the same rows! 
        }
        
        lup[0] = L;
        lup[1] = U;
        lup[2] = P;
        
        return lup;
    }
    
    @Override
    public Matrix[] qr()
    {
        int N = cols(), M  = rows();
        Matrix[] qr = new Matrix[2];
        
        DenseMatrix Q = Matrix.eye(M);
        DenseMatrix A;
        if(isSquare())
        {
            mutableTranspose();
            A = this;
        }
        else
            A = (DenseMatrix) this.transpose();
        int to = cols() > rows() ? M : N;
        double[] vk = new double[M];
        for(int k = 0; k < to; k++)
        {
            double[] A_k = A.matrix[k];
            
            double vkNorm = initalVKNormCompute(k, M, vk, A_k);
            double beta = vkNorm;
            
            double vk_k = vk[k] = A_k[k];//force into register, help the JIT!
            vkNorm += vk_k*vk_k;
            vkNorm = sqrt(vkNorm);
            
            
            double alpha = -signum(vk_k) * vkNorm;
            vk_k  -= alpha;
            vk[k] = vk_k;
            beta += vk_k*vk_k;
            
            
            if(beta == 0)
                continue;
            double TwoOverBeta = 2.0/beta;
            qrUpdateQ(Q, k, vk, TwoOverBeta);
            qrUpdateR(k, N, A, vk, TwoOverBeta, M);
        }
        qr[0] = Q;
        if(isSquare())
        {
            A.mutableTranspose();
            qr[1] = A;
        }
        else
            qr[1] = A.transpose();
        return qr;
    }
    
    private class QRRun implements Runnable
    {
        DenseMatrix A, Q;
        double[] vk;
        double TwoOverBeta;
        int k, threadID, N, M;
        CountDownLatch latch;

        public QRRun(DenseMatrix A, DenseMatrix Q, double[] vk, double TwoOverBeta, int k, int threadID, CountDownLatch latch)
        {
            this.A = A;
            this.Q = Q;
            this.vk = vk;
            this.TwoOverBeta = TwoOverBeta;
            this.k = k;
            this.threadID = threadID;
            this.latch = latch;
            this.N = A.rows();
            this.M = A.cols();
        }
        
        public void run()
        {
            //Computing Q
            {
                //We are computing Q' in what we are treating as the column major order, which represents Q in row major order, which is what we want!
                for(int j = 0+threadID; j < Q.cols(); j+=LogicalCores)
                {
                    double[] Q_j = Q.matrix[j];
                    double y = 0;//y = vk dot A_j
                    for (int i = k; i < Q.cols(); i++)
                        y += vk[i] * Q_j[i];

                    y *= TwoOverBeta;
                    for (int i = k; i < Q.rows(); i++)
                    {
                        Q_j[i] -= y*vk[i];
                    }
                }
            }
            
            //First run of loop removed, as it will be setting zeros. More accurate to just set them ourselves
            if(k < N && threadID == 0)
            {
                qrUpdateRFirstIteration(A, k, vk, TwoOverBeta, M);
            }
            //The rest of the normal look
            for(int j = k+1+threadID; j < N; j+=LogicalCores)
            {
                double[] A_j = A.matrix[j];
                double y = 0;//y = vk dot A_j
                for(int i = k; i < A.cols(); i++)
                    y += vk[i]*A_j[i];
        
                y *= TwoOverBeta;
                for(int i = k; i < M; i++)
                    A_j[i] -= y*vk[i];
            }
            latch.countDown();
        }
        
    }
    
    @Override
    public Matrix[] qr(ExecutorService threadPool)
    {
        int N = cols(), M  = rows();
        Matrix[] qr = new Matrix[2];
        
        DenseMatrix Q = Matrix.eye(M);
        DenseMatrix A;
        if(isSquare())
        {
            mutableTranspose();
            A = this;
        }
        else
            A = (DenseMatrix) this.transpose();
        
        double[] vk = new double[M];
        
        int to = cols() > rows() ? M : N;
        for(int k = 0; k < to; k++)
        {
            double[] A_k = A.matrix[k];
            
            double vkNorm = initalVKNormCompute(k, M, vk, A_k);
            double beta = vkNorm;
            
            double vk_k = vk[k] = A_k[k];
            vkNorm += vk_k*vk_k;
            vkNorm = sqrt(vkNorm);
            
            
            double alpha = -signum(vk_k) * vkNorm;
            vk_k -= alpha;
            beta += vk_k*vk_k;
            vk[k] = vk_k;
            
            
            if(beta == 0)
                continue;
            
            double TwoOverBeta = 2.0/beta;
            
            CountDownLatch latch = new CountDownLatch(LogicalCores);
            for(int threadID = 0; threadID < LogicalCores; threadID++)
                threadPool.submit(new QRRun(A, Q, vk, TwoOverBeta, k, threadID, latch));
            try
            {
                latch.await();
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        qr[0] = Q;
        if(isSquare())
        {
            A.mutableTranspose();
            qr[1] = A;
        }
        else
            qr[1] = A.transpose();
        return qr;
    }
    
    @Override
    public DenseMatrix clone()
    {
        DenseMatrix copy = new DenseMatrix(rows(), cols());
        for(int i = 0; i < matrix.length; i++)
            System.arraycopy(matrix[i], 0, copy.matrix[i], 0, matrix[i].length);
        
        return copy;
    }
    
}
