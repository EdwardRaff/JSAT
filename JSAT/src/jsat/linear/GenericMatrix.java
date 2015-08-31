
package jsat.linear;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import jsat.utils.FakeExecutor;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import static jsat.utils.SystemInfo.*;
import static java.lang.Math.*;

/**
 * This Class provides default implementations of most all functions in row major form. 
 * Only a small portion must be implemented by the extending class
 * 
 * @author Edward Raff
 */
public abstract class GenericMatrix extends Matrix
{

	private static final long serialVersionUID = -8173419025024676713L;
	/**
     * Step size if the computation accesses 2*NB2^2 * dataTypeSize data, 
     * so that the data being worked on fits into the L2 cache
     */
    protected static int NB2 = (int) sqrt(L2CacheSize/(8.0*2.0));
    
    
    /**
     * Creates a new matrix of the same type
     * @param rows the number of rows for the matrix to have
     * @param cols the number of columns for the matrix to have
     * @return the empty all zero new matrix
     */
    abstract protected Matrix getMatrixOfSameType(int rows, int cols);
    
    @Override
    public void mutableAdd(double c, Matrix b)
    {
        if(!sameDimensions(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                increment(i, j, c*b.get(i, j));
    }
    
    @Override
    public void mutableAdd(final double c, final Matrix b, ExecutorService threadPool)
    {
        if(!sameDimensions(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        
        final CountDownLatch latch = new CountDownLatch(LogicalCores);
        
        for(int threadId = 0; threadId < LogicalCores; threadId++)
        {
            final int ID = threadId;
            threadPool.submit(new Runnable() {

                public void run()
                {
                    for(int i = 0+ID; i < rows(); i+=LogicalCores)
                        for(int j = 0; j < cols(); j++)
                            increment(i, j, c*b.get(i, j));
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
            Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    @Override
    public void mutableAdd(double c)
    {
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                increment(i, j, c);
    }
    
    @Override
    public void mutableAdd(final double c, ExecutorService threadPool)
    {
        final CountDownLatch latch = new CountDownLatch(LogicalCores);

        for (int threadId = 0; threadId < LogicalCores; threadId++)
        {
            final int ID = threadId;
            threadPool.submit(new Runnable()
            {

                public void run()
                {
                    for (int i = 0 + ID; i < rows(); i += LogicalCores)
                        for (int j = 0; j < cols(); j++)
                            increment(i, j, c);
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
            Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    @Override
    public void multiply(Vec b, double z, Vec c)
    {
        if(this.cols() != b.length())
            throw new ArithmeticException("Matrix dimensions do not agree, [" + rows() +"," + cols() + "] x [" + b.length() + ",1]" );
        if(this.rows() != c.length())
            throw new ArithmeticException("Target vector dimension does not agree with matrix dimensions. Matrix has " + rows() + " rows but tagert has " + c.length());

        if (b.isSparse())
        {
            for (int i = 0; i < rows(); i++)
            {
                double dot = 0;
                for(IndexValue iv : b)
                    dot += this.get(i, iv.getIndex()) * iv.getValue();
                c.increment(i, dot * z);
            }
        }
        else
        {
            for (int i = 0; i < rows(); i++)
            {
                double dot = 0;
                for (int j = 0; j < cols(); j++)
                    dot += this.get(i, j) * b.get(j);
                c.increment(i, dot * z);
            }
        }
    }

    @Override
    public void multiply(Matrix b, Matrix C)
    {
        if(!canMultiply(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree: [" + this.rows() + ", " + this.cols() + "] * [" + b.rows() + ", " + b.cols() + "]");
        else if (this.rows() != C.rows() || b.cols() != C.cols())
            throw new ArithmeticException("Target Matrix is no the correct size");

        /*
         * In stead of row echelon order (i, j, k), we compue in "pure row oriented",  see
         * Data structures in Java for matrix computations
         * CONCURRENCY AND COMPUTATION: PRACTICE AND EXPERIENCE
         * Concurrency Computat.: Pract. Exper. 2004; 16:799–815 (DOI: 10.1002/cpe.793)
         */

        for (int i = 0; i < C.rows(); i++)
            for (int k = 0; k < this.cols(); k++)
            {
                double a = this.get(i, k);
                for (int j = 0; j < C.cols(); j++)
                    C.increment(i, j, a * b.get(k, j));
            }
    }
    
    @Override
    public void multiplyTranspose(Matrix b, Matrix C)
    {
        if(this.cols() != b.cols())
            throw new ArithmeticException("Matrix dimensions do not agree");
        else if (this.rows() != C.rows() || b.rows() != C.cols())
            throw new ArithmeticException("Target Matrix is no the correct size");

        final int iLimit = this.rows();
        final int jLimit = b.rows();
        final int kLimit = this.cols();

        for (int i0 = 0; i0 < iLimit; i0 += NB2)
            for (int j0 = 0; j0 < jLimit; j0 += NB2)
                for (int k0 = 0; k0 < kLimit; k0 += NB2)
                    for (int i = i0; i < min(i0 + NB2, iLimit); i++)
                        for (int j = j0; j < min(j0 + NB2, jLimit); j++)
                        {
                            double C_ij = 0;
                            for (int k = k0; k < min(k0 + NB2, kLimit); k++)
                                C_ij += this.get(i, k) * b.get(j, k);
                            C.increment(i, j, C_ij);
                        }
    }
    
    @Override
    public void multiplyTranspose(final Matrix b, final Matrix C, ExecutorService threadPool)
    {
        if(this.cols() != b.cols())
            throw new ArithmeticException("Matrix dimensions do not agree");
        else if (this.rows() != C.rows() || b.rows() != C.cols())
            throw new ArithmeticException("Destination matrix does not have matching dimensions");
        final Matrix A = this;
        ///Should choose step size such that 2*NB2^2 * dataTypeSize <= CacheSize
        
        final int iLimit = this.rows();
        final int jLimit = b.rows();
        final int kLimit = this.cols();
        final int blockStep = Math.min(NB2, Math.max(iLimit/LogicalCores, 1));//reduce block size so we can use all cores if needed.
        final CountDownLatch cdl = new CountDownLatch(LogicalCores);
        
        for(int threadNum = 0; threadNum < LogicalCores; threadNum++)
        {
            final int threadID = threadNum;
            threadPool.submit(new Runnable() {

                @Override
                public void run()
                {
                    for (int i0 = blockStep * threadID; i0 < iLimit; i0 += blockStep * LogicalCores)
                        for (int k0 = 0; k0 < kLimit; k0 += blockStep)
                            for (int j0 = 0; j0 < jLimit; j0 += blockStep)
                                for (int i = i0; i < min(i0 + blockStep, iLimit); i++)
                                    for (int j = j0; j < min(j0 + blockStep, jLimit); j++)
                                    {
                                        double C_ij = 0;
                                        for (int k = k0; k < min(k0 + blockStep, kLimit); k++)
                                            C_ij += A.get(i, k) * b.get(j, k);
                                        C.increment(i, j, C_ij);
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
    public void multiply(final Matrix b, final Matrix C, ExecutorService threadPool)
    {
        if(!canMultiply(this, b))
            throw new ArithmeticException("Matrix dimensions do not agree");
        else if(this.rows() != C.rows() || b.cols() != C.cols())
            throw new ArithmeticException("Destination matrix does not match the multiplication dimensions");
        final CountDownLatch cdl = new CountDownLatch(LogicalCores);
        final Matrix A = this;
        
        
        if(this.rows()/NB2 >= LogicalCores)//Perform block execution only when we have a large enough matrix to keep ever core busy!
        {
            final int kLimit = A.cols();
            final int jLimit = C.cols();
            final int iLimit = C.rows();
            for (int threadID = 0; threadID < LogicalCores; threadID++)
            {
                final int ID = threadID;
                threadPool.submit(new Runnable()
                {

                    public void run()
                    {
                        for (int i0 = NB2 * ID; i0 < iLimit; i0 += NB2 * LogicalCores)
                            for (int k0 = 0; k0 < kLimit; k0 += NB2)
                                for (int j0 = 0; j0 < jLimit; j0 += NB2)
                                    for (int i = i0; i < min(i0 + NB2, iLimit); i++)
                                        for (int k = k0; k < min(k0 + NB2, kLimit); k++)
                                        {
                                            double a = A.get(i, k);
                                            for (int j = j0; j < min(j0 + NB2, jLimit); j++)
                                                C.increment(i, j, a * b.get(k, j));
                                        }
                    }
                });
            }
            return;
        }
        //Else, normal
        for (int threadID = 0; threadID < LogicalCores; threadID++)
        {
            final int ID = threadID;
            threadPool.submit(new Runnable()
            {

                public void run()
                {
                    for (int i = 0 + ID; i < C.rows(); i += LogicalCores)
                        for (int k = 0; k < A.cols(); k++)
                        {
                            double a = A.get(i, k);
                            for (int j = 0; j < C.cols(); j++)
                                C.increment(i, j, a * b.get(k, j));
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
            //faulre? Gah - try seriel
            this.multiply(b, C);
        }
    }
    
    @Override
    public void mutableMultiply(double c)
    {
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                set(i, j, get(i, j)*c);
    }
    
    @Override
    public void mutableMultiply(final double c, ExecutorService threadPool)
    {
        final CountDownLatch latch = new CountDownLatch(LogicalCores);
        for(int threadID = 0; threadID < LogicalCores; threadID++)
        {
            final int ID = threadID;
            threadPool.submit(new Runnable() {

                public void run()
                {
                    for(int i = ID; i < rows(); i+=LogicalCores)
                        for(int j = 0; j < cols(); j++)
                            set(i, j, get(i, j)*c);
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
            Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
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
            
            for(int j = 0; j < cols(); j++)
                x.increment(j, c*b_i*this.get(i, j));
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
            throw new ArithmeticException("Matrix dimensions do not agree");
        else if(this.cols() != C.rows() || b.cols() != C.cols())
            throw new ArithmeticException("Destination matrix does not have matching dimensions");
        final Matrix A = this;
        ///Should choose step size such that 2*NB2^2 * dataTypeSize <= CacheSize
        
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
                    for (int i0 = blockStep * threadID; i0 < iLimit; i0 += blockStep * LogicalCores)
                        for (int k0 = 0; k0 < kLimit; k0 += blockStep)
                            for (int j0 = 0; j0 < jLimit; j0 += blockStep)
                                for (int k = k0; k < min(k0 + blockStep, kLimit); k++)
                                    for (int i = i0; i < min(i0 + blockStep, iLimit); i++)
                                    {
                                        double a = A.get(k, i);

                                        for (int j = j0; j < min(j0 + blockStep, jLimit); j++)
                                            C.increment(i, j, a * b.get(k, j));
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
    public void mutableTranspose()
    {
        if (!this.isSquare())
            throw new ArithmeticException("Can only mutable transpose square matrices");
        for (int i = 0; i < rows() - 1; i++)
            for (int j = i + 1; j < cols(); j++)
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
        
        for (int i0 = 0; i0 < rows(); i0 += NB2)
            for (int j0 = 0; j0 < cols(); j0 += NB2)
                for (int i = i0; i < min(i0+NB2, rows()); i++)
                    for (int j = j0; j < min(j0+NB2, cols()); j++)
                        C.set(j, i, this.get(i, j));
    }

    @Override
    public void swapRows(int r1, int r2)
    {
        if(r1 >= rows() || r2 >= rows())
            throw new ArithmeticException("Can not swap row, matrix is smaller then requested");
        else if(r1 < 0 || r2 < 0)
            throw new ArithmeticException("Can not swap row, there are no negative row indices");
        for(int j = 0; j < cols(); j++)
        {
            double tmp = get(r1, j);
            set(r1, j, get(r2, j));
            set(r2, j, tmp);
        }
    }

    @Override
    public void zeroOut()
    {
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                set(i, j, 0);
    }
    
    @Override
    public Matrix[] lup()
    {
        Matrix[] lup = new Matrix[3];
        
        Matrix P = eye(rows());
        Matrix L;
        Matrix U = this;
        
        //Initalization is a little wierd b/c we want to handle rectangular cases as well!
        if(rows() > cols())//In this case, we will be changing U before returning it (have to make it smaller, but we can still avoid allocating extra space
            L = getMatrixOfSameType(rows(), cols());
        else
            L = getMatrixOfSameType(rows(), rows());        
        
        for(int i = 0; i < U.rows(); i++)
        {
            //If rectangular, we still need to loop through to update ther est of L - even though we wont make many other changes
            if(i < U.cols())
            {
                //Partial pivoting, find the largest value in this colum and move it to the top! 
                //Find the largest magintude value in the colum k, row j
                int largestRow = i;
                double largestVal = Math.abs(U.get(i, i));
                for (int j = i + 1; j < U.rows(); j++)
                {
                    double rowJLeadVal = Math.abs(U.get(j, i));
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
                
                L.set(i, i, 1);
            }   

            //Seting up L 
            for(int k = 0; k < Math.min(i, U.cols()); k++)
            {
                double tmp = U.get(i, k)/U.get(k, k); 
                L.set(i, k, (Double.isNaN(tmp) ? 0.0 : tmp) );
                U.set(i, k, 0.0);

                for(int j = k+1; j < U.cols(); j++)
                {
                    U.increment(i, j, -L.get(i, k)*U.get(k, j));
                }
            }
        }
        
        
        if(rows() > cols())//Clean up!
        {
            //We need to change U to a square nxn matrix in this case, we can safely drop the last 2 rows!
            Matrix newU = getMatrixOfSameType(cols(), cols());
            for(int i = 0; i < cols(); i++)
                for(int j = 0; j < cols(); j++)
                    newU.set(i, j, U.get(i, j));
            U = newU;
        }
        
        lup[0] = L;
        lup[1] = U;
        lup[2] = P;
        
        return lup;
    }
    
    @Override
    public Matrix[] lup(ExecutorService threadPool)
    {
        Matrix[] lup = new Matrix[3];
        
        final Matrix P = eye(rows());
        final Matrix L;
        Matrix U = this;
        final Matrix UU = U;
        //Initalization is a little wierd b/c we want to handle rectangular cases as well!
        if(rows() > cols())//In this case, we will be changing U before returning it (have to make it smaller, but we can still avoid allocating extra space
            L = new DenseMatrix(rows(), cols());
        else
            L = new DenseMatrix(rows(), rows());
        try
        {
            List<Future<Integer>> bigIndecies = new ArrayList<Future<Integer>>(LogicalCores);
            for (int k = 0; k < Math.min(rows(), cols()); k++)
            {
                //Partial pivoting, find the largest value in this colum and move it to the top! 
                //Find the largest magintude value in the colum k, row j
                int largestRow = k;
                double largestVal = Math.abs(U.get(k, k));
                if (bigIndecies.isEmpty())
                    for (int j = k + 1; j < U.rows(); j++)
                    {
                        double rowJLeadVal = Math.abs(U.get(j, k));
                        if (rowJLeadVal > largestVal)
                        {
                            largestRow = j;
                            largestVal = rowJLeadVal;
                        }
                    }
                else
                {
                    for (Future<Integer> fut : bigIndecies)
                    {

                        int j = fut.get();
                        if(j < 0)//Can happen if they are all zeros 
                            continue;
                        double rowJLeadVal = Math.abs(U.get(j, k));
                        if (rowJLeadVal > largestVal)
                        {
                            largestRow = j;
                            largestVal = rowJLeadVal;
                        }


                    }

                    bigIndecies.clear();
                }

                //SWAP!
                U.swapRows(largestRow, k);
                P.swapRows(largestRow, k);
                L.swapRows(largestRow, k);

                L.set(k, k, 1.0);
                //Seting up L 
                final int kk = k;

                for (int threadNumber = 0; threadNumber < LogicalCores; threadNumber++)
                {
                    final int threadID = threadNumber;
                    bigIndecies.add(threadPool.submit(new Callable<Integer>() {

                        public Integer call() throws Exception
                        {
                            double largestSeen = 0.0;
                            int largestIndex = -1;
                            for(int i = kk+1+threadID; i < UU.rows(); i+=LogicalCores)
                            {
                                double tmp = UU.get(i, kk)/UU.get(kk, kk); 
                                L.set(i, kk, (Double.isNaN(tmp) ? 0.0 : tmp) );

                                //We perform the first iteration of the loop outside, as we want to cache its value for searching later
                                UU.increment(i, kk+1, -L.get(i, kk)*UU.get(kk, kk+1));
                                if(Math.abs(UU.get(i,kk+1)) > largestSeen)
                                {
                                    largestSeen = Math.abs(UU.get(i,kk+1));
                                    largestIndex = i;
                                }
                                for(int j = kk+2; j < UU.cols(); j++)
                                {
                                    UU.increment(i, j, -L.get(i, kk)*UU.get(kk, j));
                                }
                            }

                            return largestIndex;
                        }
                    }));
                }
            }


            //Zero out the bottom rows
            for (int k = 0; k < Math.min(rows(), cols()); k++)
                for (int j = 0; j < k; j++)
                    U.set(k, j, 0);


            if(rows() > cols())//Clean up!
            {
                //We need to change U to a square nxn matrix in this case, we can safely drop the last 2 rows!
                Matrix newU = getMatrixOfSameType(cols(), cols());
                for(int i = 0; i < cols(); i++)
                    for(int j = 0; j < cols(); j++)
                        newU.set(i, j, U.get(i, j));
                U = newU;
            }

            lup[0] = L;
            lup[1] = U;
            lup[2] = P;

            return lup;
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
        catch (ExecutionException ex)
        {
            Logger.getLogger(DenseMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        throw new RuntimeException("Uncrecoverable Error");
    }
    
    @Override
    public Matrix[] qr()
    {
        int N = cols(), M  = rows();
        Matrix[] qr = new Matrix[2];
        
        Matrix Q = Matrix.eye(M);
        Matrix A;
        if(isSquare())
        {
            mutableTranspose();
            A = this;
        }
        else
            A = this.transpose();
        int to = cols() > rows() ? M : N;
        double[] vk = new double[M];
        for(int k = 0; k < to; k++)
        {
            
            double vkNorm = initalVKNormCompute(k, M, vk, A);
            double beta = vkNorm;
            
            double vk_k = vk[k] = A.get(k, k);//force into register, help the JIT!
            vkNorm += vk_k*vk_k;
            vkNorm = sqrt(vkNorm);
            
            
            double alpha = -signum(vk_k) * vkNorm;
            vk_k  -= alpha;
            vk[k] = vk_k;
            beta += vk_k*vk_k;
            

            if (beta == 0)
                continue;
            double TwoOverBeta = 2.0 / beta;

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

    private void qrUpdateR(int k, int N, Matrix A, double[] vk, double TwoOverBeta, int M)
    {
        //First run of loop removed, as it will be setting zeros. More accurate to just set them ourselves
        if(k < N)
        {
            qrUpdateRInitalLoop(k, A, vk, TwoOverBeta, M);
        }
        //The rest of the normal look
        for(int j = k+1; j < N; j++)
        {
            double y = 0;//y = vk dot A_j
            for(int i = k; i < A.cols(); i++)
                y += vk[i]*A.get(j, i);
    
            y *= TwoOverBeta;
            for(int i = k; i < M; i++)
                A.increment(j, i, -y*vk[i]);
        }
    }

    private void qrUpdateRInitalLoop(int k, Matrix A, double[] vk, double TwoOverBeta, int M)
    {
        double y = 0;//y = vk dot A_j
        for(int i = k; i < A.cols(); i++)
            y += vk[i]*A.get(k, i);

        y *= TwoOverBeta;
        A.increment(k, k, -y*vk[k]);
        
        for(int i = k+1; i < M; i++)
            A.set(k, i, 0.0);
    }

    private void qrUpdateQ(Matrix Q, int k, double[] vk, double TwoOverBeta)
    {
        //We are computing Q' in what we are treating as the column major order, which represents Q in row major order, which is what we want!
        for(int j = 0; j < Q.cols(); j++)
        {
            double y = 0;//y = vk dot A_j
            for (int i = k; i < Q.cols(); i++)
                y += vk[i] * Q.get(j, i);

            y *= TwoOverBeta;
            for (int i = k; i < Q.rows(); i++)
                Q.increment(j, i, -y*vk[i]);
            
        }
    }

    private double initalVKNormCompute(int k, int M, double[] vk, Matrix A)
    {
        double vkNorm = 0.0;
        for(int i = k+1; i < M; i++)
        {
            vk[i] = A.get(k, i);
            vkNorm += vk[i]*vk[i];
        }
        return vkNorm;
    }
    
    @Override
    public Matrix[] qr(ExecutorService threadPool)
    {
        final int N = cols(), M  = rows();
        Matrix[] qr = new Matrix[2];
        
        final Matrix Q = Matrix.eye(M);
        final Matrix A;
        if(isSquare())
        {
            mutableTranspose();
            A = this;
        }
        else
            A = this.transpose();
        
        final double[] vk = new double[M];
        
        int to = cols() > rows() ? M : N;
        for(int k = 0; k < to; k++)
        {
            double vkNorm = initalVKNormCompute(k, M, vk, A);
            double beta = vkNorm;
            
            double vk_k = vk[k] = A.get(k, k);
            vkNorm += vk_k*vk_k;
            vkNorm = sqrt(vkNorm);
            
            
            double alpha = -signum(vk_k) * vkNorm;
            vk_k -= alpha;
            beta += vk_k*vk_k;
            vk[k] = vk_k;
            
            
            if(beta == 0)
                continue;
            
            final double TwoOverBeta = 2.0/beta;
            
            final CountDownLatch latch = new CountDownLatch(LogicalCores);
            for (int ID = 0; ID < LogicalCores; ID++)
            {
                final int threadID = ID;
                final int kk = k;
                threadPool.submit(new Runnable()
                {

                    public void run()
                    {
                        parallelQRUpdateQ();
                        parallelQRUpdateR();
                        latch.countDown();
                    }

                    private void parallelQRUpdateR()
                    {
                        //First run of loop removed, as it will be setting zeros. More accurate to just set them ourselves
                        if (kk < N && threadID == 0)
                        {
                            parallelQRUpdateRFirstIteration();
                        }
                        //The rest of the normal look
                        for (int j = kk + 1 + threadID; j < N; j += LogicalCores)
                        {
                            double y = 0;//y = vk dot A_j
                            for (int i = kk; i < A.cols(); i++)
                                y += vk[i] * A.get(j, i);

                            y *= TwoOverBeta;
                            for (int i = kk; i < M; i++)
                                A.increment(j, i, -y * vk[i]);
                        }
                    }

                    private void parallelQRUpdateRFirstIteration()
                    {
                        double y = 0;//y = vk dot A_j
                        for (int i = kk; i < A.cols(); i++)
                            y += vk[i] * A.get(kk, i);

                        y *= TwoOverBeta;
                        A.increment(kk, kk, -y * vk[kk]);

                        for (int i = kk + 1; i < M; i++)
                            A.set(kk, i, 0.0);
                    }

                    private void parallelQRUpdateQ()
                    {
                        //We are computing Q' in what we are treating as the column major order, which represents Q in row major order, which is what we want!
                        for (int j = 0 + threadID; j < Q.cols(); j += LogicalCores)
                        {
                            double y = 0;//y = vk dot A_j
                            for (int i = kk; i < Q.cols(); i++)
                                y += vk[i] * Q.get(j, i);

                            y *= TwoOverBeta;
                            for (int i = kk; i < Q.rows(); i++)
                                Q.increment(j, i, -y * vk[i]);
                        }
                    }
                });
            }
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
    public Matrix clone()
    {
        Matrix clone = getMatrixOfSameType(rows(), cols());
        clone.mutableAdd(this);
        return clone;
    }
}
