package jsat.linear;

import java.io.Serializable;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import static java.lang.Math.*;
import static jsat.linear.RowColumnOps.*;

/**
 * The Singular Value Decomposition (SVD) of a matrix A<sub>m,n </sub> = U<sub>m,n </sub> &Sigma;<sub>n,n </sub> V<sup>T</sup><sub>n,n </sub>, 
 * where S is the diagonal matrix of the singular values sorted in descending order and are all non negative. 
 * <br> The SVD of a matrix has many practical uses, but is expensive to compute. 
 * <br><br>
 * Implementation adapted from the Public Domain work of <a href="http://math.nist.gov/javanumerics/jama/"> JAMA: A Java Matrix Package</a> 
 * <br>
 * <b>NOTE:</b> The current implementation has been revised and is now passing all test cases. 
 * However, it is still being tested. Use with awareness that it used to be bugged.
 * Note left at revision 597
 * 
 * @author Edward Raff
 */
public class SingularValueDecomposition implements Cloneable, Serializable
{

	private static final long serialVersionUID = 1711766946748622002L;
	private Matrix U, V;
    /**
     * Stores the diagonal values of the S matrix, and contains the bidiagonal values of A during initial steps. 
     */
    private double[] s;
    
    /**
     * Creates a new SVD of the matrix {@code A} such that A = U &Sigma; V<sup>T</sup>. The matrix 
     * {@code  A} will be modified and used as temp space when computing the SVD. 
     * @param A the matrix to create the SVD of
     */
    public SingularValueDecomposition(Matrix A)
    {
        this(A, 100);
    }

    /**
     * Creates a new SVD of the matrix {@code A} such that A = U &Sigma; V<sup>T</sup>. The matrix 
     * {@code  A} will be modified and used as temp space when computing the SVD. 
     * @param A the matrix to create the SVD of
     * @param maxIterations the maximum number of iterations to perform per singular value till convergence. 
     */
    public SingularValueDecomposition(Matrix A, int maxIterations)
    {
        //By doing this we get to keep the colum major algo and get row major performance 
        final boolean transposedWord = A.rows() < A.cols();
        Matrix AA = transposedWord ? new TransposeView(A) : A;
        int m = AA.rows();
        int n = AA.cols();

        int nu = min(m, n);
        U = new DenseMatrix(m, nu);
        V = new DenseMatrix(n, n);
        
        s = new double[min(m + 1, n)];
        double[] e = new double[n];
        double[] work = new double[m];

        int nct = min(m - 1, n);
        int nrt = max(0, min(n - 2, m));
        bidiagonalize(nct, nrt, m, AA, n, e, work);

        // Set up the final bidiagonal matrix or order p.

        int p = min(n, m + 1);
        if (nct < n)
            s[nct] = AA.get(nct, nct);
        
        if (m < p)
            s[p - 1] = 0.0;
        
        if (nrt + 1 < p)
            e[nrt] = AA.get(nrt, p - 1);
        
        e[p - 1] = 0.0;

        generateU(nct, nu, m);
        generateV(n, nrt, e, nu);
        mainIterationLoop(p, e, n, m, maxIterations);
        
        if(transposedWord)
        {
            /*
             * A = U S V'
             * A' = V S' U'
             */
            Matrix tmp = V;
            V = U;
            U = tmp;
        }
    }
    
    /**
     * Sets the values for a SVD explicitly. This is not a copy constructor, and
     * will hold the given values. 
     * 
     * @param U the U matrix of an SVD
     * @param V the V matrix of an SVD
     * @param s the singular sorted by magnitude of an SVD
     */
    public SingularValueDecomposition(Matrix U, Matrix V, double[] s)
    {
        this.U = U;
        this.V = V;
        this.s = s;
    }

    private void bidiagonalize(int nct, int nrt, int m, Matrix A, int n, double[] e, double[] work)
    {
        for (int k = 0; k < max(nct, nrt); k++)
        {
            if (k < nct)
            {
                // Compute the transformation for the k-th column and
                // place the k-th diagonal in s[k].
                s[k] = 0;
                for (int i = k; i < m; i++)
                    s[k] = hypot(s[k], A.get(i, k));
                if (s[k] != 0.0)
                {
                    if (A.get(k, k) < 0.0)
                        s[k] = -s[k];
                    divCol(A, k, k, m, s[k]);
                    A.increment(k, k, 1.0);
                }
                s[k] = -s[k];
            }
            
            for (int j = k + 1; j < n; j++)
            {
                if ((k < nct) & (s[k] != 0.0))
                {

                    // Apply the transformation.

                    double t = 0;
                    for (int i = k; i < m; i++)
                        t += A.get(i, k) * A.get(i, j);
                    t = -t / A.get(k, k);
                    
                    for(int i = k; i < m; i++)
                        A.increment(i, j, t*A.get(i, k));
                }

                // Place the k-th row of A into e for the
                // subsequent calculation of the row transformation.

                e[j] = A.get(k, j);
            }
            if (k < nct)
            {
                // Place the transformation in U for subsequent back
                // multiplication.

                for (int i = k; i < m; i++)
                    U.set(i, k, A.get(i, k));
            }

            if (k < nrt)
            {
                superDiagonalCreation(e, k, n, m, work, A);
            }
        }
    }

    private int sLength()
    {
        return min(U.rows(), V.rows());
    }

    private void superDiagonalCreation(double[] e, int k, int n, int m, double[] work, Matrix A)
    {
        // Compute the k-th row transformation and place the
        // k-th super-diagonal in e[k].
        e[k] = 0;
        for (int i = k + 1; i < n; i++)
            e[k] = Math.hypot(e[k], e[i]);
        
        if (e[k] != 0.0)
        {
            if (e[k + 1] < 0.0)
                e[k] = -e[k];
            
            for (int i = k + 1; i < n; i++)
                e[i] /= e[k];
            
            e[k + 1] += 1.0;
        }
        
        e[k] = -e[k];
        if ((k + 1 < m) & (e[k] != 0.0))
        {

            // Apply the transformation.

            Arrays.fill(work, k+1, m, 0.0);
            for (int j = k + 1; j < n; j++)
                for (int i = k + 1; i < m; i++)
                    work[i] += e[j] * A.get(i, j);
                
            
            for (int j = k + 1; j < n; j++)
            {
                double t = -e[j] / e[k + 1];
                addMultCol(A, j, k+1, m, t, work);
            }
        }


        // Place the transformation in V for subsequent
        // back multiplication.

        for (int i = k + 1; i < n; i++)
            V.set(i, k, e[i]);
    }

    private void generateV(int n, int nrt, double[] e, int nu)
    {
        for (int k = n - 1; k >= 0; k--)
        {
            if ((k < nrt) & (e[k] != 0.0))
            {
                for (int j = k + 1; j < nu; j++)
                {
                    double t = 0;
                    for (int i = k + 1; i < n; i++)
                        t += V.get(i, k) * V.get(i, j);
                    t = -t / V.get(k + 1, k);
                    for (int i = k + 1; i < n; i++)
                        V.increment(i, j, t * V.get(i, k));
                }
            }
            for (int i = 0; i < n; i++)
                V.set(i, k, 0.0);
            V.set(k, k, 1.0);
        }
    }

    private void generateU(int nct, int nu, int m)
    {
        for (int j = nct; j < nu; j++)
        {
            for (int i = 0; i < m; i++)
                U.set(i, j, 0.0);
            
            U.set(j, j, 1.0);
        }
        for (int k = nct - 1; k >= 0; k--)
        {
            if (s[k] != 0.0)
            {
                for (int j = k + 1; j < nu; j++)
                {
                    double t = 0;
                    for (int i = k; i < m; i++)
                        t += U.get(i, k) * U.get(i, j);
                    
                    t = -t / U.get(k, k);
                    for (int i = k; i < m; i++)
                        U.increment(i, j, t * U.get(i, k));
                    
                }
                for (int i = k; i < m; i++)
                    U.set(i, k, -U.get(i, k));
                U.set(k, k, 1.0 + U.get(k, k));
                for (int i = 0; i < k - 1; i++)
                    U.set(i, k, 0.0);
            }
            else
            {
                for (int i = 0; i < m; i++)
                    U.set(i, k, 0.0);
                U.set(k, k, 1.0);
            }
        }
    }

    private void mainIterationLoop(int p, double[] e, int n, int m, int maxIterations)
    {
        // Main iteration loop for the singular values.

        int pp = p - 1;
        int iter = 0;
        double eps = pow(2.0, -52.0);
        while (p > 0 && iter < maxIterations)
        {
            int k, kase;

            // This section of the program inspects for
            // negligible elements in the s and e arrays.  On
            // completion the variables kase and k are set as follows.

            // kase = 1     if s(p) and e[k-1] are negligible and k<p
            // kase = 2     if s(k) is negligible and k<p
            // kase = 3     if e[k-1] is negligible, k<p, and
            //              s(k), ..., s(p) are not negligible (qr step).
            // kase = 4     if e(p-1) is negligible (convergence).

            for (k = p - 2; k >= -1; k--)
            {
                if (k == -1)
                {
                    break;
                }
                if (abs(e[k]) <= eps * (abs(s[k]) + abs(s[k + 1])))
                {
                    e[k] = 0.0;
                    break;
                }
            }
            if (k == p - 2)
            {
                kase = 4;
            }
            else
            {
                int ks;
                for (ks = p - 1; ks >= k; ks--)
                {
                    if (ks == k)
                        break;
                    
                    double t = (ks != p ? abs(e[ks]) : 0.)
                            + (ks != k + 1 ? abs(e[ks - 1]) : 0.);
                    if (abs(s[ks]) <= eps * t)
                    {
                        s[ks] = 0.0;
                        break;
                    }
                }
                if (ks == k)
                {
                    kase = 3;
                }
                else if (ks == p - 1)
                {
                    kase = 1;
                }
                else
                {
                    kase = 2;
                    k = ks;
                }
            }
            k++;

            // Perform the task indicated by kase.

            switch (kase)
            {

                // Deflate negligible s(p).

                case 1:
                {
                    case1(e, p, k, n);
                }
                break;

                // Split at negligible s(k).

                case 2:
                {
                    case2(e, k, p, m);
                }
                break;

                case 3:
                {
                    case3QRStep(p, e, k, n, m);
                    iter++;
                }
                break;

                // Convergence.

                case 4:
                {

                    // Make the singular values positive.

                    if (s[k] <= 0.0)
                    {
                        s[k] = (s[k] < 0.0 ? -s[k] : 0.0);

                        multCol(V, k, 0, pp+1, -1);
                    }

                    // Order the singular values.

                    while (k < pp)
                    {
                        if (s[k] >= s[k + 1])
                            break;
                        
                        double t = s[k];
                        s[k] = s[k + 1];
                        s[k + 1] = t;

                        if (k < n - 1)
                            swapCol(V, k, k+1, 0, n);
                        if (k < m - 1)
                            swapCol(U, k, k+1, 0, m);
                        
                        k++;
                    }
                    iter = 0;
                    p--;
                }
                break;
            }

        }
    }

    private void case1(double[] e, int p, int k, int n)
    {
        double f = e[p - 2];
        e[p - 2] = 0.0;
        for (int j = p - 2; j >= k; j--)
        {
            double t = hypot(s[j], f);
            double cs = s[j] / t;
            double sn = f / t;
            s[j] = t;
            if (j != k)
            {
                f = -sn * e[j - 1];
                e[j - 1] = cs * e[j - 1];
            }

            UVCase12Update(V, n, cs, j, sn, p);
        }
    }

    private void case2(double[] e, int k, int p, int m)
    {
        double f = e[k - 1];
        e[k - 1] = 0.0;
        for (int j = k; j < p; j++)
        {
            double t = hypot(s[j], f);
            double cs = s[j] / t;
            double sn = f / t;
            s[j] = t;
            f = -sn * e[j];
            e[j] = cs * e[j];
            UVCase12Update(U, m, cs, j, sn, k);
        }
    }

    private void UVCase12Update(Matrix UV, int m, double cs, int j, double sn, int k)
    {
        double t;
        for (int i = 0; i < m; i++)
        {
            t = cs * UV.get(i, j) + sn * UV.get(i, k - 1);
            UV.set(i, k - 1, -sn * UV.get(i, j) + cs * UV.get(i, k - 1));
            UV.set(i, j, t);
        }
    }
    
    private void case3QRStep(int p, double[] e, int k, int n, int m)
    {
        // Calculate the shift.

        double scale = max(max(max(max(
                abs(s[p - 1]), abs(s[p - 2])), abs(e[p - 2])),
                abs(s[k])), abs(e[k]));
        double sp = s[p - 1] / scale;
        double spm1 = s[p - 2] / scale;
        double epm1 = e[p - 2] / scale;
        double sk = s[k] / scale;
        double ek = e[k] / scale;
        double b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
        double c = (sp * epm1) * (sp * epm1);
        double shift = 0.0;
        if ((b != 0.0) | (c != 0.0))
        {
            shift = sqrt(b * b + c);
            if (b < 0.0)
            {
                shift = -shift;
            }
            shift = c / (b + shift);
        }
        double f = (sk + sp) * (sk - sp) + shift;
        double g = sk * ek;

        // Chase zeros.

        for (int j = k; j < p - 1; j++)
        {
            double t = hypot(f, g);
            double cs = f / t;
            double sn = g / t;
            if (j != k)
                e[j - 1] = t;
            f = cs * s[j] + sn * e[j];
            e[j] = cs * e[j] - sn * s[j];
            g = sn * s[j + 1];
            s[j + 1] = cs * s[j + 1];

            UVCase3Update(V, n, cs, j, sn);

            t = hypot(f, g);
            cs = f / t;
            sn = g / t;
            s[j] = t;
            f = cs * e[j] + sn * s[j + 1];
            s[j + 1] = -sn * e[j] + cs * s[j + 1];
            g = sn * e[j + 1];
            e[j + 1] = cs * e[j + 1];
            if (j < m - 1)
            {
                UVCase3Update(U, m, cs, j, sn);
            }
        }
        e[p - 2] = f;
    }
    
    private void UVCase3Update(Matrix UV, int m, double cs, int j, double sn)
    {
        double t;
        for (int i = 0; i < m; i++)
        {
            t = cs * UV.get(i, j) + sn * UV.get(i, j + 1);
            UV.set(i, j + 1, -sn * UV.get(i, j) + cs * UV.get(i, j + 1));
            UV.set(i, j, t);
        }
    }
    
    /**
     * Returns the backing matrix U of the SVD. Do not alter this matrix. 
     * @return the matrix U of the SVD
     */
    public Matrix getU()
    {
        return U;
    }

    /**
     * Returns the backing matrix V of the SVD. Do not later this matrix.
     * @return the matrix V of the SVD
     */
    public Matrix getV()
    {
        return V;
    }

    /**
     * Returns a copy of the sorted array of the singular values, include the near zero ones. 
     * @return a copy of the sorted array of the singular values, including the near zero ones. 
     */
    public double[] getSingularValues()
    {
        return Arrays.copyOf(s, sLength());
    }
    
    /**
     * Returns the diagonal matrix S such that the SVD product results in the original matrix. The diagonal contains the singular values. 
     * @return a dense diagonal matrix containing the singular values
     */
    public Matrix getS()
    {
        Matrix DS = new DenseMatrix(U.rows(), V.rows());
        for(int i = 0; i < sLength(); i++)
            DS.set(i, i, s[i]);
        return DS;
    }
    
    /**
     * Returns the 2 norm of the matrix, which is the maximal singular value. 
     * @return the 2 norm of the matrix
     */
    public double getNorm2()
    {
        return s[0];
    }
    
    /**
     * Returns the condition number of the matrix. The condition number is a positive measure of the numerical 
     * instability of the matrix. The larger the value, the less stable the matrix. For singular matrices, 
     * the result is {@link Double#POSITIVE_INFINITY}. 
     * @return the condition number of the matrix
     */
    public double getCondition()
    {
        return getNorm2()/s[sLength()-1];
    }
    
    private double getDefaultTolerance()
    {
        return max(U.rows(), V.rows())*(nextUp(getNorm2())-getNorm2());
    }
    
    /**
     * Returns the numerical rank of the matrix. Near zero values will be ignored. 
     * @return the rank of the matrix
     */
    public int getRank()
    {
        return getRank(getDefaultTolerance());
    }
    
    /**
     * Indicates whether or not the input matrix was of full rank, full 
     * rank matrices are more numerically stable. 
     * 
     * @return <tt>true</tt> if the matrix was of full tank
     */
    public boolean isFullRank()
    {
        return getRank() == sLength();
    }
    
    /**
     * Returns the numerical rank of the matrix. Values &lt;= than <tt>tol</tt> will be ignored. 
     * @param tol the cut of for singular values
     * @return the rank of the matrix
     */
    public int getRank(double tol)
    {
        for(int i = 0; i < sLength(); i++)
            if(s[i] <= tol)
                return i;
        return sLength();
    }
    
    /**
     * Returns an array containing the inverse singular values. Near zero values are converted to zero. 
     * @return an array containing the inverse singular values
     */
    public double[] getInverseSingularValues()
    {
        return getInverseSingularValues(getDefaultTolerance());
    }
    /**
     * Returns an array containing the inverse singular values. Values that are &lt;= <tt>tol</tt> are converted to zero. 
     * @param tol the cut of for singular values
     * @return an array containing the inverse singular values
     */
    public double[] getInverseSingularValues(double tol)
    {
        double[] sInv = Arrays.copyOf(s, sLength());
        for(int i = 0; i < sInv.length; i++)
            if(sInv[i] > tol)
                sInv[i] = 1.0/sInv[i];
            else
                sInv[i] = 0;
        return sInv;
    }
    
    /**
     * Returns the Moore–Penrose pseudo inverse of the matrix. The pseudo inverse for a matrix is unique. If a matrix 
     * is non singular, the pseudo inverse is the inverse. 
     * 
     * @return the pseudo inverse of the matrix
     */
    public Matrix getPseudoInverse()
    {
        return getPseudoInverse(getDefaultTolerance());
    }
    
    /**
     * Returns the Moore–Penrose pseudo inverse of the matrix. The pseudo inverse for a matrix is unique. If a matrix 
     * is non singular, the pseudo inverse is the inverse. 
     * 
     * @param tol the tolerance for singular values to ignore 
     * @return the pseudo inverse of the matrix
     */
    public Matrix getPseudoInverse(double tol)
    {
        Matrix UT = U.transpose();
        Matrix.diagMult(DenseVector.toDenseVec(getInverseSingularValues(tol)), UT);
        
        return V.multiply(UT);
    }
    
    /**
     * Computes the pseudo determinant of the matrix, which corresponds to absolute value of 
     * the determinant of the full rank square sub matrix that contains all non zero singular values. 
     * 
     * @return the pseudo determinant. 
     */
    public double getPseudoDet()
    {
        return getPseudoDet(getDefaultTolerance());
    }
    
    /**
     * Computes the pseudo determinant of the matrix, which corresponds to absolute value of 
     * the determinant of the full rank square sub matrix that contains all non singular values &gt; <tt>tol</tt>. 
     * 
     * @param tol the cut of for singular values
     * @return the pseudo determinant
     */
    public double getPseudoDet(double tol)
    {
        double det = 1;
        for (double d : s)
            if (d <= tol)
                break;
            else
                det *= d;

        return det;
    }
    
    /**
     * Computes the absolute value of the determinant for the full matrix. 
     * @return {@link Math#abs(double) abs}(determinant) 
     */
    public double absDet()
    {
        double absDet = 1.0;
        
        for(double d : s)
            absDet *= d;
        
        return absDet;
    }
    
    /**
     * Solves the linear system of equations for A x = b by using the equation<br><code>
     * x = A<sup>-1</sup>  b = V  S<sup>-1</sup>  U<sup>T</sup>  b </code>
     * <br>
     * When A is not full rank, this results in a more numerically stable approximation that minimizes the least squares error. 
     * 
     * @param b the vector to solve for
     * @return the vector that gives the least squares solution to A x = b
     */
    public Vec solve(Vec b)
    {
        Vec x = U.transposeMultiply(1.0, b);
        x.mutablePairwiseMultiply(DenseVector.toDenseVec(getInverseSingularValues()));
        
        return V.multiply(x);
    }
    /**
     * Solves the linear system of equations for A x = B by using the equation<br><code>
     * x = A<sup>-1</sup>  B = V  S<sup>-1</sup>  U<sup>T</sup>  B </code>
     * <br>
     * When A is not full rank, this results in a more numerically stable approximation that minimizes the least squares error. 
     * 
     * @param B the matrix to solve for
     * @return the matrix that gives the least squares solution to A x = B
     */
    public Matrix solve(Matrix B)
    {
        Matrix x = U.transposeMultiply(B);
        Matrix.diagMult(DenseVector.toDenseVec(getInverseSingularValues()), x);
        
        return V.multiply(x);
    }
    
    /**
     * Solves the linear system of equations for A x = B by using the equation<br><code>
     * x = A<sup>-1</sup>  B = V  S<sup>-1</sup>  U<sup>T</sup>  B </code>
     * <br>
     * When A is not full rank, this results in a more numerically stable approximation that minimizes the least squares error. 
     * 
     * @param b the matrix to solve for
     * @param threadpool
     * @return the matrix that gives the least squares solution to A x = B
     */
    public Matrix solve(Matrix b, ExecutorService threadpool)
    {
        Matrix x = U.transposeMultiply(b, threadpool);
        Matrix.diagMult(DenseVector.toDenseVec(getInverseSingularValues()), x);
        
        return V.multiply(x, threadpool);
    }

    @Override
    public SingularValueDecomposition clone() 
    {
        return new SingularValueDecomposition(U.clone(), V.clone(), Arrays.copyOf(s, s.length));
    }
}
