
package jsat.linear;

import java.io.Serializable;
import static java.lang.Math.*;
import java.util.Arrays;
import java.util.Comparator;
import jsat.math.Complex;
import jsat.utils.DoubleList;
import jsat.utils.IndexTable;

/**
 * Class for performing the Eigen Value Decomposition of a matrix. The EVD of a 
 * real matrix may contain complex results. When this occurs, the EVD is less 
 * useful since JSAT only supports real matrices. The 
 * {@link SingularValueDecomposition} of a real matrix is always real, and may 
 * be more useful. 
 * <br><br>
 * Implementation adapted from the Public Domain work
 * of <a href="http://math.nist.gov/javanumerics/jama/"> JAMA: A Java Matrix
 * Package</a> 
 * <br><br> 
 * If A is symmetric, then A = V*D*V' where the eigenvalue
 * matrix D is diagonal and the eigenvector matrix V is orthogonal. V * V' equals the
 * identity matrix. 
 * <br><br> 
 * If A is not symmetric, then the eigenvalue matrix D
 * is block diagonal with the real eigenvalues in 1-by-1 blocks and any complex
 * eigenvalues, lambda + i*mu, in 2-by-2 blocks, [lambda, mu; -mu, lambda]. The
 * columns of V represent the eigenvectors in the sense that A*V = V*D. 
 * The matrix V may be badly conditioned, or even
 * singular, so the validity of the equation A = V*D*inverse(V) depends upon
 * the condition of V. <br>
 * If there are no complex eigen values, which can be checked using 
 * {@link #isComplex() }, then D is a normal diagonal matrix. 
 *
 * @author Edward Raff
 */
public class EigenValueDecomposition implements Serializable
{


	private static final long serialVersionUID = -7169205761148043008L;
	/**
     * Row and column dimension (square matrix).
     *
     * @serial matrix dimension.
     */
    private int n;
    /**
     * Arrays for internal storage of eigenvalues.
     *
     * @serial internal storage of eigenvalues.
     */
    private double[] d, e;
    /**
     * Array for internal storage of eigenvectors.
     *
     * @serial internal storage of eigenvectors.
     */
    private Matrix V;
    /**
     * Array for internal storage of nonsymmetric Hessenberg form.
     *
     * @serial internal storage of nonsymmetric Hessenberg form.
     */
    private Matrix H;
    /**
     * Used to indicate if the result contains complex eigen values
     */
    private boolean complexResult;

    /**
     * Symmetric Householder reduction to tridiagonal form.
     */
    private void tred2()
    {
        for(int j = 0; j < n; j++)
            d[j] = V.get(n-1, j);

        // Householder reduction to tridiagonal form.

        for (int i = n - 1; i > 0; i--)
        {

            // Scale to avoid under/overflow.

            double scale = 0.0;
            double h = 0.0;
            for (int k = 0; k < i; k++)
            {
                scale = scale + abs(d[k]);
            }
            if (scale == 0.0)
            {
                e[i] = d[i - 1];
                for (int j = 0; j < i; j++)
                {
                    d[j] = V.get(i-1, j);
                    V.set(i, j, 0.0);
                    V.set(j, i, 0.0);
                }
            }
            else
            {

                // Generate Householder vector.

                for (int k = 0; k < i; k++)
                {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }
                double f = d[i - 1];
                double g = sqrt(h);
                if (f > 0)
                    g = -g;
                e[i] = scale * g;
                h -= f * g;
                d[i - 1] = f - g;
                Arrays.fill(e, 0, i, 0.0);

                // Apply similarity transformation to remaining columns.

                for (int j = 0; j < i; j++)
                {
                    f = d[j];
                    V.set(j, i, f);
                    g = e[j] + V.get(j, j) * f;
                    for (int k = j + 1; k <= i - 1; k++)
                    {
                        g += V.get(k,j) * d[k];
                        e[k] += V.get(k, j) * f;
                    }
                    e[j] = g;
                }
                f = 0.0;
                for (int j = 0; j < i; j++)
                {
                    e[j] /= h;
                    f += e[j] * d[j];
                }
                double hh = f / (h + h);
                for (int j = 0; j < i; j++)
                {
                    e[j] -= hh * d[j];
                }
                for (int j = 0; j < i; j++)
                {
                    f = d[j];
                    g = e[j];
                    
                    for (int k = j; k <= i - 1; k++)
                    {
                        V.increment(k, j, -(f * e[k] + g * d[k]));
                    }
                    d[j] = V.get(i-1, j);
                    V.set(i, j, 0.0);
                }
            }
            d[i] = h;
        }


        // Accumulate transformations.

        for (int i = 0; i < n - 1; i++)
        {
            V.set(n-1, i, V.get(i, i));
            V.set(i, i, 1.0);
            double h = d[i + 1];
            if (h != 0.0)
            {
                for (int k = 0; k <= i; k++)
                {
                    d[k] = V.get(k, i+1) / h;
                }
                for (int j = 0; j <= i; j++)
                {
                    double g = 0.0;
                    for (int k = 0; k <= i; k++)
                    {
                        g += V.get(k, i+1) * V.get(k, j);
                    }
                    
                    RowColumnOps.addMultCol(V, j, 0, i+1, -g, d);
                }
            }

            RowColumnOps.fillCol(V, i+1, 0, i+1, 0.0);
            
        }
        for (int j = 0; j < n; j++)
        {
            d[j] = V.get(n-1, j);
            V.set(n-1, j, 0.0);
        }
        V.set(n-1, n-1, 1.0);
        e[0] = 0.0;
    }

    /**
     * Symmetric tridiagonal QL algorithm.
     */
    private void tql2()
    {

        //  This is derived from the Algol procedures tql2, by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.

        for (int i = 1; i < n; i++)
        {
            e[i - 1] = e[i];
        }
        e[n - 1] = 0.0;

        double f = 0.0;
        double tst1 = 0.0;
        double eps = pow(2.0, -52.0);
        for (int l = 0; l < n; l++)
        {

            // Find small subdiagonal element

            tst1 = max(tst1, abs(d[l]) + abs(e[l]));
            int m = l;
            while (m < n)
            {
                if (abs(e[m]) <= eps * tst1)
                {
                    break;
                }
                m++;
            }

            // If m == l, d[l] is an eigenvalue,
            // otherwise, iterate.

            if (m > l)
            {
                int iter = 0;
                do
                {
                    iter = iter + 1;  // (Could check iteration count here.)

                    // Compute implicit shift

                    double g = d[l];
                    double p = (d[l + 1] - g) / (2.0 * e[l]);
                    double r = hypot(p, 1.0);
                    if (p < 0)
                    {
                        r = -r;
                    }
                    d[l] = e[l] / (p + r);
                    d[l + 1] = e[l] * (p + r);
                    double dl1 = d[l + 1];
                    double h = g - d[l];
                    for (int i = l + 2; i < n; i++)
                    {
                        d[i] -= h;
                    }
                    f = f + h;

                    // Implicit QL transformation.

                    p = d[m];
                    double c = 1.0;
                    double c2 = c;
                    double c3 = c;
                    double el1 = e[l + 1];
                    double s = 0.0;
                    double s2 = 0.0;
                    for (int i = m - 1; i >= l; i--)
                    {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * e[i];
                        h = c * p;
                        r = hypot(p, e[i]);
                        e[i + 1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);

                        // Accumulate transformation.

                        columnOpTransform(V, 0, n - 1, i, c, s, 1);
                    }
                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;

                    // Check for convergence.

                }
                while (abs(e[l]) > eps * tst1);
            }
            d[l] = d[l] + f;
            e[l] = 0.0;
        }

        // Sort eigenvalues and corresponding vectors.

        for (int i = 0; i < n - 1; i++)
        {
            int k = i;
            double p = d[i];
            for (int j = i + 1; j < n; j++)
            {
                if (d[j] < p)
                {
                    k = j;
                    p = d[j];
                }
            }
            if (k != i)
            {
                d[k] = d[i];
                d[i] = p;
                RowColumnOps.swapCol(V, i, k);
            }
        }
    }

    /**
     * Nonsymmetric reduction to Hessenberg form.
     */
    private void orthes()
    {
        final double[] ort = new double[n];

        //  This is derived from the Algol procedures orthes and ortran,
        //  by Martin and Wilkinson, Handbook for Auto. Comp.,
        //  Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutines in EISPACK.

        int low = 0;
        int high = n - 1;

        for (int m = low + 1; m <= high - 1; m++)
        {

            // Scale column.

            double scale = 0.0;
            for (int i = m; i <= high; i++)
                scale = scale + abs(H.get(i, m-1));

            if (scale != 0.0)
            {

                // Compute Householder transformation.

                double h = 0.0;
                double tmp;
                for (int i = high; i >= m; i--)
                {
                    ort[i] = tmp = H.get(i, m-1) / scale;
                    h += tmp*tmp;
                }
                double g = sqrt(h);
                if ((tmp = ort[m]) > 0)
                    g = -g;
                
                h = h - tmp * g;
                ort[m] = tmp - g;
                orthesApplyHouseholder(m, high, ort, h);
                ort[m] *= scale;
                H.set(m, m-1, scale*g);
            }
        }

        // Accumulate transformations (Algol's ortran).

        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                V.set(i, j, (i == j ? 1.0 : 0.0));
            }
        }
        
        orthesAccumulateTransforamtions(high, low, ort);
    }


    /**
     * Nonsymmetric reduction from Hessenberg to real Schur form.
     */
    private void hqr2()
    {

        //  This is derived from the Algol procedure hqr2,
        //  by Martin and Wilkinson, Handbook for Auto. Comp.,
        //  Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.

        // Initialize

        int nn = this.n;
        int n = nn - 1;
        int low = 0;
        int high = nn - 1;
        double eps = pow(2.0, -52.0);
        double exshift = 0.0;
        double p = 0, q = 0, r = 0, s = 0, z = 0, t, w, x, y;
        /**
         * Output from complex division
         */
        final double[] cr = new double[2];
        double norm = hqr2GetNormStart(nn, low, high);

        // Outer loop over eigenvalue index

        int iter = 0;
        while (n >= low)
        {

            // Look for single small sub-diagonal element

            int l = n;
            while (l > low)
            {
                s = abs(H.get(l-1, l-1)) + abs(H.get(l, l));
                if (s == 0.0)
                {
                    s = norm;
                }
                if (abs(H.get(l, l-1)) < eps * s)
                {
                    break;
                }
                l--;
            }

            // Check for convergence
            // One root found

            if (l == n)
            {
                H.increment(n, n, exshift);
                d[n] = H.get(n, n);
                e[n] = 0.0;
                n--;
                iter = 0;
            }
            else if (l == n - 1) // Two roots found
            {
                hqr2FoundTwoRoots(exshift, n, nn, low, high);
                n = n - 2;
                iter = 0;

                // No convergence yet
            }
            else
            {

                // Form shift
                
                x = H.get(n, n);
                y = 0.0;
                w = 0.0;
                if (l < n)
                {
                    y = H.get(n-1, n-1);
                    w = pow(H.get(n, n-1), 2);
                }

                // Wilkinson's original ad hoc shift

                if (iter == 10)
                {
                    exshift += x;
                    RowColumnOps.addDiag(H, low, n+1, -x);
                    s = abs(H.get(n, n-1)) + abs(H.get(n-1, n-2));
                    x = y = 0.75 * s;
                    w = -0.4375 * s * s;
                }

                // MATLAB's new ad hoc shift

                if (iter == 30)
                {
                    s = (y - x) / 2.0;
                    s = s * s + w;
                    if (s > 0)
                    {
                        s = sqrt(s);
                        if (y < x)
                        {
                            s = -s;
                        }
                        s = x - w / ((y - x) / 2.0 + s);
                        RowColumnOps.addDiag(H, low, n+1, -s);
                        exshift += s;
                        x = y = w = 0.964;
                    }
                }

                iter = iter + 1;   // (Could check iteration count here.)

                // Look for two consecutive small sub-diagonal elements

                int m = n - 2;
                while (m >= l)
                {
                    z = H.get(m, m);
                    r = x - z;
                    s = y - z;
                    p = (r * s - w) / H.get(m+1, m) + H.get(m, m+1);
                    q = H.get(m+1, m+1) - z - r - s;
                    r = H.get(m+2, m+1);
                    s = abs(p) + abs(q) + abs(r);
                    p = p / s;
                    q = q / s;
                    r = r / s;
                    if (m == l)
                    {
                        break;
                    }
                    if (abs(H.get(m, m-1)) * (abs(q) + abs(r))
                            < eps * (abs(p) * (abs(H.get(m-1, m-1)) + abs(z)
                            + abs(H.get(m+1, m+1)))))
                    {
                        break;
                    }
                    m--;
                }

                for (int i = m + 2; i <= n; i++)
                {
                    H.set(i, i-2, 0.0);
                    if (i > m + 2)
                    {
                        H.set(i, i-3, 0.0);
                    }
                }

                // Double QR step involving rows l:n and columns m:n

                for (int k = m; k <= n - 1; k++)
                {
                    boolean notlast = (k != n - 1);
                    if (k != m)
                    {
                        p = H.get(k, k-1);
                        q = H.get(k+1, k-1);
                        r = (notlast ? H.get(k+2, k-1) : 0.0);
                        x = abs(p) + abs(q) + abs(r);
                        if (x != 0.0)
                        {
                            p = p / x;
                            q = q / x;
                            r = r / x;
                        }
                    }
                    
                    if (x == 0.0)
                        break;
                    
                    s = sqrt(p * p + q * q + r * r);
                    if (p < 0)
                    {
                        s = -s;
                    }
                    if (s != 0)
                    {
                        if (k != m)
                        {
                            H.set(k, k-1, -s*x);
                        }
                        else if (l != m)
                        {
                            H.set(k, k-1, -H.get(k, k-1));
                        }
                        p = p + s;
                        x = p / s;
                        y = q / s;
                        z = r / s;
                        q = q / p;
                        r = r / p;

                        // Row modification
                        rowOpTransform2(H, k, nn - 1, x, k, y, notlast, z, r, q);

                        // Column modification
                        columnOpTransform2(H, 0, min(n, k + 3), x, k, y, notlast, z, r, q);

                        // Accumulate transformations
                        columnOpTransform2(V, low, high, x, k, y, notlast, z, r, q);
                        
                    }  // (s != 0)
                }  // k loop
            }  // check convergence
        }  // while (n >= low)
        
        // Backsubstitute to find vectors of upper triangular form

        if (norm == 0.0)
            return;
        
        backsubtituteFindVectors(nn, z, s, eps, norm, cr);
        
        // Vectors of isolated roots

        for (int i = 0; i < nn; i++)
            if (i < low | i > high)
            {
                for(int j = i; j < nn-1; j++)
                    H.set(i, j, V.get(i, j));
            }
        backtransform(nn, low, high);
    }
    
    /**
     * Creates a new new Eigen Value Decomposition. The input matrix will not be
     * altered. If the input is symmetric, a more efficient algorithm will be
     * used. 
     * 
     * @param A the square matrix to work on.
     */
    public EigenValueDecomposition(Matrix A)
    {
        this(A, 1e-15);
    }
    
    /**
     * Creates a new new Eigen Value Decomposition. The input matrix will not be
     * altered. If the input is symmetric, a more efficient algorithm will be
     * used. 
     * 
     * @param A the square matrix to work on.
     * @param eps the numerical tolerance for differences in value to be 
     * considered the same. 
     */
    public EigenValueDecomposition(Matrix A, double eps)
    {
        if (!A.isSquare())
            throw new ArithmeticException("");
        n = A.cols();
        d = new double[n];
        e = new double[n];

        if (Matrix.isSymmetric(A, eps) )
        {
            //Would give it the transpose, but the input is symmetric. So its the same thing
            Matrix VWork = A.clone();
            V = new TransposeView(VWork);

            // Tridiagonalize.
            tred2();
            
            // Diagonalize.
            tql2();
            V = VWork.transpose();//Place back
            complexResult = false;

        }
        else
        {
            Matrix HWork = A.transpose();
            H = new TransposeView(HWork);
            Matrix VWork = new DenseMatrix(n, n);
            V = new TransposeView(VWork);

            // Reduce to Hessenberg form.
            orthes();

            // Reduce Hessenberg to real Schur form.
            hqr2();
            
            complexResult = false;
            //Check if the result has complex eigen values
            for (int i = 0; i < n; i++)
                if (e[i] != 0)
                    complexResult = true;
            V = VWork.transpose();
        }
    }
    
    /**
     * Sorts the eigen values and the corresponding eigenvector columns by the 
     * associated eigen value. Sorting can not occur if complex values are 
     * present. 
     * @param cmp the comparator to use to sort the eigen values
     */
    public void sortByEigenValue(Comparator<Double> cmp)
    {
        if(isComplex())
            throw new ArithmeticException("Eigen values can not be sorted due to complex results");
        IndexTable it = new IndexTable(DoubleList.unmodifiableView(d, d.length), cmp);
        
        for(int i = 0; i < d.length; i++)
        {
            RowColumnOps.swapCol(V, i, it.index(i));
            double tmp = d[i];
            d[i] = d[it.index(i)];
            d[it.index(i)] = tmp;
            
            it.swap(i, it.index(i));
        }
        
    }


    /**
     * Return a copy of the eigenvector matrix
     *
     * @return the eigen vector matrix
     */
    public Matrix getV()
    {
        return V.clone();
    }
    
    /**
     * Returns the raw eigenvector matrix. Modifying this matrix will effect 
     * others using the same matrix. 
     * @return the eigen vector matrix
     */
    public Matrix getVRaw()
    {
        return V;
    }
    
    /**
     * Returns a copy of the transposed eigenvector matrix. 
     * @return the transposed eigen the eigen vector matrix
     */
    public Matrix getVT() {
        return V.transpose();
    }

    /**
     * Return the real parts of the eigenvalues
     *
     * @return real(diag(D))
     */
    public double[] getRealEigenvalues()
    {
        return d;
    }

    /**
     * Return the imaginary parts of the eigenvalues
     *
     * @return imag(diag(D))
     */
    public double[] getImagEigenvalues()
    {
        return e;
    }

    /**
     * Updates the columns of the matrix M such that <br><br>
     * <code><br>
     * for (int i = low; i <= high; i++)<br>
     * {<br>
     * &nbsp;&nbsp; z = M[i][n+shift];<br>
     * &nbsp;&nbsp; M[i][n+shift] = q * z + p * M[i][n];<br>
     * &nbsp;&nbsp; M[i][n] = q * M[i][n] - p * z;<br>
     * }<br>
     * </code>
     *
     * @param M the matrix to alter
     * @param low the starting column (inclusive)
     * @param high the ending column (inclusive)
     * @param n the column to alter, and the preceding column will be altered as
     * well
     * @param q first constant
     * @param p second constant
     * @param shift the direction to perform the computation. Either 1 for after
     * the current column, or -1 for before the current column.
     */
    private static void columnOpTransform(Matrix M, int low, int high, int n, double q, double p, int shift)
    {
        double z;
        for (int i = low; i <= high; i++)
        {
            z = M.get(i, n+shift);
            M.set(i, n+shift,  q * z + p * M.get(i, n));
            M.set(i, n,        q * M.get(i, n) - p * z);
        }
    }
    
    /**
     * Updates the rows of the matrix M such that 
     * <br>
     * M[n-1][j] = q * M[n-1][j] + p * M[n][j] <br>
     * simultaneously altering <br>
     * M[n][j] = q * M[n][j] - p * M[n-1][j] <br>
     * as if M[n-1][j] had not been altered
     * 
     * @param M the matrix to alter
     * @param low the starting row (inclusive)
     * @param high the ending row (inclusive)
     * @param n the row to alter, and the preceding row will be altered as well
     * @param q the first constant
     * @param p the second constant
     */
    private static void rowOpTransform(Matrix M, int low, int high, int n, double q, double p)
    {
        double z;
        for (int j = low; j <= high; j++)
        {
            z = M.get(n-1, j);
            M.set(n - 1, j,  q * z + p * M.get(n, j));
            M.set(n, j,      q * M.get(n, j) - p * z);
        }
    }

    /**
     * Alters the columns accordin to <br>
     * <code><p>
     * for (int i = low; i <= high; i++)<br>
     * {<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;      p = x * M[i][k] + y * M[i][k + 1];<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;      if (notlast)<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;      {<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          p = p + z * M[i][k + 2];<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          M[i][k + 2] = M[i][k + 2] - p * r;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;      }<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;      M[i][k] = M[i][k] - p;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;      M[i][k + 1] = M[i][k + 1] - p * q;<br>
     *   }<br>
     * </p></code>
     * 
     * 
     * @param M the matrix to alter
     * @param low the starting column (inclusive)
     * @param high the ending column (inclusive)
     * @param x first constant
     * @param k this column and the column after will be altered
     * @param y second constant
     * @param notlast <tt>true<tt> if the 2nd column after <tt>k</tt> should be updated
     * @param z third constant
     * @param r fourth constant
     * @param q fifth constant
     */
    private void columnOpTransform2(Matrix M, int low, int high, double x, int k, double y, boolean notlast, double z, double r, double q)
    {
        double p;
        for (int i = low; i <= high; i++)
        {
            p = x * M.get(i, k) + y * M.get(i, k+1);
            if (notlast)
            {
                p = p + z * M.get(i, k+2);
                M.set(i, k + 2,  M.get(i, k+2) - p * r);
            }
            M.increment(i, k,   -p);
            M.increment(i, k+1, -p*q);
        }
    }
    
    /**
     * Alters the rows of the matrix M according to
     * <code><br>
     * for (int j = low; j <= high; j++)
     * {<br>
     * &nbsp;&nbsp; p = M[k][j] + q * M[k + 1][j];<br>
     * &nbsp;&nbsp; if (notlast)<br>
     * &nbsp;&nbsp; {<br>
     * &nbsp;&nbsp;&nbsp;&nbsp; p = p + r * M[k + 2][j];<br>
     * &nbsp;&nbsp;&nbsp;&nbsp; M[k + 2][j] = M[k + 2][j] - p * z;<br>
     * &nbsp;&nbsp; }<br>
     * &nbsp;&nbsp; M[k][j] = M[k][j] - p * x;<br>
     * &nbsp;&nbsp; M[k + 1][j] = M[k + 1][j] - p * y;<br>
     * }
     * </code>
     * @param M the matrix to alter
     * @param low the starting column (inclusive)
     * @param high the ending column (inclusive)
     * @param x first constant
     * @param k this row and the row after will be altered
     * @param y second constant
     * @param notlast <tt>true<tt> if the 2nd row after <tt>k</tt> should be updated
     * @param z third constant
     * @param r fourth constant
     * @param q fifth constant
     */
    private void rowOpTransform2(Matrix M, int low, int high, double x, int k, double y, boolean notlast, double z, double r, double q)
    {
        double p;
        for (int j = low; j <= high; j++)
        {
            p = M.get(k, j) + q * M.get(k + 1,j);
            if (notlast)
            {
                p = p + r * M.get(k + 2,j);
                M.set(k + 2,j,  M.get(k+2, j) - p * z);
            }
            M.increment(k,   j, -p*x);
            M.increment(k+1, j, -p*y);
        }
    }

    /**
     * Return the block diagonal eigenvalue matrix
     *
     * @return D
     */
    public Matrix getD()
    {
        Matrix X = new DenseMatrix(n, n);
        for (int i = 0; i < n; i++)
        {
            X.set(i, i, d[i]);
            
            if (e[i] > 0)
                X.set(i, i+1, e[i]);
            else if (e[i] < 0)
                X.set(i, i-1, e[i]);
        }
        return X;
    }
    
    /**
     * Indicates wether or not the EVD contains complex eigen values. Because 
     * EVD works with real matrices, the complex eigen vectors are lost - and 
     * the complex eigen values are in the off diagonal spaces of the D matrix.  
     * 
     * @return <tt>true</tt> if the EVD results in complex eigen values. 
     */
    public boolean isComplex()
    {
        return complexResult;
    }

    private void hqr2SolveComplexEigenEquation(final int i, final double p, 
                                               final double q, final double eps,
                                               final double norm, final double w,
                                               final double z, final double r, 
                                               final double ra, final double sa,
                                               final double s, final double[] cr,
                                               final int n)
    {
        double x;
        double y;
        double vr;
        double vi;
        // Solve complex equations
        x = H.get(i, i+1);
        y = H.get(i+1, i);
        vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q * q;
        vi = (d[i] - p) * 2.0 * q;
        if (vr == 0.0 & vi == 0.0)
        {
            vr = eps * norm * (abs(w) + abs(q)
                    + abs(x) + abs(y) + abs(z));
        }
        Complex.cDiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi, cr);
        H.set(i, n-1, cr[0]);
        H.set(i, n,   cr[1]);
        if (abs(x) > (abs(z) + abs(q)))
        {
            H.set(i+1, n-1, (-ra - w * H.get(i, n-1) + q * H.get(i, n)) / x);
            H.set(i+1, n,   (-sa - w * H.get(i, n) - q * H.get(i, n-1)) / x);
        }
        else
        {
            Complex.cDiv(-r - y * H.get(i, n-1), -s - y * H.get(i, n), z, q, cr);
            H.set(i+1, n-1, cr[0]);
            H.set(i+1, n,   cr[1]);
        }
    }

    private void backsubtituteFindVectors(int nn, double z, double s, double eps, double norm, final double[] cr)
    {
        double p;
        double q;
        double w;
        double r = 0;
        double x;
        double y;
        double t;
        for (int n = nn - 1; n >= 0; n--)
        {
            p = d[n];
            q = e[n];

            // Real vector

            if (q == 0)
            {
                int l = n;
                H.set(n, n, 1.0);
                for (int i = n - 1; i >= 0; i--)
                {
                    w = H.get(i, i) - p;
                    r = 0.0;
                    for (int j = l; j <= n; j++)
                    {
                        r = r + H.get(i, j) * H.get(j, n);
                    }
                    if (e[i] < 0.0)
                    {
                        z = w;
                        s = r;
                    }
                    else
                    {
                        l = i;
                        if (e[i] == 0.0)
                        {
                            if (w != 0.0)
                            {
                                H.set(i, n, -r / w);
                            }
                            else
                            {
                                H.set(i, n, -r/(eps*norm));
                            }

                            // Solve real equations

                        }
                        else
                        {
                            x = H.get(i, i+1);
                            y = H.get(i+1, i);
                            q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
                            t = (x * s - z * r) / q;
                            H.set(i, n, t);
                            if (abs(x) > abs(z))
                            {
                                H.set(i+1, n, (-r-w*t)/x);
                            }
                            else
                            {
                                H.set(i+1, n, (-s - y * t) / z);
                            }
                        }

                        // Overflow control

                        t = abs(H.get(i, n));
                        if ((eps * t) * t > 1)
                        {
                            RowColumnOps.divCol(H, n, t);
                        }
                    }
                }

                // Complex vector

            }
            else if (q < 0)
            {
                int l = n - 1;

                // Last vector component imaginary so matrix is triangular

                if (abs(H.get(n, n-1)) > abs(H.get(n-1, n)))
                {
                    H.set(n-1, n-1,  q / H.get(n, n-1));
                    H.set(n-1, n,    -(H.get(n, n) - p) / H.get(n, n-1));
                }
                else
                {
                    Complex.cDiv(0.0, -H.get(n-1, n), H.get(n-1, n-1) - p, q, cr);
                    H.set(n-1, n-1,  cr[0]);
                    H.set(n-1, n, cr[1]);
                }
                H.set(n, n-1, 0.0);
                H.set(n, n, 1.0);
                for (int i = n - 2; i >= 0; i--)
                {
                    double ra, sa, vr, vi;
                    ra = 0.0;
                    sa = 0.0;
                    for (int j = l; j <= n; j++)
                    {
                        ra = ra + H.get(i, j) * H.get(j, n-1);
                        sa = sa + H.get(i, j) * H.get(j, n);
                    }
                    w = H.get(i, i) - p;

                    if (e[i] < 0.0)
                    {
                        z = w;
                        r = ra;
                        s = sa;
                    }
                    else
                    {
                        l = i;
                        if (e[i] == 0)
                        {
                            Complex.cDiv(-ra, -sa, w, q, cr);
                            H.set(i, n-1, cr[0]);
                            H.set(i, n,   cr[1]);
                        }
                        else
                        {
                            hqr2SolveComplexEigenEquation(i, p, q, eps, norm, w, z, r, ra, sa, s, cr, n);
                        }

                        // Overflow control

                        t = max(abs(H.get(i, n-1)), abs(H.get(i, n)));
                        if ((eps * t) * t > 1)
                        {
                            RowColumnOps.multCol(H, n-1, i, n+1, (1/t));
                            RowColumnOps.multCol(H, n  , i, n+1, (1/t));
                        }
                    }
                }
            }
        }
    }

    private double hqr2GetNormStart(int nn, int low, int high)
    {
        // Store roots isolated by balanc and compute matrix norm
        double norm = 0.0;
        for (int i = 0; i < nn; i++)
        {
            if (i < low | i > high)
            {
                d[i] = H.get(i, i);
                e[i] = 0.0;
            }
            for (int j = max(i - 1, 0); j < nn; j++)
            {
                norm = norm + abs(H.get(i, j));
            }
        }
        return norm;
    }

    private void backtransform(int nn, int low, int high)
    {
        double z;
        // Back transformation to get eigenvectors of original matrix
        for (int j = nn - 1; j >= low; j--)
        {
            for (int i = low; i <= high; i++)
            {
                z = 0.0;
                for (int k = low; k <= min(j, high); k++)
                {
                    z = z + V.get(i, k) * H.get(k, j);
                }
                V.set(i, j, z);
            }
        }
    }

    private void hqr2FoundTwoRoots(double exshift, int n, int nn, int low, int high)
    {
        double w, p, q, z, x, s, r;
        w = H.get(n, n - 1) * H.get(n - 1, n);
        p = (H.get(n - 1, n - 1) - H.get(n, n)) / 2.0;
        q = p * p + w;
        z = sqrt(abs(q));
        H.increment(n, n, exshift);
        H.increment(n - 1, n - 1, exshift);
        x = H.get(n, n);

        // Real pair

        if (q >= 0)
        {
            if (p >= 0)
                z = p + z;
            else
                z = p - z;

            d[n - 1] = x + z;
            d[n] = d[n - 1];
            if (z != 0.0)
                d[n] = x - w / z;

            e[n - 1] = 0.0;
            e[n] = 0.0;
            x = H.get(n, n - 1);
            s = abs(x) + abs(z);
            p = x / s;
            q = z / s;
            r = sqrt(p * p + q * q);
            p = p / r;
            q = q / r;

            // Row modification
            rowOpTransform(H, n - 1, nn - 1, n, q, p);

            // Column modification
            columnOpTransform(H, 0, n, n, q, p, -1);

            // Accumulate transformations
            columnOpTransform(V, low, high, n, q, p, -1);

        }
        else // Complex pair
        {
            d[n - 1] = x + p;
            d[n] = x + p;
            e[n - 1] = z;
            e[n] = -z;
        }
    }

    private void orthesAccumulateTransforamtions(int high, int low, final double[] ort)
    {
        for (int m = high - 1; m >= low + 1; m--)
        {
            if (H.get(m, m-1) != 0.0)
            {
                for (int i = m + 1; i <= high; i++)
                {
                    ort[i] = H.get(i, m-1);
                }
                for (int j = m; j <= high; j++)
                {
                    double g = 0.0;
                    for (int i = m; i <= high; i++)
                    {
                        g += ort[i] * V.get(i, j);
                    }
                    // Double division avoids possible underflow
                    g = (g / ort[m]) / H.get(m, m-1);
                    RowColumnOps.addMultCol(V, j, m, high+1, g, ort);
                }
            }
        }
    }

    private void orthesApplyHouseholder(int m, int high, final double[] ort, double h)
    {
        // Apply Householder similarity transformation
        // H = (I-u*u'/h)*H*(I-u*u')/h)

        for (int j = m; j < n; j++)
        {
            double f = 0.0;
            for(int i = m; i <= high; i++)
            {
                f += ort[i] * H.get(i, j);
            }
            f /= h;
            RowColumnOps.addMultCol(H, j, m, high+1, -f, ort);
        }

        for (int i = 0; i <= high; i++)
        {
            double f = 0.0;
            for(int j = m; j <= high; j++)
            {
                f += ort[j] * H.get(i, j);
            }
            f/= h;
            RowColumnOps.addMultRow(H, i, m, high+1, -f, ort);
        }
    }
}
