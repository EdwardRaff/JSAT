
package jsat.linear;

/**
 * A collection of mutable Row and Column operations that can be performed on matrices. 
 * @author Edward Raff
 */
public class RowColumnOps
{
    /**
     * Updates the values along the main diagonal of the matrix by adding a constant to them
     * @param A the matrix to perform the update on
     * @param start the first index of the diagonals to update (inclusive)
     * @param to the last index of the diagonals to update (exclusive)
     * @param c the constant to add to the diagonal
     */
    public static void addDiag(Matrix A, int start, int to, double c)
    {
        for(int i = start; i < to; i++)
            A.increment(i, i, c);
    }
    
    /**
     * Updates the values of row <tt>i</tt> in the given matrix to be A[i,:] = A[i,:]+ c
     * 
     * @param A the matrix to perform he update on
     * @param i the row to update
     * @param start the first index of the row to update from (inclusive)
     * @param to the last index of the row to update (exclusive)
     * @param c the constant to add to each element
     */
    public static void addRow(Matrix A, int i, int start, int to, double c)
    {
        for(int j = start; j < to; j++)
            A.increment(i, j, c);
    }
    
    /**
     * Updates the values of row <tt>i</tt> in the given matrix to be A[i,:] = A[i,:]+ c
     * 
     * @param A the matrix to perform he update on
     * @param i the row to update
     * @param c the constant to add to each element
     */
    public static void addRow(Matrix A, int i, double c)
    {
        addRow(A, i, 0, A.cols(), c);
    }
    
    /**
     * Updates the values of row <tt>i</tt> in the given matrix to be A[i,:] = A[i,:] * c
     * 
     * @param A the matrix to perform he update on
     * @param i the row to update
     * @param start the first index of the row to update from (inclusive)
     * @param to the last index of the row to update (exclusive)
     * @param c the constant to multiply each element by
     */
    public static void multRow(Matrix A, int i, int start, int to, double c)
    {
        for(int j = start; j < to; j++)
            A.set(i, j, A.get(i, j)*c);
    }
    
    /**
     * Updates the values of row <tt>i</tt> in the given matrix to be A[i,:] = A[i,:] * c
     * 
     * @param A the matrix to perform he update on
     * @param i the row to update
     * @param c the constant to multiply each element by
     */
    public static void multRow(Matrix A, int i, double c)
    {
        multRow(A, i, 0, A.cols(), c);
    }
    
    /**
     * Updates the values of row <tt>i</tt> in the given matrix to be A[i,:] = A[i,:] .* c[i]
     * The Matrix <tt>A</tt> and vector <tt>c</tt> do not need to have the same dimensions,
     * so long as they both have indices in the given range. 
     * 
     * @param A the matrix to perform he update on
     * @param i the row to update
     * @param start the first index of the row to update from (inclusive)
     * @param to the last index of the row to update (exclusive)
     * @param c the vector of values to multiple the elements of <tt>A</tt> by
     */
    public static void multRow(Matrix A, int i, int start, int to, Vec c)
    {
        for(int j = start; j < to; j++)
            A.set(i, j, A.get(i, j)*c.get(j));
    }
    
    /**
     * Updates the values of row <tt>i</tt> in the given matrix to be A[i,:] = A[i,:] .* c[i]
     * The Matrix <tt>A</tt> and vector <tt>c</tt> do not need to have the same dimensions,
     * so long as they both have indices in the given range. 
     * 
     * @param A the matrix to perform he update on
     * @param i the row to update
     * @param c the vector of values to multiple the elements of <tt>A</tt> by
     */
    public static void multRow(Matrix A, int i, Vec c)
    {
        if(A.cols() != c.length())
            throw new ArithmeticException("Can not perform row update, length miss match " + A.cols() + " and " + c.length());
        multRow(A, i, 0, c.length(), c);
    }
    
    /**
     * Updates the values of row <tt>i</tt> in the given matrix to be A[i,:] = A[i,:] .* c[i]
     * The Matrix <tt>A</tt> and array <tt>c</tt> do not need to have the same dimensions,
     * so long as they both have indices in the given range. 
     * 
     * @param A the matrix to perform he update on
     * @param i the row to update
     * @param start the first index of the row to update from (inclusive)
     * @param to the last index of the row to update (exclusive)
     * @param c the array of values to multiple the elements of <tt>A</tt> by
     */
    public static void multRow(Matrix A, int i, int start, int to, double[] c)
    {
        for(int j = start; j < to; j++)
            A.set(i, j, A.get(i, j)*c[j]);
    }
    
    /**
     * Updates the values of row <tt>i</tt> in the given matrix to be A[i,:] = A[i,:] .* c[i]
     * 
     * @param A the matrix to perform he update on
     * @param i the row to update 
     * @param c the array of values to multiple the elements of <tt>A</tt> by
     */
    public static void multRow(Matrix A, int i, double[] c)
    {
        if(A.cols() != c.length)
            throw new ArithmeticException("Can not perform row update, length miss match " + A.cols() + " and " + c.length);
        multRow(A, i, 0, c.length, c);
    }
    
    /**
     * Updates the values of row <tt>i</tt> in the given matrix to be A[i,:] = A[i,:] / c
     * 
     * @param A the matrix to perform he update on
     * @param i the row to update
     * @param start the first index of the row to update from (inclusive)
     * @param to the last index of the row to update (exclusive)
     * @param c the constant to divide each element by
     */
    public static void divRow(Matrix A, int i, int start, int to, double c)
    {
        for(int j = start; j < to; j++)
            A.set(i, j, A.get(i, j)/c);
    }
    
    /**
     * Updates the values of row <tt>i</tt> in the given matrix to be A[i,:] = A[i,:] / c
     * 
     * @param A the matrix to perform he update on
     * @param i the row to update
     * @param c the constant to divide each element by
     */
    public static void divRow(Matrix A, int i, double c)
    {
        divRow(A, i, 0, A.cols(), c);
    }
    
    /**
     * Updates the values of column <tt>j</tt> in the given matrix to be A[:,j] = A[:,j]+ c
     * 
     * @param A the matrix to perform he update on
     * @param j the row to update
     * @param start the first index of the row to update from (inclusive)
     * @param to the last index of the row to update (exclusive)
     * @param c the constant to add to each element
     */
    public static void addCol(Matrix A, int j, int start, int to, double c)
    {
        for(int i = start; i < to; i++)
            A.increment(i, j, c);
    }
    
    /**
     * Updates the values of column <tt>j</tt> in the given matrix to be A[:,j] = A[:,j]+ c
     * 
     * @param A the matrix to perform he update on
     * @param j the row to update
     * @param c the constant to add to each element
     */
    public static void addCol(Matrix A, int j, double c)
    {
        addCol(A, j, 0, A.rows(), c);
    }
    
    /**
     * Updates the values of column <tt>j</tt> in the given matrix to be A[:,j] = A[:,j]* c
     * 
     * @param A the matrix to perform he update on
     * @param j the row to update
     * @param start the first index of the row to update from (inclusive)
     * @param to the last index of the row to update (exclusive)
     * @param c the constant to multiply each element by
     */
    public static void multCol(Matrix A, int j, int start, int to, double c)
    {
        for(int i = start; i < to; i++)
            A.set(i, j, A.get(i, j)*c);
    }
    
    /**
     * Updates the values of column <tt>j</tt> in the given matrix to be A[:,j] = A[:,j]* c
     * 
     * @param A the matrix to perform he update on
     * @param j the row to update
     * @param c the constant to multiply each element by
     */
    public static void multCol(Matrix A, int j, double c)
    {
        multCol(A, j, 0, A.rows(), c);
    }
    
    /**
     * Updates the values of column <tt>j</tt> in the given matrix to be A[:,j] = A[:,j]/c
     * 
     * @param A the matrix to perform he update on
     * @param j the row to update
     * @param start the first index of the row to update from (inclusive)
     * @param to the last index of the row to update (exclusive)
     * @param c the constant to divide each element by
     */
    public static void divCol(Matrix A, int j, int start, int to, double c)
    {
        for(int i = start; i < to; i++)
            A.set(i, j, A.get(i, j)/c);
    }
    
    /**
     * Updates the values of column <tt>j</tt> in the given matrix to be A[:,j] = A[:,j]/c
     * 
     * @param A the matrix to perform he update on
     * @param j the row to update
     * @param c the constant to divide each element by
     */
    public static void divCol(Matrix A, int j, double c)
    {
        divCol(A, j, 0, A.rows(), c);
    }
    
    /**
     * Updates the values of column <tt>j</tt> in the given matrix to be A[:,j] = A[:,j]/c[j].<br>
     * The Matrix <tt>A</tt> and vector <tt>c</tt> do not need to have the same dimensions,
     * so long as they both have indices in the given range. 
     * 
     * @param A the matrix to perform he update on
     * @param j the row to update
     * @param start the first index of the row to update from (inclusive)
     * @param to the last index of the row to update (exclusive)
     * @param c the vector of values to pairwise divide the elements of A by 
     */
    public static void divCol(Matrix A, int j, int start, int to, Vec c)
    {
        for(int i = start; i < to; i++)
            A.set(i, j, A.get(i, j)/c.get(i));
    }
    
    /**
     * Updates the values of column <tt>j</tt> in the given matrix to be A[:,j] = A[:,j]/c[j].<br>
     * The Matrix <tt>A</tt> and array <tt>c</tt> do not need to have the same dimensions, so long as they both have indices in the given range. 
     * 
     * @param A the matrix to perform he update on
     * @param j the row to update
     * @param start the first index of the row to update from (inclusive)
     * @param to the last index of the row to update (exclusive)
     * @param c the array of values to pairwise divide the elements of A by 
     */
    public static void divCol(Matrix A, int j, int start, int to, double[] c)
    {
        for(int i = start; i < to; i++)
            A.set(i, j, A.get(i, j)/c[i]);
    }

    /**
     * Updates the values of row <tt>i</tt> in the given matrix to be A[i,:] = A[i,:]+c[:]*<tt>t</tt>.<br>
     * The Matrix <tt>A</tt> and array <tt>c</tt> do not need to have the same dimensions, so long as they both have indices in the given range. 
     * 
     * @param A the matrix to perform he update on
     * @param i the row to update
     * @param start the first index of the column to update from (inclusive)
     * @param to the last index of the column to update (exclusive)
     * @param t the constant to multiply all elements of <tt>c</tt> by
     * @param c the array of values to pairwise multiply by <tt>t</tt> before adding to the elements of A
     */
    public static void addMultRow(Matrix A, int i, int start, int to, double t, double[] c)
    {
        for(int j = start; j < to; j++)
            A.increment(i, j, c[j]*t);
    }
    
    /**
     * Updates the values of row <tt>i</tt> in the given matrix to be A[i,:] = A[i,:]+c[:]*<tt>t</tt>.<br>
     * The Matrix <tt>A</tt> and array <tt>c</tt> do not need to have the same dimensions, so long as they both have indices in the given range. 
     * 
     * @param A the matrix to perform he update on
     * @param i the row to update
     * @param start the first index of the column to update from (inclusive)
     * @param to the last index of the column to update (exclusive)
     * @param t the constant to multiply all elements of <tt>c</tt> by
     * @param c the array of values to pairwise multiply by <tt>t</tt> before adding to the elements of A
     */
    public static void addMultRow(Matrix A, int i, int start, int to, double t, Vec c)
    {
        for(int j = start; j < to; j++)
            A.increment(i, j, c.get(j)*t);
    }
    
    /**
     * Updates the values of column <tt>j</tt> in the given matrix to be A[:,j] = A[:,j]+c[:]*<tt>t</tt>.<br>
     * The Matrix <tt>A</tt> and array <tt>c</tt> do not need to have the same dimensions, so long as they both have indices in the given range. 
     * 
     * @param A the matrix to perform he update on
     * @param j the row to update
     * @param start the first index of the row to update from (inclusive)
     * @param to the last index of the row to update (exclusive)
     * @param t the constant to multiply all elements of <tt>c</tt> by
     * @param c the array of values to pairwise multiply by <tt>t</tt> before adding to the elements of A
     */
    public static void addMultCol(Matrix A, int j, int start, int to, double t, double[] c)
    {
        for(int i = start; i < to; i++)
            A.increment(i, j, c[i]*t);
    }
    
    /**
     * Updates the values of column <tt>j</tt> in the given matrix to be A[:,j] = A[:,j]+c[:]*<tt>t</tt>.<br>
     * The Matrix <tt>A</tt> and vector <tt>c</tt> do not need to have the same dimensions, so long as they both have indices in the given range. 
     * 
     * @param A the matrix to perform he update on
     * @param j the row to update
     * @param start the first index of the row to update from (inclusive)
     * @param to the last index of the row to update (exclusive)
     * @param t the constant to multiply all elements of <tt>c</tt> by
     * @param c the vector of values to pairwise multiply by <tt>t</tt> before adding to the elements of A
     */
    public static void addMultCol(Matrix A, int j, int start, int to, double t, Vec c)
    {
        for(int i = start; i < to; i++)
            A.increment(i, j, c.get(i)*t);
    }
    
    /**
     * Swaps the columns <tt>j</tt> and <tt>k</tt> in the given matrix. 
     * @param A the matrix to perform he update on
     * @param j the first column to swap 
     * @param k the second column to swap 
     * @param start the first row that will be included in the swap (inclusive)
     * @param to the last row to be included in the swap (exclusive)
     */
    public static void swapCol(Matrix A, int j, int k, int start, int to)
    {
        double t;
        for(int i = start; i < to; i++)
        {
            t = A.get(i, j);
            A.set(i, j, A.get(i, k));
            A.set(i, k, t);
        }
    }
    
    /**
     * Swaps the columns <tt>j</tt> and <tt>k</tt> in the given matrix. 
     * @param A the matrix to perform he update on
     * @param j the first column to swap 
     * @param k the second column to swap 
     */
    public static void swapCol(Matrix A, int j, int k)
    {
        swapCol(A, j, k, 0, A.rows());
    }
    
    /**
     * Swaps the rows <tt>j</tt> and <tt>k</tt> in the given matrix. 
     * @param A the matrix to perform he update on
     * @param j the first row to swap 
     * @param k the second row to swap 
     * @param start the first column that will be included in the swap (inclusive)
     * @param to the last column to be included in the swap (exclusive)
     */
    public static void swapRow(Matrix A, int j, int k, int start, int to)
    {
        double t;
        for(int i = start; i < to; i++)
        {
            t = A.get(j, i);
            A.set(j, i, A.get(k, i));
            A.set(k, i, t);
        }
    }
    
    /**
     * Swaps the columns <tt>j</tt> and <tt>k</tt> in the given matrix. 
     * @param A the matrix to perform he update on
     * @param j the first column to swap 
     * @param k the second column to swap 
     */
    public static void swapRow(Matrix A, int j, int k)
    {
        swapCol(A, j, k, 0, A.cols());
    }
    
    /**
     * Fills the values in a row of the matrix
     * @param A the matrix in question
     * @param i the row of the matrix
     * @param from the first column index to fill (inclusive)
     * @param to the last column index to fill (exclusive)
     * @param val the value to fill into the matrix
     */
    public static void fillRow(Matrix A, int i, int from, int to, double val)
    {
        for(int j = from; j < to; j++)
            A.set(i, j, val);
    }
    
    /**
     * Fills the values in a column of the matrix
     * @param A the matrix in question
     * @param j the column of the matrix
     * @param from the first row index to fill (inclusive)
     * @param to the last row index to fill (exclusive)
     * @param val the value to fill into the matrix
     */
    public static void fillCol(Matrix A, int j, int from, int to, double val)
    {
        for(int i = from; i < to; i++)
            A.set(i, j, val);
    }
}
