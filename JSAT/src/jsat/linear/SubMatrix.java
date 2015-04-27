
package jsat.linear;

/**
 * This class allows for the selection of an area of a matrix to operate on independently. 
 * Mutable matrix operations perform on this sub matrix, will be visible in the base matrix
 * given. This allows for easy implementation of blocking algorithms, but care must be taken
 * to not create and use two sub matrices that overlap and are altered simultaneously. 
 * 
 * @author Edward Raff
 */
public class SubMatrix extends GenericMatrix
{

	private static final long serialVersionUID = 8842973175562587725L;
	private Matrix baseMatrix;
    private int firstRow, firstColumn, toRow, toCol;

    /**
     * Creates a new matrix that is a sub view of the given base matrix. 
     * @param baseMatrix the base matrix to get a view of
     * @param firstRow the first row to consider as part of the sub matrix, starting from 0 inclusive
     * @param firstColumn the first column to consider as part of the sub matrix, starting from 0 inclusive
     * @param toRow the last row from the base matrix to consider, exclusive 
     * @param toCol the last column from the base matrix to consider, exclusive
     */
    public SubMatrix(Matrix baseMatrix, int firstRow, int firstColumn, int toRow, int toCol)
    {
        this.baseMatrix = baseMatrix;
        if(firstColumn < 0 || firstRow < 0 || toRow < 0 || toCol < 0)
            throw new ArithmeticException("Can not give negative row or column counts");
        else if(toRow == 0 || toCol == 0)
            throw new ArithmeticException("Must give a positive number of rows and columns");
        else if(toRow > baseMatrix.rows() || toCol > baseMatrix.cols())
            throw new ArithmeticException("You can not specify a matrix that goes past the row / column boundry of the base matrix");
        else if(firstRow >= toRow || firstColumn >= toCol)
            throw new ArithmeticException("Illogical bounds given");
        this.firstRow = firstRow;
        this.firstColumn = firstColumn;
        this.toRow = toRow;
        this.toCol = toCol;
        
        //If we are given a SubMatrix, lets adjust to use it directly instead of accesing through layers
        if(baseMatrix instanceof SubMatrix)
        {
            SubMatrix given = (SubMatrix) baseMatrix;
            this.baseMatrix = given.baseMatrix;
            this.firstRow += given.firstRow;
            this.firstColumn += given.firstColumn;
            this.toRow = given.toRow;
            this.toCol = given.toCol;
        }
    }

    /**
     * Returns the matrix that is the base for this sub matrix. 
     * @return the matrix that is the base for this sub matrix. 
     */
    public Matrix getBaseMatrix()
    {
        return baseMatrix;
    }

    /**
     * Returns the row offset used from the base matrix
     * @return the row offset used from the base matrix
     */
    public int getFirstRow()
    {
        return firstRow;
    }

    /**
     * Returns the column offset used from the base matrix
     * @return the column offset used from the base matrix
     */
    public int getFirstColumn()
    {
        return firstColumn;
    }
    
    @Override
    protected Matrix getMatrixOfSameType(int rows, int cols)
    {
        return new DenseMatrix(rows, cols);//Ehhh... all well
    }

    @Override
    public double get(int i, int j)
    {
        //We MUST do a bounds check, as they might go past us but an index that does exist in the base
        if(i >= rows() || j >= cols())
            throw new ArrayIndexOutOfBoundsException("Can not access index [" + i + ", " + j + "] in the matrix of dimension [" + rows() + ", " + cols() + "]");
        return baseMatrix.get(i+firstRow, j+firstColumn);
    }

    @Override
    public void set(int i, int j, double value)
    {
        //We MUST do a bounds check, as they might go past us but an index that does exist in the base
        if(i >= rows() || j >= cols())
            throw new ArrayIndexOutOfBoundsException("Can not access index [" + i + ", " + j + "] in the matrix of dimension [" + rows() + ", " + cols() + "]");
        baseMatrix.set(i+firstRow, j+firstColumn, value);
    }

    @Override
    public Vec getRowView(int r)
    {
        if(r >= rows())
            throw new IndexOutOfBoundsException("Can not access row " + r + " of " + rows() +" by " + cols() + " matrix");
        Vec origVec = baseMatrix.getRowView(r-firstRow);
        
        return new SubVector(firstColumn, toCol-firstColumn, origVec);
    }

    @Override
    public Vec getColumnView(int j)
    {
        if(j >= cols())
            throw new IndexOutOfBoundsException("Can not access column " + j + " of " + rows() +" by " + cols() + " matrix");
        Vec origVec = baseMatrix.getColumnView(j-firstColumn);
        
        return new SubVector(firstRow, toRow-firstRow, origVec);
    }

    @Override
    public int rows()
    {
        return toRow-firstRow;
    }

    @Override
    public int cols()
    {
        return toCol-firstColumn;
    }

    @Override
    public boolean isSparce()
    {
        return baseMatrix.isSparce();
    }

    /**
     * This method alters the size of a matrix, either adding or subtracting
     * rows from the internal structure of the matrix. Every resize call may
     * cause a new allocation internally, and should not be called for excessive
     * changing of a matrix. All added rows/ columns will have values of zero.
     * If a row / column is removed, it is always the bottom/right most row /
     * column removed. Values of the removed rows / columns will be lost.<br>
     * <br>
     * When a SubMatrix attempts to change size, it may alter the values of the 
     * underlying matrix in an unusual way. If the base matrix is large enough 
     * to hold the changed size, then the values in the underlying matrix that 
     * this SubMatrix is expanding to encompass will be set to zero. If the 
     * underlying matrix is not large enough, change size will be called upon it
     * as well.<br>
     * When decreasing the number of rows / columns, the SubMatrix will always 
     * reduce its view - but never shrink the underling matrix that the view is
     * of. 
     * <br>
     * To expand the size of a SubMatrix without zeroing out the new values, 
     * create a new SubMatrix object. 
     *
     * @param newRows the new number of rows, must be positive
     * @param newCols the new number of columns, must be positive.
     */
    @Override
    public void changeSize(int newRows, int newCols)
    {
        if(newRows <= 0)
            throw new ArithmeticException("Matrix must have a positive number of rows");
        if(newCols <= 0)
            throw new ArithmeticException("Matrix must have a positive number of columns");
        //Increase the underlying matrix to the needed size
        int underNewRows = Math.max(newRows+firstRow, baseMatrix.rows());
        int underNewCols = Math.max(newCols+firstColumn, baseMatrix.cols());
        /*
         * Either the size for the base stays the same, or gets increased
         */
        baseMatrix.changeSize(underNewRows, underNewCols);
        //Zero out the values we are expanding to
        if(newRows > rows())
            new SubMatrix(baseMatrix, toRow, firstColumn, firstRow+newRows, firstColumn+newCols).zeroOut();
        if(newCols > cols())
            new SubMatrix(baseMatrix, firstRow, toCol, firstRow+newRows, firstColumn+newCols).zeroOut();
        
        toRow = firstRow+newRows;
        toCol = firstColumn+newCols;
    }
    
}
