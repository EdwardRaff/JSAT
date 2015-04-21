
package jsat.linear;

/**
 * This class provides a free view of the transpose of a matrix. This is done by accessing 
 * the matrix elements in swapped order. This has a serious performance impact. If the base 
 * matrix storage is row major order, then accessing the TransposeView in column major order 
 * will provide the best performance. 
 * 
 * @author Edward Raff
 */
public class TransposeView extends GenericMatrix
{

	private static final long serialVersionUID = 7762422292840392481L;
	private Matrix base;

    public TransposeView(Matrix base)
    {
        this.base = base;
    }

    @Override
    public Vec getColumnView(int j)
    {
        return base.getRowView(j);
    }

    @Override
    public Vec getRowView(int r)
    {
        return base.getColumnView(r);
    }
    
    @Override
    protected Matrix getMatrixOfSameType(int rows, int cols)
    {
        return new DenseMatrix(rows, cols);
    }

    @Override
    public double get(int i, int j)
    {
        return base.get(j, i);
    }

    @Override
    public void set(int i, int j, double value)
    {
        base.set(j, i, value);
    }

    @Override
    public int rows()
    {
        return base.cols();
    }

    @Override
    public int cols()
    {
        return base.rows();
    }

    @Override
    public boolean isSparce()
    {
        return base.isSparce();
    }

    @Override
    public void changeSize(int newRows, int newCols)
    {
        base.changeSize(newCols, newRows);
    }
    
}
