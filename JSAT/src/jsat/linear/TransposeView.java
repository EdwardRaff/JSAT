
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
    private Matrix base;

    public TransposeView(Matrix base)
    {
        this.base = base;
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
    
}
