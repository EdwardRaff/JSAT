package jsat.linear;

/**
 * This class provides a free view of the transpose of a matrix. This is done by
 * accessing the matrix elements in swapped order. This has a serious
 * performance impact. If the base matrix storage is row major order, then
 * accessing the TransposeView in column major order will provide the best
 * performance.
 *
 * @author Edward Raff
 */
public class TransposeView extends GenericMatrix {

  private static final long serialVersionUID = 7762422292840392481L;
  private final Matrix base;

  public TransposeView(final Matrix base) {
    this.base = base;
  }

  @Override
  public void changeSize(final int newRows, final int newCols) {
    base.changeSize(newCols, newRows);
  }

  @Override
  public int cols() {
    return base.rows();
  }

  @Override
  public double get(final int i, final int j) {
    return base.get(j, i);
  }

  @Override
  public Vec getColumnView(final int j) {
    return base.getRowView(j);
  }

  @Override
  protected Matrix getMatrixOfSameType(final int rows, final int cols) {
    return new DenseMatrix(rows, cols);
  }

  @Override
  public Vec getRowView(final int r) {
    return base.getColumnView(r);
  }

  @Override
  public boolean isSparce() {
    return base.isSparce();
  }

  @Override
  public int rows() {
    return base.cols();
  }

  @Override
  public void set(final int i, final int j, final double value) {
    base.set(j, i, value);
  }

}
