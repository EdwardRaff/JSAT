package jsat.linear;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class provides a base mechanism to create a Matrix 'view' from a list of
 * {@link Vec} objects. The vector objects will be used to back the rows of the
 * matrix, so changes to one are shown in the other. If the matirx is altered
 * using {@link #changeSize(int, int) }, this may no longer be true. <br>
 * Row oriented operations are implemented to defer to the base vector objects
 * used.
 *
 * @author Edward Raff
 */
public class MatrixOfVecs extends GenericMatrix {

  private static final long serialVersionUID = 6120353195388663462L;
  private final List<Vec> rows;

  /**
   * Creates a new Matrix of Vecs of the desired size.
   *
   * @param rows
   *          the number of rows in the matrix
   * @param cols
   *          the number of columns in the matrix
   * @param sparse
   *          if {@code true} {@link SparseVector} objects will be used for the
   *          rows. Else {@link DenseVector} will be used.
   */
  public MatrixOfVecs(final int rows, final int cols, final boolean sparse) {
    this.rows = new ArrayList<Vec>(rows);
    for (int i = 0; i < rows; i++) {
      this.rows.add(sparse ? new SparseVector(cols) : new DenseVector(cols));
    }
  }

  /**
   * Creates a new Matrix of Vecs from the given list of Vec objects. All Vec
   * objects contained should be of the same length.
   *
   * @param rows
   *          the rows of vecs to make this matrix.
   */
  public MatrixOfVecs(final List<Vec> rows) {
    this.rows = new ArrayList<Vec>(rows);
    final int cols = rows.get(0).length();
    for (final Vec v : rows) {
      if (cols != v.length()) {
        throw new IllegalArgumentException("Row vectors must all be of the same length");
      }
    }
  }

  /**
   * Creates a new Matrix of Vecs from the given array of Vec objects. All Vec
   * objects contained should be of the same length.
   *
   * @param rows
   *          the rows of vecs to make this matrix.
   */
  public MatrixOfVecs(final Vec... rows) {
    this(Arrays.asList(rows));
  }

  @Override
  public void changeSize(final int newRows, final int newCols) {
    if (newRows <= 0 || newCols <= 0) {
      throw new IllegalArgumentException(
          "Rows and columns must be positive, new dimension of [" + newRows + "," + newCols + "] is invalid");
    }
    // change cols first, add new rows of the correct size after
    if (newCols != cols()) {
      for (int i = 0; i < rows(); i++) {
        final Vec orig = rows.get(i);
        final Vec newV = orig.isSparse() ? new SparseVector(newCols) : new DenseVector(newCols);
        if (newCols < orig.length()) {
          new SubVector(0, newCols, orig).copyTo(newV);
        } else {
          orig.copyTo(new SubVector(0, orig.length(), newV));
        }
        rows.set(i, newV);
      }
    }

    if (newRows < rows()) {
      rows.subList(newRows, rows()).clear();
    } else if (newRows > rows()) {
      while (rows.size() < newRows) {
        final Vec newV = rows.get(rows.size() - 1).clone();
        newV.zeroOut();
        rows.add(newV);
      }
    }
  }

  @Override
  public MatrixOfVecs clone() {
    final MatrixOfVecs clone = new MatrixOfVecs(rows);
    for (int i = 0; i < clone.rows.size(); i++) {
      clone.rows.set(i, clone.rows.get(i).clone());
    }
    return clone;
  }

  @Override
  public int cols() {
    return rows.get(0).length();
  }

  @Override
  public double get(final int i, final int j) {
    if (i >= rows() || i < 0) {
      throw new IndexOutOfBoundsException("row " + i + " is not a valid index");
    } else if (j >= cols() || j < 0) {
      throw new IndexOutOfBoundsException("column " + j + " is not a valid index");
    }
    return rows.get(i).get(j);
  }

  @Override
  protected Matrix getMatrixOfSameType(final int rows, final int cols) {
    return new MatrixOfVecs(rows, cols, isSparce());
  }

  @Override
  public Vec getRowView(final int r) {
    if (r >= rows() || r < 0) {
      throw new IndexOutOfBoundsException("row " + r + " is not a valid index");
    }
    return rows.get(r);
  }

  @Override
  public void increment(final int i, final int j, final double value) {
    if (i >= rows() || i < 0) {
      throw new IndexOutOfBoundsException("row " + i + " is not a valid index");
    } else if (j >= cols() || j < 0) {
      throw new IndexOutOfBoundsException("column " + j + " is not a valid index");
    }
    rows.get(i).increment(j, value);
  }

  @Override
  public boolean isSparce() {
    for (final Vec v : rows) {
      if (v.isSparse()) {
        return true;
      }
    }
    return false;
  }

  @Override
  public void mutableAdd(final double c) {
    for (final Vec row : rows) {
      row.mutableAdd(c);
    }
  }

  @Override
  public void mutableMultiply(final double c) {
    for (final Vec row : rows) {
      row.mutableMultiply(c);
    }
  }

  @Override
  public int rows() {
    return rows.size();
  }

  @Override
  public void set(final int i, final int j, final double value) {
    if (i >= rows() || i < 0) {
      throw new IndexOutOfBoundsException("row " + i + " is not a valid index");
    } else if (j >= cols() || j < 0) {
      throw new IndexOutOfBoundsException("column " + j + " is not a valid index");
    }
    rows.get(i).set(j, value);
  }

  @Override
  public void updateRow(final int i, final double c, final Vec b) {
    rows.get(i).mutableAdd(c, b);
  }

  @Override
  public void zeroOut() {
    for (final Vec row : rows) {
      row.zeroOut();
    }
  }

}
