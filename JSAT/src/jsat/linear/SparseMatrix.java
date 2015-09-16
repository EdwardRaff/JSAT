package jsat.linear;

import java.util.Arrays;
import java.util.Iterator;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.utils.SystemInfo;

/**
 * Creates a new Sparse Matrix where each row is backed by a sparse vector. <br>
 * <br>
 * This implementation does not support the {@link #qr() QR} or {@link #lup() }
 * decompositions. <br>
 * {@link #transposeMultiply(jsat.linear.Matrix, jsat.linear.Matrix, java.util.concurrent.ExecutorService) }
 * currently does not use multiple cores.
 *
 * @author Edward Raff
 */
public class SparseMatrix extends Matrix {

  private static final long serialVersionUID = -4087445771022578544L;
  private SparseVector[] rows;

  /**
   * Creates a new sparse matrix
   *
   * @param rows
   *          the number of rows for the matrix
   * @param cols
   *          the number of columns for the matrix
   */
  public SparseMatrix(final int rows, final int cols) {
    this.rows = new SparseVector[rows];
    for (int i = 0; i < rows; i++) {
      this.rows[i] = new SparseVector(cols);
    }
  }

  /**
   * Creates a new sparse matrix
   *
   * @param rows
   *          the number of rows for the matrix
   * @param cols
   *          the number of columns for the matrix
   * @param rowCapacity
   *          the initial capacity for non zero values for each row
   */
  public SparseMatrix(final int rows, final int cols, final int rowCapacity) {
    this.rows = new SparseVector[rows];
    for (int i = 0; i < rows; i++) {
      this.rows[i] = new SparseVector(cols, rowCapacity);
    }
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  protected SparseMatrix(final SparseMatrix toCopy) {
    rows = new SparseVector[toCopy.rows.length];
    for (int i = 0; i < rows.length; i++) {
      rows[i] = toCopy.rows[i].clone();
    }
  }

  /**
   * Creates a new Sparse Matrix backed by the given array of SpareVectors.
   * Altering the array of any object in it will also alter the this matrix.
   *
   * @param rows
   *          the array to back this SparseMatrix
   */
  public SparseMatrix(final SparseVector[] rows) {
    this.rows = rows;
    for (int i = 0; i < rows.length; i++) {
      if (rows[i].length() != rows[0].length()) {
        throw new IllegalArgumentException(
            "Row " + i + " has " + rows[i].length() + " columns instead of " + rows[0].length());
      }
    }
  }

  @Override
  public void changeSize(final int newRows, final int newCols) {
    if (newRows <= 0) {
      throw new ArithmeticException("Matrix must have a positive number of rows");
    }
    if (newCols <= 0) {
      throw new ArithmeticException("Matrix must have a positive number of columns");
    }
    final int oldRows = rows.length;
    if (newCols != cols()) {
      for (final SparseVector row_i : rows) {
        while (row_i.getLastNonZeroIndex() >= newCols) {
          row_i.set(row_i.getLastNonZeroIndex(), 0);
        }
        row_i.setLength(newCols);
      }
    }
    // update new rows
    rows = Arrays.copyOf(rows, newRows);
    for (int i = oldRows; i < newRows; i++) {
      rows[i] = new SparseVector(newCols);
    }
  }

  @Override
  public SparseMatrix clone() {
    return new SparseMatrix(this);
  }

  @Override
  public int cols() {
    return rows[0].length();
  }

  @Override
  public double get(final int i, final int j) {
    return rows[i].get(j);
  }

  @Override
  public Vec getRowView(final int r) {
    return rows[r];
  }

  @Override
  public void increment(final int i, final int j, final double value) {
    rows[i].increment(j, value);
  }

  @Override
  public boolean isSparce() {
    return true;
  }

  @Override
  public Matrix[] lup() {
    throw new UnsupportedOperationException("Not supported yet."); // To change
                                                                   // body of
                                                                   // generated
                                                                   // methods,
                                                                   // choose
                                                                   // Tools |
                                                                   // Templates.
  }

  @Override
  public Matrix[] lup(final ExecutorService threadPool) {
    throw new UnsupportedOperationException("Not supported yet."); // To change
                                                                   // body of
                                                                   // generated
                                                                   // methods,
                                                                   // choose
                                                                   // Tools |
                                                                   // Templates.
  }

  @Override
  public void multiply(final Matrix B, final Matrix C) {
    if (!canMultiply(this, B)) {
      throw new ArithmeticException("Matrix dimensions do not agree");
    } else if (rows() != C.rows() || B.cols() != C.cols()) {
      throw new ArithmeticException("Target Matrix is no the correct size");
    }

    for (int i = 0; i < C.rows(); i++) {
      final Vec Arowi = rows[i];
      final Vec Crowi = C.getRowView(i);

      for (final IndexValue iv : Arowi) {
        final int k = iv.getIndex();
        final double a = iv.getValue();
        final Vec Browk = B.getRowView(k);
        Crowi.mutableAdd(a, Browk);
      }
    }
  }

  @Override
  public void multiply(final Matrix B, final Matrix C, final ExecutorService threadPool) {
    if (!canMultiply(this, B)) {
      throw new ArithmeticException("Matrix dimensions do not agree");
    } else if (rows() != C.rows() || B.cols() != C.cols()) {
      throw new ArithmeticException("Target Matrix is no the correct size");
    }

    final CountDownLatch latch = new CountDownLatch(C.rows());
    for (int i = 0; i < C.rows(); i++) {
      final Vec Arowi = rows[i];
      final Vec Crowi = C.getRowView(i);

      threadPool.submit(new Runnable() {
        @Override
        public void run() {
          for (final IndexValue iv : Arowi) {
            final int k = iv.getIndex();
            final double a = iv.getValue();
            final Vec Browk = B.getRowView(k);
            Crowi.mutableAdd(a, Browk);
          }

          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (final InterruptedException ex) {
      Logger.getLogger(SparseMatrix.class.getName()).log(Level.SEVERE, null, ex);
    }
  }

  @Override
  public void multiply(final Vec b, final double z, final Vec c) {
    if (cols() != b.length()) {
      throw new ArithmeticException(
          "Matrix dimensions do not agree, [" + rows() + "," + cols() + "] x [" + b.length() + ",1]");
    }
    if (rows() != c.length()) {
      throw new ArithmeticException("Target vector dimension does not agree with matrix dimensions. Matrix has "
          + rows() + " rows but tagert has " + c.length());
    }

    for (int i = 0; i < rows(); i++) {
      final SparseVector row = rows[i];
      c.increment(i, row.dot(b) * z);
    }
  }

  @Override
  public void multiplyTranspose(final Matrix B, final Matrix C) {
    if (cols() != B.cols()) {
      throw new ArithmeticException("Matrix dimensions do not agree");
    } else if (rows() != C.rows() || B.rows() != C.cols()) {
      throw new ArithmeticException("Target Matrix is no the correct size");
    }

    for (int i = 0; i < rows(); i++) {
      final SparseVector A_i = rows[i];
      for (int j = 0; j < B.rows(); j++) {
        final Vec B_j = B.getRowView(j);
        double C_ij = 0;

        if (!B_j.isSparse()) // B is dense, lets do this the easy way
        {
          for (final IndexValue iv : A_i) {
            C_ij += iv.getValue() * B_j.get(iv.getIndex());
          }
          C.increment(i, j, C_ij);
          continue;// Skip early, we did it!
        }
        // else, sparse
        final Iterator<IndexValue> A_iter = A_i.getNonZeroIterator();
        final Iterator<IndexValue> B_iter = B_j.getNonZeroIterator();
        if (!B_iter.hasNext() || !A_iter.hasNext()) {// one is all zeros,
                                                     // nothing to do
          continue;
        }

        IndexValue A_val = A_iter.next();
        IndexValue B_val = B_iter.next();

        while (A_val != null && B_val != null)// go add everything together!
        {
          if (A_val.getIndex() == B_val.getIndex()) // inc and bump both
          {
            C_ij += A_val.getValue() * B_val.getValue();
            if (A_iter.hasNext()) {
              A_val = A_iter.next();
            } else {
              A_val = null;
            }
            if (B_iter.hasNext()) {
              B_val = B_iter.next();
            } else {
              B_val = null;
            }
          } else if (A_val.getIndex() < B_val.getIndex()) // A is behind, bump
                                                          // it
          {
            if (A_iter.hasNext()) {
              A_val = A_iter.next();
            } else {
              A_val = null;
            }
          } else// B is behind, bump it
            if (B_iter.hasNext()) {
            B_val = B_iter.next();
          } else {
            B_val = null;
          }
        }

        C.increment(i, j, C_ij);
      }
    }
  }

  @Override
  public void multiplyTranspose(final Matrix B, final Matrix C, final ExecutorService threadPool) {
    if (cols() != B.cols()) {
      throw new ArithmeticException("Matrix dimensions do not agree");
    } else if (rows() != C.rows() || B.rows() != C.cols()) {
      throw new ArithmeticException("Target Matrix is no the correct size");
    }

    final SparseMatrix A = this;
    final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
    for (int id = 0; id < SystemInfo.LogicalCores; id++) {
      final int ID = id;
      threadPool.submit(new Runnable() {

        @Override
        public void run() {
          try {
            for (int i = ID; i < A.rows(); i += SystemInfo.LogicalCores) {
              final SparseVector A_i = A.rows[i];
              for (int j = 0; j < B.rows(); j++) {
                final Vec B_j = B.getRowView(j);
                double C_ij = 0;

                if (!B_j.isSparse()) // B is dense, lets do this the easy way
                {
                  for (final IndexValue iv : A_i) {
                    C_ij += iv.getValue() * B_j.get(iv.getIndex());
                  }
                  C.increment(i, j, C_ij);
                  continue;// Skip early, we did it!
                }
                // else, sparse
                final Iterator<IndexValue> A_iter = A_i.getNonZeroIterator();
                final Iterator<IndexValue> B_iter = B_j.getNonZeroIterator();
                if (!B_iter.hasNext() || !A_iter.hasNext()) {// one is all
                                                             // zeros, nothing
                                                             // to do
                  continue;
                }

                IndexValue A_val = A_iter.next();
                IndexValue B_val = B_iter.next();

                while (A_val != null && B_val != null)// go add everything
                                                      // together!
                {
                  if (A_val.getIndex() == B_val.getIndex()) // inc and bump both
                  {
                    C_ij += A_val.getValue() * B_val.getValue();
                    if (A_iter.hasNext()) {
                      A_val = A_iter.next();
                    } else {
                      A_val = null;
                    }
                    if (B_iter.hasNext()) {
                      B_val = B_iter.next();
                    } else {
                      B_val = null;
                    }
                  } else if (A_val.getIndex() < B_val.getIndex()) // A is
                                                                  // behind,
                                                                  // bump it
                  {
                    if (A_iter.hasNext()) {
                      A_val = A_iter.next();
                    } else {
                      A_val = null;
                    }
                  } else// B is behind, bump it
                    if (B_iter.hasNext()) {
                    B_val = B_iter.next();
                  } else {
                    B_val = null;
                  }
                }

                C.increment(i, j, C_ij);
              }
            }

          } catch (final Exception ex) {
            ex.printStackTrace();
          }
          System.out.println(ID + " fin");
          latch.countDown();
        }
      });
    }

    try {
      latch.await();
    } catch (final InterruptedException ex) {
      Logger.getLogger(SparseMatrix.class.getName()).log(Level.SEVERE, null, ex);
    }
  }

  @Override
  public void mutableAdd(final double c) {
    for (final SparseVector row : rows) {
      row.mutableAdd(c);
    }
  }

  @Override
  public void mutableAdd(final double c, final ExecutorService threadPool) {
    final CountDownLatch latch = new CountDownLatch(rows.length);
    for (final SparseVector row : rows) {
      threadPool.submit(new Runnable() {
        @Override
        public void run() {
          row.mutableAdd(c);
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (final InterruptedException ex) {
      Logger.getLogger(SparseMatrix.class.getName()).log(Level.SEVERE, null, ex);
    }
  }

  @Override
  public void mutableAdd(final double c, final Matrix B) {
    if (!Matrix.sameDimensions(this, B)) {
      throw new ArithmeticException("Matrices must be the same dimension to be added");
    }
    for (int i = 0; i < rows.length; i++) {
      rows[i].mutableAdd(c, B.getRowView(i));
    }
  }

  @Override
  public void mutableAdd(final double c, final Matrix B, final ExecutorService threadPool) {
    if (!Matrix.sameDimensions(this, B)) {
      throw new ArithmeticException("Matrices must be the same dimension to be added");
    }

    final CountDownLatch latch = new CountDownLatch(rows.length);
    for (int i = 0; i < rows.length; i++) {
      final int ii = i;
      threadPool.submit(new Runnable() {
        @Override
        public void run() {
          rows[ii].mutableAdd(c, B.getRowView(ii));
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (final InterruptedException ex) {
      Logger.getLogger(SparseMatrix.class.getName()).log(Level.SEVERE, null, ex);
    }
  }

  @Override
  public void mutableMultiply(final double c) {
    for (final SparseVector row : rows) {
      row.mutableMultiply(c);
    }
  }

  @Override
  public void mutableMultiply(final double c, final ExecutorService threadPool) {
    final CountDownLatch latch = new CountDownLatch(rows.length);
    for (final SparseVector row : rows) {
      threadPool.submit(new Runnable() {
        @Override
        public void run() {
          row.mutableMultiply(c);
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (final InterruptedException ex) {
      Logger.getLogger(SparseMatrix.class.getName()).log(Level.SEVERE, null, ex);
    }
  }

  @Override
  public void mutableTranspose() {
    for (int i = 0; i < rows() - 1; i++) {
      for (int j = i + 1; j < cols(); j++) {
        final double tmp = get(j, i);
        set(j, i, get(i, j));
        set(i, j, tmp);
      }
    }
  }

  @Override
  public long nnz() {
    int nnz = 0;
    for (final Vec v : rows) {
      nnz += v.nnz();
    }
    return nnz;
  }

  @Override
  public Matrix[] qr() {
    throw new UnsupportedOperationException("Not supported yet."); // To change
                                                                   // body of
                                                                   // generated
                                                                   // methods,
                                                                   // choose
                                                                   // Tools |
                                                                   // Templates.
  }

  @Override
  public Matrix[] qr(final ExecutorService threadPool) {
    throw new UnsupportedOperationException("Not supported yet."); // To change
                                                                   // body of
                                                                   // generated
                                                                   // methods,
                                                                   // choose
                                                                   // Tools |
                                                                   // Templates.
  }

  @Override
  public int rows() {
    return rows.length;
  }

  @Override
  public void set(final int i, final int j, final double value) {
    rows[i].set(j, value);
  }

  @Override
  public void swapRows(final int r1, final int r2) {
    final SparseVector tmp = rows[r2];
    rows[r2] = rows[r1];
    rows[r1] = tmp;
  }

  @Override
  public void transpose(final Matrix C) {
    if (rows() != C.cols() || cols() != C.rows()) {
      throw new ArithmeticException("Target matrix does not have the correct dimensions");
    }

    C.zeroOut();
    for (int row = 0; row < rows.length; row++) {
      for (final IndexValue iv : rows[row]) {
        C.set(iv.getIndex(), row, iv.getValue());
      }
    }
  }

  @Override
  public void transposeMultiply(final double c, final Vec b, final Vec x) {
    if (rows() != b.length()) {
      throw new ArithmeticException(
          "Matrix dimensions do not agree, [" + cols() + "," + rows() + "] x [" + b.length() + ",1]");
    } else if (cols() != x.length()) {
      throw new ArithmeticException("Matrix dimensions do not agree with target vector");
    }

    for (final IndexValue b_iv : b) {
      x.mutableAdd(c * b_iv.getValue(), rows[b_iv.getIndex()]);
    }
  }

  @Override
  public void transposeMultiply(final Matrix B, final Matrix C) {
    if (rows() != B.rows()) {// Normaly it is A_cols == B_rows, but we are doint
                             // A'*B, not A*B
      throw new ArithmeticException("Matrix dimensions do not agree");
    } else if (cols() != C.rows() || B.cols() != C.cols()) {
      throw new ArithmeticException("Destination matrix does not have matching dimensions");
    }
    final SparseMatrix A = this;
    /// Should choose step size such that 2*NB2^2 * dataTypeSize <= CacheSize

    final int kLimit = rows();

    for (int k = 0; k < kLimit; k++) {
      final Vec bRow_k = B.getRowView(k);
      final Vec aRow_k = A.getRowView(k);

      for (final IndexValue iv : aRow_k)// iterating over "i"
      {

        final Vec cRow_i = C.getRowView(iv.getIndex());
        final double a = iv.getValue();// A.get(k, i);

        cRow_i.mutableAdd(a, bRow_k);
      }
    }
  }

  @Override
  public void transposeMultiply(final Matrix B, final Matrix C, final ExecutorService threadPool) {
    transposeMultiply(B, C);// TODO use the multiple threads
  }

  @Override
  public void zeroOut() {
    for (final Vec row : rows) {
      row.zeroOut();
    }
  }

}
