package jsat.linear;

import java.io.Serializable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;

import jsat.utils.SystemInfo;

/**
 * This class uses the LUP decomposition of a matrix to provide efficient
 * methods for solving A x = b, as well as computing the determinant of A.
 *
 * @author Edward Raff
 */
public class LUPDecomposition implements Cloneable, Serializable {

  private static final long serialVersionUID = -149659693838168048L;
  private static final int threads = SystemInfo.LogicalCores;

  /**
   * Solves for the matrix x such that U x = y
   *
   * @param U
   *          an upper triangular matrix
   * @param y
   *          a matrix with the same number of rows as U
   * @return x such that U x = y
   */
  public static Matrix backSub(final Matrix U, final Matrix y) {
    if (y.rows() != U.rows()) {
      throw new ArithmeticException("Vector and matrix sizes do not agree");
    }

    final Matrix x = new DenseMatrix(U.cols(), y.cols());

    final double[] x_col_k = new double[y.rows()];

    final int start = Math.min(U.rows(), U.cols()) - 1;

    for (int k = 0; k < y.cols(); k++) {
      for (int i = start; i >= 0; i--)// We operate the same as
                                      // forwardSub(Matrix, Vec), but we aplly
                                      // each column of B as its own Vec.
      {
        x_col_k[i] = y.get(i, k);
        for (int j = i + 1; j <= start; j++) {
          x_col_k[i] -= U.get(i, j) * x_col_k[j];
        }
        x_col_k[i] /= U.get(i, i);
      }

      for (int i = 0; i < x_col_k.length; i++) {
        if (Double.isInfinite(x_col_k[i])) {
          x.set(i, k, 0);
        } else {
          x.set(i, k, x_col_k[i]);
        }
      }

    }

    return x;
  }

  /**
   * Solves for the matrix x such that U x = y
   *
   * @param U
   *          an upper triangular matrix
   * @param y
   *          a matrix with the same number of rows as U
   * @param threadpool
   *          source of threads for the parallel computation
   * @return x such that U x = y
   */
  public static Matrix backSub(final Matrix U, final Matrix y, final ExecutorService threadpool) {
    if (y.rows() != U.rows()) {
      throw new ArithmeticException("Vector and matrix sizes do not agree");
    }

    final Matrix x = new DenseMatrix(U.cols(), y.cols());
    final CountDownLatch latch = new CountDownLatch(threads);

    final int start = Math.min(U.rows(), U.cols()) - 1;

    for (int threadNum = 0; threadNum < threads; threadNum++) {
      final int threadID = threadNum;
      threadpool.submit(new Runnable() {

        @Override
        public void run() {
          final double[] x_col_k = new double[y.rows()];
          for (int k = threadID; k < y.cols(); k += threads) {
            for (int i = start; i >= 0; i--)// We operate the same as
                                            // forwardSub(Matrix, Vec), but we
                                            // aplly each column of B as its own
                                            // Vec.
            {
              x_col_k[i] = y.get(i, k);
              for (int j = i + 1; j <= start; j++) {
                x_col_k[i] -= U.get(i, j) * x_col_k[j];
              }
              x_col_k[i] /= U.get(i, i);
            }

            for (int i = 0; i < x_col_k.length; i++) {
              if (Double.isInfinite(x_col_k[i])) {
                x.set(i, k, 0);
              } else {
                x.set(i, k, x_col_k[i]);
              }
            }
          }
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (final InterruptedException ex) {
      Logger.getLogger(LUPDecomposition.class.getName()).log(Level.SEVERE, null, ex);
      return backSub(U, y);
    }

    return x;
  }

  /**
   * Solves for the vector x such that U x = y
   *
   * @param U
   *          an upper triangular matrix
   * @param y
   *          a vector whos length is equal to the rows in U
   * @return x such that U x = y
   */
  public static Vec backSub(final Matrix U, final Vec y) {
    if (y.length() != U.rows()) {
      throw new ArithmeticException("Vector and matrix sizes do not agree");
    }

    final Vec x = y instanceof SparseVector ? new SparseVector(U.cols()) : new DenseVector(U.cols());

    final int start = Math.min(U.rows(), U.cols()) - 1;

    for (int i = start; i >= 0; i--) {
      double x_i = y.get(i);
      for (int j = i + 1; j <= start; j++) {
        x_i -= U.get(i, j) * x.get(j);
      }
      x_i /= U.get(i, i);
      if (Double.isInfinite(x_i)) {// Occurs when U_(i,i) = 0
        x_i = 0;
      }
      x.set(i, x_i);
    }

    return x;
  }

  /**
   * Solves for the matrix x such that L x = b
   *
   * @param L
   *          a lower triangular matrix
   * @param b
   *          a matrix with the same number of rows as L
   * @return x such that L x = b
   */
  public static Matrix forwardSub(final Matrix L, final Matrix b) {
    if (b.rows() != L.rows()) {
      throw new ArithmeticException("Vector and matrix sizes do not agree");
    }

    final Matrix y = new DenseMatrix(b.rows(), b.cols());
    // Store the colum seperatly so that we can access this array in row major
    // order, instead of the matrix in column major (yay cache!)
    final double[] y_col_k = new double[b.rows()];
    for (int k = 0; k < b.cols(); k++) {
      for (int i = 0; i < b.rows(); i++)// We operate the same as
                                        // forwardSub(Matrix, Vec), but we aplly
                                        // each column of B as its own Vec.
      {
        y_col_k[i] = b.get(i, k);
        for (int j = 0; j < i; j++) {
          y_col_k[i] -= L.get(i, j) * y_col_k[j];
        }
        y_col_k[i] /= L.get(i, i);
      }

      for (int z = 0; z < y_col_k.length; z++) {
        y.set(z, k, y_col_k[z]);
      }
    }

    return y;
  }

  /**
   * Solves for the matrix x such that L x = b
   *
   * @param L
   *          a lower triangular matrix
   * @param b
   *          a matrix with the same number of rows as L
   * @param threadpool
   *          source of threads for the parallel computation
   * @return x such that L x = b
   */
  public static Matrix forwardSub(final Matrix L, final Matrix b, final ExecutorService threadpool) {
    if (b.rows() != L.rows()) {
      throw new ArithmeticException("Vector and matrix sizes do not agree");
    }

    final CountDownLatch latch = new CountDownLatch(threads);

    final Matrix y = new DenseMatrix(b.rows(), b.cols());
    for (int threadNum = 0; threadNum < threads; threadNum++) {
      final int threadID = threadNum;
      threadpool.submit(new Runnable() {

        @Override
        public void run() {
          // Store the colum seperatly so that we can access this array in row
          // major order, instead of the matrix in column major (yay cache!)
          final double[] y_col_k = new double[b.rows()];
          for (int k = threadID; k < b.cols(); k += threads) {
            for (int i = 0; i < b.rows(); i++)// We operate the same as
                                              // forwardSub(Matrix, Vec), but we
                                              // aplly each column of B as its
                                              // own Vec. We sawp the order for
                                              // better cache use
            {
              y_col_k[i] = b.get(i, k);
              for (int j = 0; j < i; j++) {
                y_col_k[i] -= L.get(i, j) * y_col_k[j];
              }
              y_col_k[i] /= L.get(i, i);

              // y.set(i, k, y_i);
            }

            for (int z = 0; z < y_col_k.length; z++) {
              y.set(z, k, y_col_k[z]);
            }
          }
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (final InterruptedException ex) {
      Logger.getLogger(LUPDecomposition.class.getName()).log(Level.SEVERE, null, ex);
      return forwardSub(L, b);
    }

    return y;
  }

  /**
   * Solves for the vector x such that L x = b
   *
   * @param L
   *          a lower triangular matrix
   * @param b
   *          a vector whos length is equal to the rows in L
   * @return x such that L x = b
   */
  public static Vec forwardSub(final Matrix L, final Vec b) {
    if (b.length() != L.rows()) {
      throw new ArithmeticException("Vector and matrix sizes do not agree");
    }

    final Vec y = b instanceof SparseVector ? new SparseVector(b.length()) : new DenseVector(b.length());

    for (int i = 0; i < b.length(); i++) {
      double y_i = b.get(i);
      for (int j = 0; j < i; j++) {
        y_i -= L.get(i, j) * y.get(j);
      }
      y_i /= L.get(i, i);

      y.set(i, y_i);
    }

    return y;
  }

  private final Matrix L, U, P;

  public LUPDecomposition(final Matrix A) {
    final Matrix[] lup = A.clone().lup();
    L = lup[0];
    U = lup[1];
    P = lup[2];
  }

  public LUPDecomposition(final Matrix A, final ExecutorService threadpool) {
    final Matrix[] lup = A.clone().lup(threadpool);
    L = lup[0];
    U = lup[1];
    P = lup[2];
  }

  public LUPDecomposition(final Matrix L, final Matrix U, final Matrix P) {
    this.L = L;
    this.U = U;
    this.P = P;
  }

  @Override
  public LUPDecomposition clone() {
    return new LUPDecomposition(L.clone(), U.clone(), P.clone());
  }

  /**
   *
   * @return the determinant of the original Matrix A, |A|
   */
  public double det() {
    if (!isSquare()) {
      throw new ArithmeticException("Rectangual matricies do not have a determinat");
    }
    double det = 1;

    for (int i = 0; i < Math.min(U.rows(), U.cols()); i++) {
      det *= U.get(i, i);
    }

    // We need to swap back P to get the sign, so we make a clone. This could be
    // cached if we need to
    int rowSwaps = 0;

    final Matrix pCopy = P.clone();
    // The number of row swaps in P is the sign change
    for (int i = 0; i < pCopy.cols(); i++) {
      if (pCopy.get(i, i) != 1) {
        rowSwaps++;
        int j = i + 1;
        while (pCopy.get(j, i) == 0) {
          j++;
        }
        pCopy.swapRows(i, j); // Dont really care who we swap with, it will work
                              // out in the end
      }
    }

    return rowSwaps % 2 != 0 ? -det : det;
  }

  /**
   *
   * @return true if the original matrix A, from which this factorization is
   *         from, is a square matrix
   */
  public boolean isSquare() {
    return L.isSquare() && U.isSquare();
  }

  public Matrix solve(final Matrix B) {
    // Solve P A x = L U x = P b, for x

    // First solve L y = P b
    final Matrix y = forwardSub(L, P.multiply(B));
    // Sole U x = y
    final Matrix x = backSub(U, y);

    return x;
  }

  public Matrix solve(final Matrix B, final ExecutorService threadpool) {
    // Solve P A x = L U x = P b, for x

    // First solve L y = P b
    final Matrix y = forwardSub(L, P.multiply(B), threadpool);
    // Sole U x = y
    final Matrix x = backSub(U, y, threadpool);

    return x;
  }

  public Vec solve(final Vec b) {
    // Solve P A x = L U x = P b, for x

    // First solve L y = P b
    final Vec y = forwardSub(L, P.multiply(b));
    // Sole U x = y
    final Vec x = backSub(U, y);

    return x;
  }
}
