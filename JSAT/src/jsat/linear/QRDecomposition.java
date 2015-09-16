package jsat.linear;

import static java.lang.Math.abs;
import static java.lang.Math.min;
import java.io.Serializable;
import java.util.concurrent.ExecutorService;

/**
 *
 * @author Edward Raff
 */
public class QRDecomposition implements Serializable {

  private static final long serialVersionUID = 7578073062361216223L;
  private final Matrix Q_T, R;

  public QRDecomposition(final Matrix A) {
    final Matrix[] qr = A.clone().qr();
    Q_T = qr[0];
    Q_T.mutableTranspose();
    R = qr[1];
  }

  public QRDecomposition(final Matrix A, final ExecutorService threadpool) {
    final Matrix[] qr = A.clone().qr(threadpool);
    Q_T = qr[0];
    Q_T.mutableTranspose();
    R = qr[1];
  }

  public QRDecomposition(final Matrix Q, final Matrix R) {
    if (!Q.isSquare()) {
      throw new ArithmeticException("Q is always square, rectangular Q is invalid");
    } else if (Q.rows() != R.rows()) {
      throw new ArithmeticException("Q and R do not agree");
    }

    Q_T = Q;
    Q_T.mutableTranspose();
    this.R = R;
  }

  /**
   *
   * @return the absolute value of the determinant of the original Matrix,
   *         abs(|A|)
   */
  public double absDet() {
    if (!R.isSquare()) {
      throw new ArithmeticException("Can only compute the determinant of a square matrix");
    }

    double absD = 1;
    for (int i = 0; i < min(R.rows(), R.cols()); i++) {
      absD *= R.get(i, i);
    }

    return abs(absD);
  }

  public Matrix solve(final Matrix B) {
    // A * x = B, we want x
    // QR x = b
    // R * x = Q' * b

    final Matrix y = Q_T.multiply(B);

    // Solve R * x = y using back substitution
    final Matrix x = LUPDecomposition.backSub(R, y);

    return x;
  }

  public Matrix solve(final Matrix B, final ExecutorService threadpool) {
    // A * x = B, we want x
    // QR x = b
    // R * x = Q' * b

    final Matrix y = Q_T.multiply(B, threadpool);

    // Solve R * x = y using back substitution
    final Matrix x = LUPDecomposition.backSub(R, y, threadpool);

    return x;
  }

  public Vec solve(final Vec b) {
    if (b.length() != R.rows()) {
      throw new ArithmeticException("Matrix vector dimensions do not agree");
      // A * x = b, we want x
      // QR x = b
      // R * x = Q' * b
    }

    final Vec y = Q_T.multiply(b);

    // Solve R * x = y using back substitution
    final Vec x = LUPDecomposition.backSub(R, y);

    return x;
  }
}
