package jsat.math.rootfinding;

import static java.lang.Math.abs;
import static java.lang.Math.signum;
import static java.lang.Math.sqrt;
import jsat.linear.Vec;
import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public class RiddersMethod implements RootFinder {

  private static final long serialVersionUID = 8154909945080099018L;

  public static double root(final double eps, final double a, final double b, final Function f, final double... args) {
    return root(eps, 1000, a, b, 0, f, args);
  }

  public static double root(final double eps, final double a, final double b, final int pos, final Function f,
      final double... args) {
    return root(eps, 1000, a, b, pos, f, args);
  }

  public static double root(final double a, final double b, final Function f, final double... args) {
    return root(1e-15, 1000, a, b, 0, f, args);
  }

  public static double root(final double eps, int maxIterations, double x1, double x2, int pos, final Function f,
      double... args) {
    // We assume 1 dimensional function then
    if (args == null || args.length == 0) {
      pos = 0;
      args = new double[1];
    }

    args[pos] = x1;
    double fx1 = f.f(args);
    args[pos] = x2;
    double fx2 = f.f(args);
    final double halfEps = eps * 0.5;

    if (fx1 * fx2 >= 0) {
      throw new ArithmeticException("The given interval does not appear to bracket the root");
    }

    double dif = 1;// Measure the change interface values
    while (abs(x1 - x2) > eps && maxIterations-- > 0) {
      final double x3 = (x1 + x2) * 0.5;

      args[pos] = x3;
      final double fx3 = f.f(args);

      final double x4 = x3 + (x3 - x1) * signum(fx1 - fx2) * fx3 / sqrt(fx3 * fx3 - fx1 * fx2);

      args[pos] = x4;
      final double fx4 = f.f(args);
      if (fx3 * fx4 < 0) {
        x1 = x3;
        fx1 = fx3;
        x2 = x4;
        fx2 = fx4;
      } else if (fx1 * fx4 < 0) {
        dif = abs(x4 - x2);
        if (dif <= halfEps) {// WE are no longer updating, return the value
          return x4;
        }
        x2 = x4;
        fx2 = fx4;
      } else {
        dif = abs(x4 - x1);
        if (dif <= halfEps) {// WE are no longer updating, return the value
          return x4;
        }
        x1 = x4;
        fx1 = fx4;
      }

    }

    return x2;
  }

  @Override
  public int guessesNeeded() {
    return 2;
  }

  @Override
  public double root(final double eps, final int maxIterations, final double[] initialGuesses, final Function f,
      final int pos, final double... args) {
    return root(eps, maxIterations, initialGuesses[0], initialGuesses[1], pos, f, args);
  }

  @Override
  public double root(final double eps, final int maxIterations, final double[] initialGuesses, final Function f,
      final int pos, final Vec args) {
    return root(eps, maxIterations, initialGuesses[0], initialGuesses[1], pos, f, args.arrayCopy());
  }
}
