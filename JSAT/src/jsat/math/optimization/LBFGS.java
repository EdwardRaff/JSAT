package jsat.math.optimization;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionP;
import jsat.math.FunctionVec;
import jsat.utils.DoubleList;

/**
 * Implementation of the Limited memory variant of {@link BFGS}. It uses a
 * history of {@link #setM(int) m} items to solve {@code n} dimension problems
 * with {@code O(m n)} work per iteration.
 *
 * @author Edward Raff
 */
public class LBFGS implements Optimizer2 {

  /**
   * See Algorithm 7.4 (L-BFGS two-loop recursion).
   *
   * @param x_grad
   *          the initial value &nabla; f<sub>k</sub>
   * @param rho
   * @param s
   * @param y
   * @param q
   *          the location to store the value of H<sub>k</sub> &nabla; f
   *          <sub>k</sub>
   * @param alphas
   *          temp space to do work, should be as large as the number of history
   *          vectors
   */
  public static void twoLoopHp(final Vec x_grad, final List<Double> rho, final List<Vec> s, final List<Vec> y,
      final Vec q, final double[] alphas) {
    // q ← ∇ fk;
    x_grad.copyTo(q);
    if (s.isEmpty()) {
      return;// identity, we are done
    }
    // for i = k−1,k−2,...,k−m
    for (int i = 0; i < s.size(); i++) {
      final Vec s_i = s.get(i);
      final Vec y_i = y.get(i);
      final double alpha_i = alphas[i] = rho.get(i) * s_i.dot(q);
      q.mutableSubtract(alpha_i, y_i);
    }

    // r ← Hk0q; and see eq (7.20), done in place in q
    q.mutableMultiply(s.get(0).dot(y.get(0)) / y.get(0).dot(y.get(0)));
    // for i = k−m,k−m+1,...,k−1
    for (int i = s.size() - 1; i >= 0; i--) {
      // β ← ρ_i y_i^T r ;
      final double beta = rho.get(i) * y.get(i).dot(q);
      // r ← r + si (αi − β)
      q.mutableAdd(alphas[i] - beta, s.get(i));
    }
  }

  private int m;
  private int maxIterations;
  private LineSearch lineSearch;

  private boolean inftNormCriterion = true;

  /**
   * Creates a new L-BFGS optimization object that uses a maximum of 500
   * iterations and a {@link BacktrackingArmijoLineSearch Backtracking} line
   * search. A {@link #setM(int) history} of 10 items will be used
   */
  public LBFGS() {
    this(10);
  }

  /**
   * Creates a new L-BFGS optimization object that uses a maximum of 500
   * iterations and a {@link BacktrackingArmijoLineSearch Backtracking} line
   * search.
   *
   * @param m
   *          the number of history items
   */
  public LBFGS(final int m) {
    this(m, 500, new BacktrackingArmijoLineSearch());
  }

  /**
   * Creates a new L-BFGS optimization object
   *
   * @param m
   *          the number of history items
   * @param maxIterations
   *          the maximum number of iterations before stopping
   * @param lineSearch
   *          the line search method to use for optimization
   */
  public LBFGS(final int m, final int maxIterations, final LineSearch lineSearch) {
    setM(m);
    setMaximumIterations(maxIterations);
    setLineSearch(lineSearch);
  }

  @Override
  public LBFGS clone() {
    return new LBFGS(m, maxIterations, lineSearch.clone());
  }

  /**
   * Returns the line search method used at each iteration
   *
   * @return the line search method used at each iteration
   */
  public LineSearch getLineSearch() {
    return lineSearch;
  }

  /**
   * Returns the number of history items that will be used
   *
   * @return the number of history items that will be used
   */
  public int getM() {
    return m;
  }

  @Override
  public int getMaximumIterations() {
    return maxIterations;
  }

  private double gradConvgHelper(final Vec grad) {
    if (!inftNormCriterion) {
      return grad.pNorm(2);
    }
    double max = 0;
    for (final IndexValue iv : grad) {
      max = Math.max(max, Math.abs(iv.getValue()));
    }
    return max;
  }

  /**
   * Returns whether or not the infinity norm ({@code true}) or 2 norm (
   * {@code false}) is used to determine convergence.
   *
   * @return {@code true} if the infinity norm is in use, {@code false} for the
   *         2 norm
   */
  public boolean isInftNormCriterion() {
    return inftNormCriterion;
  }

  @Override
  public void optimize(final double tolerance, final Vec w, final Vec x0, final Function f, final FunctionVec fp,
      final FunctionVec fpp) {
    optimize(tolerance, w, x0, f, fp, fpp, null);
  }

  @Override
  public void optimize(final double tolerance, final Vec w, final Vec x0, final Function f, final FunctionVec fp,
      final FunctionVec fpp, final ExecutorService ex) {
    final LineSearch search = lineSearch.clone();
    final double[] f_xVal = new double[1];// store place for f_x

    // history for implicit H
    final List<Double> Rho = new DoubleList(m);
    final List<Vec> S = new ArrayList<Vec>(m);
    final List<Vec> Y = new ArrayList<Vec>(m);

    final Vec x_prev = x0.clone();
    final Vec x_cur = x0.clone();
    f_xVal[0] = ex != null && f instanceof FunctionP ? ((FunctionP) f).f(x_prev, ex) : f.f(x_prev);
    // graidnet
    Vec x_grad = x0.clone();
    x_grad.zeroOut();
    final Vec x_gradPrev = x_grad.clone();
    // p_l
    final Vec p_k = x_grad.clone();
    final Vec s_k = x_grad.clone();
    final Vec y_k = x_grad.clone();

    x_grad = ex != null ? fp.f(x_cur, x_grad, ex) : fp.f(x_cur, x_grad);

    final double[] alphas = new double[m];
    int iter = 0;

    while (gradConvgHelper(x_grad) > tolerance && iter < maxIterations) {
      // p_k = −H_k ∇f_k; (6.18)
      twoLoopHp(x_grad, Rho, S, Y, p_k, alphas);
      p_k.mutableMultiply(-1);

      // Set x_k+1 = x_k + α_k p_k where α_k is computed from a line search
      x_cur.copyTo(x_prev);
      x_grad.copyTo(x_gradPrev);

      final double alpha_k = search.lineSearch(1.0, x_prev, x_gradPrev, p_k, f, fp, f_xVal[0], x_gradPrev.dot(p_k),
          x_cur, f_xVal, x_grad, ex);
      if (alpha_k < 1e-12 && iter > 25) {// if we are making near epsilon steps
                                         // consider it done
        break;
      }

      if (!search.updatesGrad()) {
        if (ex != null) {
          fp.f(x_cur, x_grad, ex);
        } else {
          fp.f(x_cur, x_grad);
        }
      }

      // Define s_k =x_k+1 −x_k and y_k = ∇f_k+1 −∇f_k;
      x_cur.copyTo(s_k);
      s_k.mutableSubtract(x_prev);
      S.add(0, s_k.clone());

      x_grad.copyTo(y_k);
      y_k.mutableSubtract(x_gradPrev);
      Y.add(0, y_k.clone());

      Rho.add(0, 1 / s_k.dot(y_k));
      if (Double.isInfinite(Rho.get(0)) || Double.isNaN(Rho.get(0))) {
        Rho.clear();
        S.clear();
        Y.clear();
      }
      while (Rho.size() > m) {
        Rho.remove(m);
        S.remove(m);
        Y.remove(m);
      }

      iter++;
    }

    x_cur.copyTo(w);
  }

  /**
   * By default the infinity norm is used to judge convergence. If set to
   * {@code false}, the 2 norm will be used instead.
   *
   * @param inftNormCriterion
   */
  public void setInftNormCriterion(final boolean inftNormCriterion) {
    this.inftNormCriterion = inftNormCriterion;
  }

  /**
   * Sets the line search method used at each iteration
   *
   * @param lineSearch
   *          the line search method used at each iteration
   */
  public void setLineSearch(final LineSearch lineSearch) {
    this.lineSearch = lineSearch;
  }

  /**
   * Sets the number of history items to keep that are used to approximate the
   * Hessian of the problem
   *
   * @param m
   *          the number of history items to keep
   */
  public void setM(final int m) {
    if (m < 1) {
      throw new IllegalArgumentException("m must be positive, not " + m);
    }
    this.m = m;
  }

  @Override
  public void setMaximumIterations(final int iterations) {
    if (iterations < 1) {
      throw new IllegalArgumentException("Number of iterations must be positive, not " + iterations);
    }
    maxIterations = iterations;
  }

}
