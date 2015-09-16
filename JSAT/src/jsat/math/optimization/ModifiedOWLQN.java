/*
 * Copyright (C) 2015 Edward Raff <Raff.Edward@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.math.optimization;

import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.signum;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.linear.ConstantVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionP;
import jsat.math.FunctionVec;
import jsat.utils.DoubleList;

/**
 * This implements the Modified Orthant-Wise Limited memory
 * Quasi-Newton(mOWL-QN) optimizer. This algorithm is an extension of
 * {@link LBFGS}, and solves minimization problems of the form: f(x) +
 * {@link #setLambda(double) &lambda;} ||x||<sub>1</sub>. It requires the
 * function and it's gradient to work. <br>
 * <br>
 * See:<br>
 * <ul>
 * <li>Gong, P., & Ye, J. (2015). <i>A Modified Orthant-Wise Limited Memory
 * Quasi-Newton Method with Convergence Analysis</i>. In The 32nd International
 * Conference on Machine Learning (Vol. 37).</li>
 * <li>Andrew, G., & Gao, J. (2007). <i>Scalable training of L1 -regularized
 * log-linear models</i>. In Proceedings of the 24th international conference on
 * Machine learning - ICML ’07 (pp. 33–40). New York, New York, USA: ACM Press.
 * doi:10.1145/1273496.1273501</li>
 * </ul>
 *
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class ModifiedOWLQN implements Optimizer2 {

  private static final double DEFAULT_EPS = 1e-12;
  private static final double DEFAULT_ALPHA_0 = 1;
  private static final double DEFAULT_BETA = 0.2;
  private static final double DEFAULT_GAMMA = 1e-2;
  private int m = 10;
  private double lambda;
  private Vec lambdaMultipler = null;
  private double eps = DEFAULT_EPS;
  private double alpha_0 = DEFAULT_ALPHA_0;
  private double beta = DEFAULT_BETA;
  private double gamma = DEFAULT_GAMMA;
  private int maxIterations = 500;

  /**
   * Creates a new mOWL-QN optimizer with no regularization penalty
   */
  public ModifiedOWLQN() {
    this(0.0);
  }

  /**
   * Creates a new mOWL-QN optimizer
   *
   * @param lambda
   *          the regularization penalty to use
   */
  public ModifiedOWLQN(final double lambda) {
    setLambda(lambda);
  }

  /**
   * copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  protected ModifiedOWLQN(final ModifiedOWLQN toCopy) {
    this(toCopy.lambda);
    if (toCopy.lambdaMultipler != null) {
      lambdaMultipler = toCopy.lambdaMultipler.clone();
    }
    eps = toCopy.eps;
    m = toCopy.m;
    alpha_0 = toCopy.alpha_0;
    beta = toCopy.beta;
    gamma = toCopy.gamma;
    maxIterations = toCopy.maxIterations;
  }

  @Override
  public ModifiedOWLQN clone() {
    return new ModifiedOWLQN(this);
  }

  public double getBeta() {
    return beta;
  }

  public double getEps() {
    return eps;
  }

  private double getL1Penalty(final Vec w, final Vec lambdaMul) {
    if (lambda <= 0) {
      return 0;
    }
    double pen = 0;
    for (final IndexValue iv : w) {
      pen += lambda * lambdaMul.get(iv.getIndex()) * abs(iv.getValue());
    }
    return pen;
  }

  public Vec getLambdaMultipler() {
    return lambdaMultipler;
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

  @Override
  public void optimize(final double tolerance, final Vec w, final Vec x0, final Function f, final FunctionVec fp,
      final FunctionVec fpp) {
    optimize(tolerance, w, x0, f, fp, fpp, null);
  }

  @Override
  public void optimize(final double tolerance, final Vec w, final Vec x0, final Function f, final FunctionVec fp,
      final FunctionVec fpp, final ExecutorService ex) {
    // Algorithm 2 mOWL-QN: modified Orthant-Wise Limited memory Quasi-Newton

    Vec lambdaMul = lambdaMultipler;
    if (lambdaMultipler == null) {
      lambdaMul = new ConstantVector(1.0, x0.length());
    }

    final Vec x_cur = x0.clone();

    Vec x_grad = x0.clone();
    Vec x_gradNext = x0.clone();
    final Vec x_grad_diff = x0.clone();
    /**
     * This value is where <> f(x) lives
     */
    final Vec v_k = x0.clone();
    final Vec d_k = x0.clone();
    final Vec p_k = x0.clone();
    final Vec x_alpha = x0.clone();
    /**
     * Difference between x_alpha and x_cur
     */
    final Vec x_diff = x0.clone();

    // history for implicit H
    final List<Double> Rho = new DoubleList(m);
    final List<Vec> S = new ArrayList<Vec>(m);
    final List<Vec> Y = new ArrayList<Vec>(m);
    final double[] alphas = new double[m];

    double f_x = ex != null && f instanceof FunctionP ? ((FunctionP) f).f(x_cur, ex) : f.f(x_cur);
    f_x += getL1Penalty(x_cur, lambdaMul);
    x_grad = ex != null ? fp.f(x_cur, x_grad, ex) : fp.f(x_cur, x_grad);

    // 2: for k = 0 to maxiter do
    for (int k = 0; k < maxIterations; k++) {
      double v_k_norm = 0;
      // 3: Compute v_k ← - <> f(xk)

      for (int i = 0; i < x_grad.length(); i++) {
        final double x_i = x_cur.get(i);
        final double l_i = x_grad.get(i);
        final double lambda_i = lambda * lambdaMul.get(i);
        double newVal;
        if (x_i > 0) {
          newVal = l_i + lambda_i;
        } else if (x_i < 0) {
          newVal = l_i - lambda_i;
        } else if (l_i + lambda_i < 0) {// x_i == 0 is implicit
          newVal = l_i + lambda_i;
        } else if (l_i - lambda_i > 0) {// x_i == 0 is implicit
          newVal = l_i - lambda_i;
        } else {
          newVal = 0;
        }

        v_k.set(i, -newVal);
        v_k_norm += newVal * newVal;
      }
      v_k_norm = Math.sqrt(v_k_norm);

      // Ik = {i ∈ {1, · · · ,n} : 0 < |x^k_i | ≤ ϵk,xk i vk i < 0}, where ϵk =
      // min(∥vk∥, ϵ);
      // we only really need to know if the set I_k is empty or not, the
      // indicies are never used
      final double eps_k = Math.min(v_k_norm, eps);

      boolean doGDstep = false;
      for (int i = 0; i < v_k.length() && !doGDstep; i++) {
        final double x_i = x_cur.get(i);
        final double v_i = v_k.get(i);
        final boolean isInI = 0 < abs(x_i) && abs(x_i) < eps_k && x_i * v_i < 0;
        if (isInI) {
          doGDstep = true;
        }
      }

      // 5: Initialize α←α0;
      double alpha = alpha_0;

      double f_x_alpha = 0;// objective value for new x

      if (!doGDstep) // 6:if Ik = ∅ then (QN-step)
      {
        // 8: Compute dk ←Hkvk using L-BFGS with S, Y ;
        LBFGS.twoLoopHp(v_k, Rho, S, Y, d_k, alphas);

        // 9: Alignment: pk ←π(dk;vk);
        for (int i = 0; i < p_k.length(); i++) {
          if (Math.signum(d_k.get(i)) == Math.signum(v_k.get(i))) {
            p_k.set(i, d_k.get(i));
          } else {
            p_k.set(i, 0.0);
          }
        }

        // 10: while Eq. (7) is not satisfied do
        final double rightSideMainTerm = gamma * v_k.dot(d_k);

        alpha /= beta;// so when we multiply below we get the correct startng
                      // value
        do {
          // 11: α←αβ;
          alpha *= beta;
          // 12: x^k(α)←π(x^k +α p^k; ξ^k);
          x_cur.copyTo(x_alpha);
          x_alpha.mutableSubtract(-alpha, p_k);
          // projection step
          for (int i = 0; i < p_k.length(); i++) {
            final double x_i = x_cur.get(i);
            final double v_i = v_k.get(i);
            final double toUse = x_i != 0 ? x_i : v_i;
            if (Math.signum(x_alpha.get(i)) != Math.signum(toUse)) {
              x_alpha.set(i, 0.0);
            }

          }
          f_x_alpha = ex != null && f instanceof FunctionP ? ((FunctionP) f).f(x_alpha, ex) : f.f(x_alpha);
          f_x_alpha += getL1Penalty(x_alpha, lambdaMul);
        } while (f_x_alpha > f_x - alpha * rightSideMainTerm);

        x_alpha.copyTo(x_diff);
        x_diff.mutableSubtract(x_cur);
      } else// (GD-step)
      {
        alpha /= beta;
        do {
          alpha *= beta;

          /*
           * see section 2.3 of below to solve problem Gong, P., Zhang, C., Lu,
           * Z., Huang, J., & Ye, J. (2013). A general iterative shrinkage and
           * thresholding algorithm for non-convex regularized optimization
           * problems. International Conference on Machine Learning, 28, 37–45.
           * Retrieved from http://arxiv.org/abs/1303.4434
           *
           */
          // first use def u(k) = w(k) − ∇l(w)/t , where t = 1/alpha
          x_grad.copyTo(x_alpha);
          x_alpha.mutableMultiply(-alpha);
          x_alpha.mutableAdd(x_cur);
          // x_alpha noew has the value of u(k)
          // we can now modify it into the correct solution using
          // w^(k+1) = sign(u)max(0, |u|−λ/t)
          for (int i = 0; i < x_alpha.length(); i++) {
            final double u_i = x_alpha.get(i);
            final double lambda_i = lambda * lambdaMul.get(i);
            x_alpha.set(i, signum(u_i) * max(0, abs(u_i) - lambda_i * alpha));
          }
          x_alpha.copyTo(x_diff);
          x_diff.mutableSubtract(x_cur);

          f_x_alpha = ex != null && f instanceof FunctionP ? ((FunctionP) f).f(x_alpha, ex) : f.f(x_alpha);
          f_x_alpha += getL1Penalty(x_alpha, lambdaMul);
        } while (f_x_alpha > f_x - gamma / (2 * alpha) * x_diff.dot(x_diff));// eq(8)
                                                                             // f(x^k(α))
                                                                             // ≤
                                                                             // f(x^k)−
                                                                             // γ/(2α)
                                                                             // ||
                                                                             // x^k(α)−x^k||^2
      }

      // update history
      S.add(0, x_diff.clone());

      x_gradNext = ex != null ? fp.f(x_alpha, x_gradNext, ex) : fp.f(x_alpha, x_gradNext);

      // convergence check
      double maxGrad = 0;
      for (int i = 0; i < x_gradNext.length(); i++) {
        maxGrad = max(maxGrad, abs(x_gradNext.get(i)));
      }

      if (maxGrad < tolerance || f_x < tolerance || x_diff.pNorm(1) < tolerance) {
        break;
      }

      x_gradNext.copyTo(x_grad_diff);
      x_grad_diff.mutableSubtract(x_grad);

      Y.add(0, x_grad_diff.clone());

      Rho.add(0, 1 / x_diff.dot(x_grad_diff));
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

      // prepr for next iterations
      f_x = f_x_alpha;
      x_alpha.copyTo(x_cur);
      x_gradNext.copyTo(x_grad);
    }

    x_cur.copyTo(w);
  }

  /**
   * Sets the shrinkage term used for the line search.
   *
   * @param beta
   *          the line search shrinkage term
   */
  public void setBeta(final double beta) {
    if (beta <= 0 || beta >= 1 || Double.isNaN(beta)) {
      throw new IllegalArgumentException("shrinkage term must be in (0, 1), not " + beta);
    }
    this.beta = beta;
  }

  /**
   * Sets the epsilon term that helps control when the gradient descent step is
   * taken instead of the normal Quasi-Newton step. Larger values cause more GD
   * steps. You shouldn't need to alter this variable
   *
   * @param eps
   *          tolerance term for GD steps
   */
  public void setEps(final double eps) {
    if (eps < 0 || Double.isInfinite(eps) || Double.isNaN(eps)) {
      throw new IllegalArgumentException("eps must be non-negative, not " + eps);
    }
    this.eps = eps;
  }

  /**
   * Sets the regularization term for the optimizer
   *
   * @param lambda
   *          the regularization penalty
   */
  public void setLambda(final double lambda) {
    if (lambda < 0 || Double.isInfinite(lambda) || Double.isNaN(lambda)) {
      throw new IllegalArgumentException("lambda must be non-negative, not " + lambda);
    }
    this.lambda = lambda;
  }

  /**
   * This method sets a vector that will contain a separate multiplier for
   * {@link #setLambda(double) lambda} for each dimension of the problem. This
   * allows for each dimension to have a different regularization penalty.<br>
   * <br>
   * If set to {@code null}, all dimensions will simply use &lambda; as their
   * regularization value.
   *
   * @param lambdaMultipler
   *          the per-dimension regularization multiplier, or {@code null}.
   */
  public void setLambdaMultipler(final Vec lambdaMultipler) {
    this.lambdaMultipler = lambdaMultipler;
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
