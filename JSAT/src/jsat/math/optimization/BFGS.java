package jsat.math.optimization;

import java.util.concurrent.ExecutorService;
import jsat.linear.*;
import jsat.math.*;

/**
 * Implementation of the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm for 
 * function minimization. For {@code n} dimensional problems it requires 
 * <i>O(n<sup>2</sup>)</i> work per iteration and uses first order information 
 * to approximate the Hessian. 
 * 
 * @author Edward Raff
 */
public class BFGS implements Optimizer2 
{
    private LineSearch lineSearch;
    private int maxIterations;
    private boolean inftNormCriterion = true;

    /**
     * Creates a new BFGS optimization object that uses a maximum of 250 
     * iterations and a {@link BacktrackingArmijoLineSearch backtracking} line 
     * search. 
     */
    public BFGS()
    {
        this(250, new BacktrackingArmijoLineSearch());
    }

    /**
     * Creates a new BFGS optimization object
     * @param maxIterations the maximum number of iterations to allow before 
     * stopping
     * @param lineSearch the line search method to use on updates
     */
    public BFGS(int maxIterations, LineSearch lineSearch)
    {
        setMaximumIterations(maxIterations);
        setLineSearch(lineSearch);
    }
    
    @Override
    public void optimize(double tolerance, Vec w, Vec x0, Function f, FunctionVec fp, FunctionVec fpp)
    {
        optimize(tolerance, w, x0, f, fp, fpp, null);
    }
    
    @Override
    public void optimize(double tolerance, Vec w, Vec x0, Function f, FunctionVec fp, FunctionVec fpp, ExecutorService ex)
    {
        LineSearch search = lineSearch.clone();
        
        Matrix H = Matrix.eye(x0.length());
        Vec x_prev = x0.clone();
        Vec x_cur = x0.clone();
        final double[] f_xVal = new double[1];//store place for f_x
        
        //graidnet
        Vec x_grad = x0.clone();
        x_grad.zeroOut();
        Vec x_gradPrev = x_grad.clone();
        //p_l
        Vec p_k = x_grad.clone();
        Vec s_k = x_grad.clone();
        Vec y_k = x_grad.clone();
        
        f_xVal[0] = (ex != null && f instanceof FunctionP) ? ((FunctionP)f).f(x_cur, ex) : f.f(x_cur);
        x_grad = (ex != null) ? fp.f(x_cur, x_grad, ex) : fp.f(x_cur, x_grad);
       
        int iter = 0;
        while(gradConvgHelper(x_grad) > tolerance && iter < maxIterations)
        {
            iter++;
            p_k.zeroOut();
            H.multiply(x_grad, -1, p_k);//p_k = −H_k ∇f_k; (6.18)
            
            //Set x_k+1 = x_k + α_k p_k where α_k is computed from a line search
            x_cur.copyTo(x_prev);
            x_grad.copyTo(x_gradPrev);
            
            double alpha_k = search.lineSearch(1.0, x_prev, x_gradPrev, p_k, f, fp, f_xVal[0], x_gradPrev.dot(p_k), x_cur, f_xVal, x_grad, ex);
            if(alpha_k < 1e-12 && iter > 5)//if we are making near epsilon steps consider it done
                break;

            if(!search.updatesGrad())
                if (ex != null)
                    fp.f(x_cur, x_grad, ex);
                else
                    fp.f(x_cur, x_grad);
            
            //Define s_k =x_k+1 −x_k and y_k = ∇f_k+1 −∇f_k;
            x_cur.copyTo(s_k);
            s_k.mutableSubtract(x_prev);
            
            x_grad.copyTo(y_k);
            y_k.mutableSubtract(x_gradPrev);
            //Compute H_k+1 by means of (6.17);
            
            double skyk = s_k.dot(y_k);
            if(skyk <= 0)
            {
                H.zeroOut();
                for(int i = 0; i < H.rows(); i++)
                    H.set(i, i, 1);
                continue;
            }
            if(iter == 0 && skyk > 1e-12)
                for(int i = 0; i < H.rows(); i++)
                    H.set(i, i, skyk/y_k.dot(y_k));
            
            /*
             * From "A Perfect Example for The BFGS Method" equation 1.5
             * aamath:  H_(k+1)=H_k-(s_k*y_k^T*H_k+H_k*y_k*s_k^T)/(s_k^T*y_k)+(1+(y_k^T*H_k*y_k)/(s_k^T*y_k))*((s_k*s_k^T)/(s_k^T*y_k))
             * 
             *                   T             T   /     T      \     T
             *               s  y  H  + H  y  s    |    y  H  y | s  s
             *                k  k  k    k  k  k   |     k  k  k|  k  k
             * H      = H  - ------------------- + |1 + --------| -----
             *  k + 1    k           T             |       T    |  T
             *                      s  y           |      s  y  | s  y
             *                       k  k          \       k  k /  k  k
             * 
             * TODO: y_k^T H_k y_k should be just a scalar constant
             * TODO: exploit the symetry of H_k
             */
            
            Vec Hkyk = H.multiply(y_k);
            Vec ykHk = y_k.multiply(H);
            double b = (1+y_k.dot(Hkyk)/skyk)/skyk;//coef for right rank update

            //update
            Matrix.OuterProductUpdate(H, s_k, ykHk, -1/skyk);
            Matrix.OuterProductUpdate(H, Hkyk, s_k, -1/skyk);
            Matrix.OuterProductUpdate(H, s_k, s_k, b);
        }
        
        x_cur.copyTo(w);
    }
    
    /**
     * By default the infinity norm is used to judge convergence. If set to 
     * {@code false}, the 2 norm will be used instead. 
     * @param inftNormCriterion 
     */
    public void setInftNormCriterion(boolean inftNormCriterion)
    {
        this.inftNormCriterion = inftNormCriterion;
    }

    /**
     * Returns whether or not the infinity norm ({@code true}) or 2 norm 
     * ({@code false}) is used to determine convergence. 
     * @return {@code true} if the infinity norm is in use, {@code false} for 
     * the 2 norm
     */
    public boolean isInftNormCriterion()
    {
        return inftNormCriterion;
    }
    
    private double gradConvgHelper(Vec grad)
    {
        if(!inftNormCriterion)
            return grad.pNorm(2);
        double max = 0;
        for(IndexValue iv : grad)
            max = Math.max(max, Math.abs(iv.getValue()));
        return max;
    }

    @Override
    public void setMaximumIterations(int iterations)
    {
        if(iterations < 1)
            throw new IllegalArgumentException("Iterations must be a positive value, not " + iterations);
        this.maxIterations = iterations;
    }

    @Override
    public int getMaximumIterations()
    {
        return maxIterations;
    }
    
    /**
     * Sets the line search method used at each iteration
     * @param lineSearch the line search method used at each iteration
     */
    public void setLineSearch(LineSearch lineSearch)
    {
        this.lineSearch = lineSearch;
    }

    /**
     * Returns the line search method used at each iteration
     * @return the line search method used at each iteration
     */
    public LineSearch getLineSearch()
    {
        return lineSearch;
    }
    
    @Override
    public Optimizer2 clone()
    {
        return new BFGS(maxIterations, lineSearch.clone());
    }
    
}
