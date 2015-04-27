package jsat.math.optimization;

import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionVec;
import static java.lang.Math.*;
import java.util.concurrent.ExecutorService;
import jsat.math.*;

/**
 * An implementation of the Wolfe Line Search algorithm described by Nocedal and
 * Wright in <i>Numerical Optimization</i> (2nd edition) on pages 59-63. 
 * 
 * @author Edward Raff
 */
public class WolfeNWLineSearch implements LineSearch
{
    //default values that make setting in the constructor simple (shouldn't actually use)
    private double c1 = Math.nextUp(0), c2 = Math.nextAfter(1, Double.NEGATIVE_INFINITY);
    
    /**
     * Creates a new Wolfe line search with {@link #setC1(double) } set to 
     * {@code 1e-4} and {@link #setC2(double) } to {@code 0.9}
     */
    public WolfeNWLineSearch()
    {
        this(1e-4, 0.9);
    }

    /**
     * Creates a new Wolfe line search
     * @param c1 the <i>sufficient decrease condition</i> constant
     * @param c2 the <i>curvature condition</i> constant
     */
    public WolfeNWLineSearch(double c1, double c2)
    {
        setC1(c1);
        setC2(c2);
    }
    
    private AlphaInit initMethod = AlphaInit.METHOD1;
    
    double alpha_prev = -1, f_x_prev = Double.NaN, gradP_prev = Double.NaN;

    public enum AlphaInit
    {
        /**
         * Initializes the new &alpha; value via &alpha;<sub>prev</sub> 
         * &nabla;f(x<sub>prev</sub>)<sup>T</sup>p<sub>prev</sub>/
         * &nabla;f(x<sub>cur</sub>)<sup>T</sup>p<sub>cur</sub>
         */
        METHOD1, 
        /**
         * Initializes the new &alpha; value via 
         * 2( f(x<sub>cur</sub>)-f(x<sub>prev</sub>))/&phi;'(0)
         */
        METHOD2
    }

    /**
     * Sets the constant used for the <i>sufficient decrease condition</i> 
     * f(x+&alpha; p) &le; f(x) + c<sub>1</sub> &alpha; p<sup>T</sup>&nabla;f(x)
     * <br>
     * <br>
     * This value must always be less than {@link #setC2(double) }
     * @param c1 the <i>sufficient decrease condition</i> 
     */
    public void setC1(double c1)
    {
        if(c1 <= 0)
            throw new IllegalArgumentException("c1 must be greater than 0, not " + c1);
        else if(c1 >= c2)
            throw new IllegalArgumentException("c1 must be less than c2");
        this.c1 = c1;
    }

    /**
     * Returns the <i>sufficient decrease condition</i> constant
     * @return the <i>sufficient decrease condition</i> constant
     */
    public double getC1()
    {
        return c1;
    }
    
    /**
     * Sets the constant used for the <i>curvature condition</i> 
     * p<sup>T</sup> &nabla;f(x+&alpha; p) &ge; c<sub>2</sub> p<sup>T</sup>&nabla;f(x)
     * @param c2 the <i>curvature condition</i> constant
     */
    public void setC2(double c2)
    {
        if(c2 >= 1)
            throw new IllegalArgumentException("c2 must be less than 1, not " + c2);
        else if(c2 <= c1)
            throw new IllegalArgumentException("c2 must be greater than c1");
        this.c2 = c2;
    }

    /**
     * Returns the <i>curvature condition</i> constant
     * @return the <i>curvature condition</i> constant
     */
    public double getC2()
    {
        return c2;
    }
    
    @Override
    public double lineSearch(double alpha_max, Vec x_k, Vec x_grad, Vec p_k, Function f, FunctionVec fp, double f_x, double gradP, Vec x_alpha_pk, double[] fxApRet, Vec grad_x_alpha_pk)
    {
        return lineSearch(alpha_max, x_k, x_grad, p_k, f, fp, f_x, gradP, x_alpha_pk, fxApRet, grad_x_alpha_pk, null);
    }
    
    @Override
    public double lineSearch(double alpha_max, Vec x_k, Vec x_grad, Vec p_k, Function f, FunctionVec fp, double f_x, double gradP, Vec x_alpha_pk, double[] fxApRet, Vec grad_x_alpha_pk, ExecutorService ex)
    {
        if(Double.isNaN(f_x))
            f_x = (ex != null && f instanceof FunctionP) ? ((FunctionP)f).f(x_k, ex): f.f(x_k);
        if(Double.isNaN(gradP))
            gradP = x_grad.dot(p_k);
        final double phi0 = f_x, phi0P = gradP;
        
        double alpha_cur = 1;
        if(!Double.isNaN(gradP_prev) && initMethod == AlphaInit.METHOD1)
        {
            alpha_cur = alpha_prev*gradP_prev/gradP;
        }
        else if(!Double.isNaN(f_x_prev) && initMethod == AlphaInit.METHOD2)
        {
            alpha_cur = 2*(f_x-f_x_prev)/phi0P;
            alpha_cur = min(1, 1.01*(alpha_cur));
        }
        alpha_cur = max(alpha_cur, 1e-13);
        //2.5.13 from OPTIMIZATION THEORY AND  METHODS Nonlinear Programming
        alpha_prev = 0;
        
        double phi_prev = phi0;
        double phi_prevP = phi0P;
        
        double valToUse = 0;
        
        x_k.copyTo(x_alpha_pk);
        for(int iter = 1; iter <= 10 && valToUse == 0; iter++)
        {
            //Evaluate φ(αi );
            x_alpha_pk.mutableAdd(alpha_cur-alpha_prev, p_k);
            double phi_cur = (ex != null && f instanceof FunctionP) ? ((FunctionP)f).f(x_alpha_pk, ex): f.f(x_alpha_pk);
            if(fxApRet != null)
                fxApRet[0] = phi_cur;
            double phi_curP = (ex != null) ? fp.f(x_alpha_pk, grad_x_alpha_pk, ex).dot(p_k) : fp.f(x_alpha_pk, grad_x_alpha_pk).dot(p_k);//computed early b/c used in interpolation in zoom
            //if φ(αi)>φ(0)+c1 αi φ'(0) or[φ(αi)≥φ(αi−1) and i >1]
            if(phi_cur > phi0 + c1*alpha_cur*phi0P || (phi_cur >= phi_prev && iter > 1) )
            {
                //α∗ ←zoom(αi−1,αi) and stop;
                valToUse = zoom(alpha_prev, alpha_cur, phi_prev, phi_cur, phi_prevP, phi_curP, phi0, phi0P, x_k, x_alpha_pk, p_k, f, fp, fxApRet, grad_x_alpha_pk, ex);
                break;
            }
            //Evaluate φ'(αi );
            
            //if |φ'(αi )| ≤ −c2φ'(0)
            if(abs(phi_curP) <= -c2*phi0P)
            {
                valToUse = alpha_cur;//set α∗ ← αi and stop;
                break;
            }
            //if φ'(αi ) ≥ 0
            if(phi_curP >= 0)
            {
                //set α∗ ←zoom(αi,αi−1) and stop;
                valToUse = zoom(alpha_cur, alpha_prev, phi_cur, phi_prev, phi_curP, phi_prevP, phi0, phi0P, x_k, x_alpha_pk, p_k, f, fp, fxApRet, grad_x_alpha_pk, ex);
                break;
            }
            //Choose αi+1 ∈(αi,αmax);
            ///err, just double it?
            alpha_prev = alpha_cur;
            phi_prev = phi_cur;
            phi_prevP = phi_curP;
            
            alpha_cur *= 2;
            
            if(alpha_cur >= alpha_max)//hit the limit
            {
                valToUse = alpha_max;
                break;
            }
        }
        
        alpha_prev = valToUse;
        f_x_prev = f_x;
        gradP_prev = gradP;

        return valToUse;
    }
    
    /**
     *
     *
     * @param alphaLow the value of alphaLow
     * @param alphaHi the value of alphaHi
     * @param phi_alphaLow the value of phi_alphaLow
     * @param phi_alphaHigh the value of phi_alphaHigh
     * @param phi_alphaLowP the value of phi_alphaLowP
     * @param phi_alphaHighP the value of phi_alphaHighP
     * @param phi0 the value of phi0
     * @param phi0P the value of phi0P
     * @param x the value of x
     * @param x_alpha_p the value of x_alpha_p
     * @param p the value of p
     * @param f the value of f
     * @param fp the value of fp
     * @param fxApRet the value of fxApRet
     * @param grad_x_alpha_pk the value of grad_x_alpha_pk
     * @param ex the value of ex
     * @return the double
     */
    private double zoom(double alphaLow, double alphaHi, double phi_alphaLow, double phi_alphaHigh, double phi_alphaLowP, double phi_alphaHighP, double phi0, double phi0P, Vec x, Vec x_alpha_p, Vec p, Function f, FunctionVec fp, double[] fxApRet, Vec grad_x_alpha_pk, ExecutorService ex)
    {
        double alpha_j = alphaLow;
        for(int iter = 0; iter < 10; iter++)
        {
            
            //try cubic interp eq  (3.59)
            {
                double d1 = phi_alphaLowP+phi_alphaHighP-3*(phi_alphaLow-phi_alphaHigh)/(alphaLow-alphaHi);
                double d2 = signum(alphaHi-alphaLow)*pow(d1*d1-phi_alphaLowP*phi_alphaHighP, 0.5);
                alpha_j = alphaHi-(alphaHi-alphaLow)*(phi_alphaHighP+d2-d1)/(phi_alphaHighP-phi_alphaLowP+2*d2);
            }
            //check if we were too close to the edge
            if(alpha_j-(alphaHi-alphaLow)/2*0.1 < alphaLow || alpha_j > alphaHi*0.9)
                alpha_j = min(alphaLow, alphaHi) + abs(alphaHi-alphaLow)/2;
            x.copyTo(x_alpha_p);
            x_alpha_p.mutableAdd(alpha_j, p);
            
            //Evaluate φ(αj );
            double phi_j = (ex != null && f instanceof FunctionP) ? ((FunctionP)f).f(x_alpha_p, ex): f.f(x_alpha_p);
            if(fxApRet != null)
                fxApRet[0] = phi_j;
            double phi_jP = (ex != null) ? fp.f(x_alpha_p, grad_x_alpha_pk, ex).dot(p) : fp.f(x_alpha_p, grad_x_alpha_pk).dot(p);//computed early
            //if φ(αj ) > φ(0) + c1αj φ'(0) or φ(αj ) ≥ φ(αlo)
            if(phi_j > phi0 + c1*alpha_j*phi0 || phi_j >= phi_alphaLow)
            {
                //αhi ←αj;
                alphaHi = alpha_j;
                phi_alphaHigh = phi_j;
                phi_alphaHighP = phi_jP;
            }
            else
            {
                //Evaluate φ'(αj );
                
                //if |φ'(αj )| ≤ −c2φ'(0)
                if(abs(phi_jP) <= c2*phi0P)
                    return alpha_j;//Set α∗ ← αj and stop;
                //if φ'(αj)(αhi −αlo)≥0
                if(phi_jP*(alphaHi-alphaLow) >= 0)
                {
                    //αhi ← αlo;
                    alphaHi = alphaLow;
                    phi_alphaHigh = phi_alphaLow;
                    phi_alphaHighP = phi_alphaLowP;
                }
                //αlo ←αj;
                alphaLow = alpha_j;
                phi_alphaLow = phi_j;
                phi_alphaLowP = phi_jP;
            }
            
        }
        return alpha_j;
    }

    @Override
    public boolean updatesGrad()
    {
        return true;
    }
    
    @Override
    public WolfeNWLineSearch clone()
    {
        WolfeNWLineSearch clone = new WolfeNWLineSearch(c1, c2);
        clone.initMethod = this.initMethod;
        clone.alpha_prev = this.alpha_prev;
        clone.f_x_prev = this.f_x_prev;
        clone.gradP_prev = this.gradP_prev;
        return clone;
    }
}
