package jsat.math.optimization;

import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionP;
import jsat.math.FunctionVec;

/**
 * An implementation of Backtraking line search using the Armijo rule. 
 * The search for alpha is done by quadratic and cubic interpolation without 
 * using any derivative evaluations. 
 * @author Edward Raff
 */
public class BacktrackingArmijoLineSearch implements LineSearch
{
    private double rho;
    private double c1;

    /**
     * Creates a new Backtracking line search 
     */
    public BacktrackingArmijoLineSearch()
    {
        this(0.5, 1e-1);
    }

    /**
     * Creates a new Backtracking line search object
     * @param rho constant to decrease alpha by in (0, 1) when interpolation is 
     * not possible
     * @param c1 the <i>sufficient decrease condition</i>  condition constant in
     * (0, 1/2)
     */
    public BacktrackingArmijoLineSearch(double rho, double c1)
    {
        if(!(rho > 0 && rho < 1))
            throw new IllegalArgumentException("rho must be in (0,1), not " + rho);
        this.rho = rho;
        setC1(c1);
    }
    
    /**
     * Sets the constant used for the <i>sufficient decrease condition</i> 
     * f(x+&alpha; p) &le; f(x) + c<sub>1</sub> &alpha; p<sup>T</sup>&nabla;f(x)
     * @param c1 the <i>sufficient decrease condition</i> 
     */
    public void setC1(double c1)
    {
        if(c1 <= 0 || c1 >= 0.5)
            throw new IllegalArgumentException("c1 must be in (0, 1/2) not " + c1);
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
        
        double alpha = alpha_max;
        if(x_alpha_pk == null)
            x_alpha_pk = x_k.clone();
        else
            x_k.copyTo(x_alpha_pk);
        x_alpha_pk.mutableAdd(alpha, p_k);
        double f_xap = (ex != null && f instanceof FunctionP) ? ((FunctionP)f).f(x_alpha_pk, ex): f.f(x_alpha_pk);
        if(fxApRet != null)
            fxApRet[0] = f_xap;
        double oldAlpha = 0;
        double oldF_xap = f_x;
        
        while (f_xap > f_x + c1 * alpha * gradP)//we return start if its already good
        {
            final double tooSmall = 0.1*alpha;
            final double tooLarge = 0.9*alpha;
            //see INTERPOLATION section of chapter 3.5
          //XXX double compare.
            if(alpha == alpha_max)//quadratic interpolation
            {
                double alphaCandidate = -gradP*oldAlpha*oldAlpha/(2*(f_xap-f_x-gradP*oldAlpha));
                oldAlpha = alpha;
                if(alphaCandidate < tooSmall || alphaCandidate > tooLarge || Double.isNaN(alphaCandidate))
                {
                    alpha = rho*oldAlpha;
                }
                else
                {
                    alpha = alphaCandidate;
                }
            }
            else//cubic interpoation
            {
                //g = φ(α1)−φ(0)−φ'(0)α1
                double g = f_xap-f_x-gradP*alpha;
                //h = φ(α0) − φ(0) − φ'(0)α0
                double h = oldF_xap-f_x-gradP*oldAlpha;
                
                double a0Sqrd = oldAlpha*oldAlpha;
                double a1Sqrd = alpha*alpha;
                
                double a = a0Sqrd*g-a1Sqrd*h;
                a /= (a0Sqrd*a1Sqrd*(alpha-oldAlpha));
                double b = -a0Sqrd*oldAlpha*g+a1Sqrd*alpha*h;
                b /= (a0Sqrd*a1Sqrd*(alpha-oldAlpha));
                
                double alphaCandidate = (-b + Math.sqrt(b*b-3*a*gradP))/(3*a);
                oldAlpha = alpha;
                if(alphaCandidate < tooSmall || alphaCandidate > tooLarge || Double.isNaN(alphaCandidate))
                {
                    alpha = rho*oldAlpha;
                }
                else
                {
                    alpha = alphaCandidate;
                }
                
            }

            if(alpha < 1e-20)
                return oldAlpha;
            x_alpha_pk.mutableSubtract(oldAlpha - alpha, p_k);
            oldF_xap = f_xap;
            f_xap = (ex != null && f instanceof FunctionP) ? ((FunctionP)f).f(x_alpha_pk, ex): f.f(x_alpha_pk);
            if(fxApRet != null)
                fxApRet[0] = f_xap;
        }
        
        return alpha;
    }

    @Override
    public boolean updatesGrad()
    {
        return false;
    }

    @Override
    public BacktrackingArmijoLineSearch clone()
    {
        return new BacktrackingArmijoLineSearch(rho, c1);
    }
}
