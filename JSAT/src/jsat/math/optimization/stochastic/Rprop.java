/*
 * Copyright (C) 2015 Edward Raff
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
package jsat.math.optimization.stochastic;

import java.util.Arrays;
import jsat.linear.IndexValue;
import jsat.linear.Vec;

/**
 * The Rprop algorithm provides adaptive learning rates using only first order
 * information. Rprop works best with the true gradient, and may not work well
 * when using stochastic gradients.<br>
 * <br>
 * See: Riedmiller, M., & Braun, H. (1993). <i>A direct adaptive method for
 * faster backpropagation learning: the RPROP algorithm</i>. IEEE International
 * Conference on Neural Networks, 1(3), 586–591. doi:10.1109/ICNN.1993.298623
 *
 * @author Edward Raff
 */
public class Rprop implements GradientUpdater
{
    private double eta_pos = 1.2;
    private double eta_neg = 0.5;
    private double eta_start = 0.1;
    private double eta_max = 50;
    private double eta_min = 1e-6;
    
    
    /**
     * holds what would be w^(t-1)
     */
    private double[] prev_w;
    private double[] prev_grad;
    private double[] cur_eta;
    private double prev_grad_bias;
    private double cur_eta_bias;
    private double prev_bias;

    /**
     * Creates a new Rprop instance for gradient updating
     */
    public Rprop()
    {
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public Rprop(Rprop toCopy)
    {
        if (toCopy.prev_grad != null)
            this.prev_grad = Arrays.copyOf(toCopy.prev_grad, toCopy.prev_grad.length);
        if (toCopy.cur_eta != null)
            this.cur_eta = Arrays.copyOf(toCopy.cur_eta, toCopy.cur_eta.length);
        if (toCopy.prev_w != null)
            this.prev_w = Arrays.copyOf(toCopy.prev_w, toCopy.prev_w.length);
        this.prev_grad_bias = toCopy.prev_grad_bias;
        this.cur_eta_bias = toCopy.cur_eta_bias;
        this.prev_bias = toCopy.prev_bias;
    }
    
    

    @Override
    public void update(Vec w, Vec grad, double eta)
    {
        update(w, grad, eta, 0, 0);
    }

    @Override
    public double update(Vec w, Vec grad, double eta, double bias, double biasGrad)
    {
        for(IndexValue iv : grad)
        {
            final int i = iv.getIndex();
            final double g_i = iv.getValue();
            final double g_prev = prev_grad[i];
            final double w_i = w.get(i);
            prev_grad[i] = g_i;
            final double sign_g_i = Math.signum(g_i);
            final double sign_g_prev = Math.signum(g_prev);
            if(sign_g_i == 0 || sign_g_prev == 0)
            {
                double eta_i = cur_eta[i];

                    w.increment(i, -sign_g_i*eta_i*eta);
            }
            else if(sign_g_i == sign_g_prev)
            {
                double eta_i = cur_eta[i] = Math.min(cur_eta[i]*eta_pos, eta_max);

                    w.increment(i, -sign_g_i*eta_i*eta);
            }
            else//not equal, sign change
            {
                double eta_i = cur_eta[i] = Math.max(cur_eta[i]*eta_neg, eta_min);
                
                w.increment(i, -prev_w[i]*eta_i*eta);
                prev_grad[i] = 0;
            }
            
            prev_w[i] = w_i;
        }
        
        //and again for the bias term
        if(bias != 0 && biasGrad != 0)
        {
            double toRet;
            final double g_i = biasGrad;
            final double g_prev = prev_grad_bias;
            final double w_i = bias;
            prev_grad_bias = g_i;
            final double sign_g_i = Math.signum(g_i);
            final double sign_g_prev = Math.signum(g_prev);
            if(sign_g_i == 0 || sign_g_prev == 0)
            {
                double eta_i = cur_eta_bias;
                toRet = sign_g_i*eta_i;
            }
            else if(sign_g_i == sign_g_prev)
            {
                double eta_i = cur_eta_bias = Math.min(cur_eta_bias*eta_pos, eta_max);
                toRet = sign_g_i*eta_i;
            }
            else//not equal, sign change
            {
                double eta_i = cur_eta_bias = Math.max(cur_eta_bias*eta_neg, eta_min);
                
                prev_grad_bias = 0;
                toRet = -prev_bias*eta_i;
            }
            
            prev_bias = w_i;
            
            return toRet*eta;
        }
        
        return 0;
    }

    @Override
    public void setup(int d)
    {
        prev_grad = new double[d];
        cur_eta = new double[d];
        Arrays.fill(cur_eta, eta_start);
        prev_w = new double[d];
        
        cur_eta_bias = eta_start;
        prev_grad_bias = 0;
        prev_bias = 0;
    }

    @Override
    public Rprop clone()
    {
        return new Rprop(this);
    }
    
}
