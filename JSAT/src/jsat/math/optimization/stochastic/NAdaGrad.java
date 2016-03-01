/*
 * Copyright (C) 2016 Edward Raff <Raff.Edward@gmail.com>
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
import jsat.linear.*;

/**
 * Normalized AdaGrad provides an adaptive learning rate for each individual
 * feature, and is mostly scale invariant to the data distribution. NAdaGrad is
 * meant for online stochastic learning where the update is obtained from one
 * dataum at a time, and it relies on the gradient being a scalar multiplication
 * of the training data. If the gradient given us a {@link ScaledVector}, where
 * the base vector is the datum, then NAdaGrad will work. If not the case,
 * NAdaGrad will degenerate into something similar to normal {@link AdaGrad}.
 * <br><br>
 * The current implementation assumes that the bias term is always scaled
 * correctly, and does normal AdaGrad on it.
 * <br>
 * See: Ross, S., Mineiro, P., & Langford, J. (2013). Normalized online
 * learning. In Twenty-Ninth Conference on Uncertainty in Artificial
 * Intelligence. Retrieved from
 * <a href="http://arxiv.org/abs/1305.6646">here</a>
 *
 * @author Edward Raff
 */
public class NAdaGrad implements GradientUpdater
{

    private static final long serialVersionUID = 5138675613579751777L;
    private double[] G;
    private double[] S;
    
    private double N;
    private double biasG;
    private long t;

    /**
     * Creates a new NAdaGrad updater
     */
    public NAdaGrad()
    {
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public NAdaGrad(NAdaGrad toCopy)
    {
        if(toCopy.G != null)
            this.G = Arrays.copyOf(toCopy.G, toCopy.G.length);
        if(toCopy.S != null)
            this.S = Arrays.copyOf(toCopy.S, toCopy.S.length);
        this.biasG = toCopy.biasG;
        this.N = toCopy.N;
        this.t = toCopy.t;
    }
    

    @Override
    public void update(Vec w, Vec grad, double eta)
    {
        update(w, grad, eta, 0, 0);
    }

    @Override
    public double update(Vec w, Vec grad, double eta, double bias, double biasGrad)
    {
        
        if(grad instanceof ScaledVector)
        {
            t++;
            //decompone our gradient back into parts, the multipler and raw datum
            Vec x = ((ScaledVector)grad).getBase();
            
            for(IndexValue iv : x)
            {

                final int indx = iv.getIndex();
                final double abs_x_i = Math.abs(iv.getValue());
                
                if(abs_x_i > S[indx])//(a)
                {
                    w.set(indx, (w.get(indx)*S[indx])/abs_x_i);
                    S[indx] = abs_x_i;
                }
                //skip step (b) for simplicity since grad was already given to us
                //(c)
                N += abs_x_i*abs_x_i/(S[indx]*S[indx]);
            }
            
            double eta_roled = -eta*Math.sqrt(t/(N+1e-6));
            for(IndexValue iv : grad)
            {
                final int indx = iv.getIndex();
                final double grad_i = iv.getValue();
                G[indx] += grad_i*grad_i;
                final double g_ii = G[indx];
                w.increment(indx, eta_roled*grad_i/(S[indx]*Math.sqrt(g_ii)));
                
            }

            double biasUpdate = eta*biasGrad/Math.sqrt(biasG);
            biasG += biasGrad*biasGrad;
            return biasUpdate;
        }
        else//lets degenerate into something at least similar to AdaGrad
        {
            double eta_roled = -eta*Math.sqrt((t+1)/Math.max(N, t+1));
            for(IndexValue iv : grad)
            {
                final int indx = iv.getIndex();
                final double grad_i = iv.getValue();
                G[indx] += grad_i*grad_i;
                final double g_ii = G[indx];
                w.increment(indx, eta_roled*grad_i/(Math.max(S[indx], 1.0)*Math.sqrt(g_ii)));
                
            }

            double biasUpdate = eta*biasGrad/Math.sqrt(biasG);
            biasG += biasGrad*biasGrad;
            return biasUpdate;
        }
    }

    @Override
    public NAdaGrad clone()
    {
        return new NAdaGrad(this);
    }

    @Override
    public void setup(int d)
    {
        G = new double[d];
        S = new double[d];
        biasG = 1;
        t = 0;
    }
    
}
