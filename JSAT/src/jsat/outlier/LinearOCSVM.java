/*
 * Copyright (C) 2018 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.outlier;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.ScaledVector;
import jsat.linear.Vec;
import jsat.math.optimization.stochastic.AdaGrad;
import jsat.math.optimization.stochastic.GradientUpdater;
import jsat.utils.random.RandomUtil;

/**
 * This class implements the One-Class SVM (OC-SVM) algorithm for outlier
 * detection. This implementation works only in the primal or "linear" space. As
 * such it works best when the data is sparse and high dimensional. If your data
 * is dense and low dimensional, you may get better results by first applying a
 * non-linear transformation to the data.
 *
 * 
 * See: 
 * <ul>
 * <li>Schölkopf, B., Williamson, R., Smola, A., Shawe-Taylor, J., & Platt, J.
 * (1999). <i>Support Vector Method for Novelty Detection</i>. In Advances in
 * Neural Information Processing Systems 12 (pp. 582–588). Denver, CO.</li>
 * <li>Manevitz, L. M., & Yousef, M. (2001). <i>One-class Svms for Document
 * Classification</i>. J. Mach. Learn. Res., 2, 139–154. Retrieved from
 * http://dl.acm.org/citation.cfm?id=944790.944808</li>
 * </ul>
 * 
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class LinearOCSVM implements Outlier
{
    
    private Vec w;
    private double p;
    private int max_epochs = 100;
    private double learningRate = 0.01;
    private double v = 0.05;

    public void setV(double v)
    {
        this.v = v;
    }

    public double getV()
    {
        return v;
    }
    
    
    @Override
    public void fit(DataSet d, boolean parallel)
    {
        
        Random rand = RandomUtil.getRandom();
        List<Vec> X = d.getDataVectors();
        
        int N = X.size();
        w = new ScaledVector(new DenseVector(X.get(0).length()));
        p = 0;
        
        
        GradientUpdater gu = new AdaGrad();
        gu.setup(w.length());
        double cnt = 1/(v);
        
        double prevLoss = Double.POSITIVE_INFINITY;
        double curLoss = Double.POSITIVE_INFINITY;
        for(int epoch = 0; epoch < max_epochs; epoch++)
        {
            Collections.shuffle(X, rand);
            
            prevLoss = curLoss;
            curLoss = 0;
            
            for(int i = 0; i < X.size(); i++)
            {
                Vec x = X.get(i);
                double loss = p - w.dot(x);
                
                double p_delta = -1;
                double x_mul = 0;
                
                if(loss > 0)
                {
                    p_delta += 1*cnt;
                    x_mul = -1*cnt;
                }
                
                curLoss += Math.max(0, loss);
                
                p -= gu.update(w, new ScaledVector(x_mul, x), learningRate, p, p_delta);
                w.mutableMultiply(1-learningRate);
            }
            
//            System.out.println("Epoch " + epoch + " " + curLoss + " " + (curLoss-prevLoss)/N);
            if(Math.abs((curLoss-prevLoss)/N) < 1e-6*v)
                break;//Convergence check
            
        }
    }

    @Override
    public double score(DataPoint x)
    {
        return w.dot(x.getNumericalValues())-p;
    }
    
}
