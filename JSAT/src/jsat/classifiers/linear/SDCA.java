/*
 * Copyright (C) 2017 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.classifiers.linear;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.SimpleWeightVectorModel;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.lossfunctions.LogisticLoss;
import jsat.lossfunctions.LossC;
import jsat.lossfunctions.LossFunc;
import jsat.lossfunctions.LossMC;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.random.RandomUtil;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class SDCA implements Classifier, Parameterized, SimpleWeightVectorModel
{
    private LossFunc loss = new LogisticLoss();
    private boolean useBias = true;
    private double tol = 0.001;
    private double lambda = 1e-4;
    private double alpha = 0.5;
    private int max_epochs = 200;
    private double[] dual_alphas;
    
    private Vec[] ws;
    private double[] bs;
    
    /**
     * Sets whether or not an implicit bias term will be added to the data set
     * @param useBias {@code true} to add an implicit bias term
     */
    public void setUseBias(boolean useBias)
    {
        this.useBias = useBias;
    }

    /**
     * Returns whether or not an implicit bias term is in use
     * @return {@code true} if a bias term is in use
     */
    public boolean isUseBias()
    {
        return useBias;
    }

    /**
     * Sets the regularization term, where larger values indicate a larger 
     * regularization penalty. 
     * 
     * @param lambda the positive regularization term
     */
    @Parameter.WarmParameter(prefLowToHigh = false)
    public void setLambda(double lambda)
    {
        if(lambda <= 0 || Double.isInfinite(lambda) || Double.isNaN(lambda))
            throw new IllegalArgumentException("Regularization term lambda must be a positive value, not " + lambda);
        this.lambda = lambda;
    }

    /**
     * 
     * @return the regularization term
     */
    public double getLambda()
    {
        return lambda;
    }
    
    /**
     * Using &alpha; = 1 corresponds to pure L<sub>1</sub> regularization, and 
     * &alpha; = 0 corresponds to pure L<sub>2</sub> regularization. Any value 
     * in-between is then an Elastic Net regularization.
     * 
     * @param alpha the value in [0, 1] for determining the regularization 
     * penalty's interpolation between pure L<sub>2</sub> and L<sub>1</sub>
     * regularization. 
     */
    public void setAlpha(double alpha)
    {
        if(alpha < 0 || alpha > 1 || Double.isNaN(alpha))
            throw new IllegalArgumentException("alpha must be in [0, 1], not " + alpha);
        
        this.alpha = alpha;
    }

    /***
     * 
     * @return the fraction of weight (in [0, 1]) to apply to L<sub>1</sub>
     * regularization instead of L<sub>2</sub> regularization. 
     */
    public double getAlpha()
    {
        return alpha;
    }
    
    /**
     * Sets the maximum number of training iterations (epochs) for the algorithm. 
     * 
     * @param maxOuterIters the maximum number of outer iterations
     */
    public void setMaxIters(int maxOuterIters)
    {
        if(maxOuterIters < 1)
            throw new IllegalArgumentException("Number of training iterations must be positive, not " + maxOuterIters);
        this.max_epochs = maxOuterIters;
    }

    /**
     * 
     * @return the maximum number of training iterations
     */
    public int getMaxIters()
    {
        return max_epochs;
    }

    
    /**
     * Sets the tolerance parameter for convergence. Smaller values will be more
     * exact, but larger values will converge faster. The default value is 
     * fairly exact at {@value #DEFAULT_EPS}, increasing it by an order of 
     * magnitude can often be done without hurting accuracy. 
     * 
     * @param e_out the tolerance parameter. 
     */
    public void setTolerance(double e_out)
    {
        if(e_out <= 0 || Double.isNaN(e_out))
            throw new IllegalArgumentException("convergence tolerance paramter must be positive, not " + e_out);
        this.tol = e_out;
    }
    
    /**
     * 
     * @return the convergence tolerance parameter
     */
    public double getTolerance()
    {
        return tol;
    }
    
    /**
     * Sets the loss function used for the model. The loss function controls 
     * whether or not regression, binary classification, or multi-class 
     * classification is supported. 
     * @param loss the loss function to use
     */
    public void setLoss(LossFunc loss)
    {
        this.loss = loss;
    }

    /**
     * Returns the loss function in use
     * @return the loss function in use
     */
    public LossFunc getLoss()
    {
        return loss;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        Vec x = data.getNumericalValues();
        if(ws.length == 1)
            return ((LossC)loss).getClassification(ws[0].dot(x)+bs[0]);
        else
        {
            Vec pred = new DenseVector(ws.length);
            for(int i = 0; i < ws.length; i++)
                pred.set(i, ws[i].dot(x)+bs[i]);
            ((LossMC)loss).process(pred, pred);
            return ((LossMC)loss).getClassification(pred);
        }
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        final int N = dataSet.getSampleSize();
        final int D = dataSet.getNumNumericalVars();
        
        if(dataSet.getPredicting().getNumOfCategories() !=2)
            throw new RuntimeException("Current SDCA implementation only support binary classification problems");
        
        dual_alphas = new double[N];
        
        ws = new Vec[]{new DenseVector(D)};
        Vec v = new DenseVector(D);
        //We need a lazily updated w to keep our work sparse! 
        final double[] w_lazy_backing = new double[D];
        final DenseVector w_lazy = new DenseVector(w_lazy_backing);
        bs = new double[1];
        
        final double[] x_norms = new double[N];
        double scaling = 1;
        for(int i = 0; i < N; i++)
        {
            x_norms[i] = dataSet.getDataPoint(i).getNumericalValues().pNorm(2);
            scaling = Math.max(scaling, x_norms[i]);
        }
        for(int i = 0; i < N; i++)
            x_norms[i] /= scaling;
        

        final double lambda_effective;
        final double sigma_p;
        final double tol_effective;
        
        if(alpha == 1)//Lasso case, but we MUST have some l2 reg to make this work
        {
            /*
             * See Section 5.5 Lasso, in "Accelerated proximal stochastic dual 
             * coordinate ascent for regularized loss minimization" paper. 
             * y_bar is given for the regression case. It appears y_bar's 
             * definition is in fact, the average loss of the 0 vector. We can 
             * compute this quickly.  
             */
            //TODO add support for weights in this
            //TODO we don't need to iterate over all points. loss will have the same output for each class, we can just iterate on the labels and average by class proportions
            double y_bar = 0;
            for(int i = 0; i < N; i++)
                y_bar += ((LossC)loss).getLoss(0.0, dataSet.getDataPointCategory(i)*2-1);
            y_bar /= N;
            
            sigma_p = lambda;
            lambda_effective = tol * Math.pow(lambda / Math.max(y_bar, 1e-7), 2) ;
            tol_effective = tol/2;
        }
        else
        {
            lambda_effective = lambda/scaling;
            sigma_p = (alpha/(1-alpha));
            tol_effective = tol;
        }
        
        Random rand = RandomUtil.getRandom();
        
        double gamma = ((LossC)loss).lipschitz();
        
        for(int epoch = 0; epoch < max_epochs; epoch++)
        {
            double dual_loss_est = 0;
            double primal_loss_est = 0;
            for(int count = 0; count < N; count++)
            {
                int i = rand.nextInt(N);

                double alpha_i_prev = dual_alphas[i];

                Vec x = dataSet.getDataPoint(i).getNumericalValues();
                int y = dataSet.getDataPointCategory(i)*2-1;
                //lets lazily determine what w should look like! 
                for(IndexValue iv : x)
                {
                    final int j = iv.getIndex();
                    final double v_j = v.get(j);
                    final double v_j_sign = Math.signum(v_j);
                    final double v_j_abs = Math.abs(v_j);
                    w_lazy_backing[j] = v_j_sign * Math.max(v_j_abs-sigma_p, 0.0);
                }
                final double raw_score = w_lazy.dot(x)/scaling+bs[0];

                //Option II 
                final double lossD = ((LossC)loss).getDeriv(raw_score, y);
                double u = -lossD;
                double q = u - alpha_i_prev;//also called z in older paper
                double q_sqrd = q*q;
                //Option III
                double phi_i = ((LossC)loss).getLoss(raw_score, y);
                double conjg = ((LossC)loss).getConjugate(-alpha_i_prev, raw_score, y);
                double x_norm = x_norms[i];
                double x_norm_sqrd = x_norm*x_norm;
                
                if(q_sqrd < 1e-14)
                    continue;//going to produce a zero in s, but will end up with NaN b/c of floating point
                //can happen qhen loss is zero and alpha_i is zero

                double s = (phi_i + conjg + raw_score*alpha_i_prev + gamma*q_sqrd/2)/(q_sqrd*(gamma+x_norm_sqrd/(lambda_effective*N)));
                s = Math.min(1, s);
                
                //for convergence check at end of epoch, record point estiamte of average primal and dual losses
                primal_loss_est += phi_i;
                if(!Double.isInfinite(conjg))
                    dual_loss_est += conjg;
                
                
                if(s == 0)
                    continue;

                double alpha_i_delta = s*q;

                //α(t)_i ←α(t−1)_i +∆α_i
                dual_alphas[i] += alpha_i_delta;
                //v^(t) ←v^(t−1) +(λn)^-1 X_i ∆α_i
                v.mutableAdd(alpha_i_delta/(scaling*lambda_effective*N), x);
                if(Double.isNaN(v.get(0)))
                    break;
                if(useBias)
                    bs[0] += alpha_i_delta/(scaling*lambda_effective*N);
                //w^(t) ←∇g∗(v^(t))
                //we do this lazily only when we need it! 
                
            }
            //gap is technically missing an estiamte of the regularization terms in the primal-dual gap 
            //But this looks close enough? Plus I don't need to do book keeping since w dosn't exist fully... 
            double gap = Math.abs(primal_loss_est-dual_loss_est)/N;
            
//            System.out.println("Epoch " + epoch + " has gap " + gap);
            if(gap < tol_effective)
            {
//                System.out.println("\tGap: " + gap + "  Epoch: " + epoch);
                break;
            }
        }
        
        //apply full sparsity patternt to w
        for(int j = 0; j < D; j++)
        {
            final double v_j = v.get(j);
            final double v_j_sign = Math.signum(v_j);
            final double v_j_abs = Math.abs(v_j);
            ws[0].set(j, scaling*v_j_sign * Math.max(v_j_abs - sigma_p, 0.0));
        }
        bs[0] *= scaling;
//        System.out.println(ws[0].nnz() + " " + lambda  + " " +  sigma_p + " " + ws[0]);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public Classifier clone()
    {
        return this;
    }

    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }

    @Override
    public Vec getRawWeight(int index)
    {
        return ws[index];
    }

    @Override
    public double getBias(int index)
    {
        return bs[index];
    }

    @Override
    public int numWeightsVecs()
    {
        return ws.length;
    }
}
