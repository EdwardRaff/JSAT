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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.SimpleWeightVectorModel;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.WarmClassifier;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.Uniform;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.lossfunctions.*;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.regression.WarmRegressor;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.random.RandomUtil;

/**
 * This class implements the Proximal Stochastic Dual Coordinate Ascent (SDCA)
 * algorithm for learning general linear models with Elastic-Net regularization.
 * It is a fast learning algorithm, and can be used for generic Logistic or
 * least-squares regression with elastic-net regularization.<br>
 * It can work with any {@link LossFunc} to determine if it solves a
 * classification or regression problem, so long as the
 * {@link LossFunc#getConjugate(double, double, double) conjugate} method of the
 * loss is implemented properly. This is especially useful for more exotic
 * cases, like using the robust {@link HuberLoss Huber loss} in a L1 regularized
 * scenario. <br>
 * NOTE: The current implementation dose not support any multi-class loss
 * function, though that isn't a limitation of the algorithm.
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class SDCA implements Classifier, Regressor, Parameterized, SimpleWeightVectorModel, WarmClassifier, WarmRegressor
{
    private LossFunc loss;
    private boolean useBias = true;
    private double tol = 0.001;
    private double lambda;
    private double alpha = 0.5;
    private int max_epochs = 200;
    private double[] dual_alphas;
    /**
     * Returns the number of epochs SDCA took until reaching convergence. 
     */
    protected int epochs_taken;
    
    private Vec[] ws;
    private double[] bs;

    /**
     * Creates a new SDCA learner for {@link LogisticLoss logistic-regression}. 
     * Pure L2 or L1 regularization can be obtained using the
     * {@link #setAlpha(double) alpha} parameter.
     */
    public SDCA()
    {
        this(1e-5);
    }

    /**
     * <br>The implementation will use Elastic-Net regularization by default.
     * Pure L2 or L1 regularization can be obtained using the
     * {@link #setAlpha(double) alpha} parameter.
     * @param lambda the regularization penalty to use. 
     */
    public SDCA(double lambda)
    {
        this(lambda, new LogisticLoss());
    }

    /**
     * Creates a new SDCA learner for either a classification or regression
     * problem, the type of which is determined by the LossFunction given.
     * <br>The implementation will use Elastic-Net regularization by default.
     * Pure L2 or L1 regularization can be obtained using the
     * {@link #setAlpha(double) alpha} parameter.
     *
     * @param lambda the regularization penalty to use.
     * @param loss the loss function to use for training, which determines if it
     * implements classification or regression
     */
    public SDCA(double lambda, LossFunc loss)
    {
        setLoss(loss);
        setLambda(lambda);
    }
    

    /**
     * Copy Constructor
     * @param toCopy the object to copy
     */
    public SDCA(SDCA toCopy)
    {
        this.loss = toCopy.loss.clone();
        this.useBias = toCopy.useBias;
        this.tol = toCopy.tol;
        this.lambda = toCopy.lambda;
        this.alpha = toCopy.alpha;
        this.max_epochs = toCopy.max_epochs;
        this.epochs_taken = toCopy.epochs_taken;
        if(toCopy.dual_alphas != null)
            this.dual_alphas = Arrays.copyOf(toCopy.dual_alphas, toCopy.dual_alphas.length);
        if(toCopy.ws != null)
        {
            this.ws = new Vec[toCopy.ws.length];
            this.bs = new double[toCopy.bs.length];
            for(int i = 0; i < toCopy.ws.length; i++)
            {
                this.ws[i] = toCopy.ws[i].clone();
                this.bs[i] = toCopy.bs[i];
            }
        }
        
    }
    
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
     * classification is supported. <br>
     * <b>NOTE:</b> SDCA requires that the given loss function implement the
     * {@link LossFunc#getConjugate(double, double, double) conjugate} method,
     * otherwise it will not work with this algorithm.
     *
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
    public double regress(DataPoint data)
    {
        Vec x = data.getNumericalValues();
        return ((LossR)loss).getRegression(ws[0].dot(x)+bs[0]);
    }

    @Override
    public void train(ClassificationDataSet dataSet, boolean parallel)
    {
        train(dataSet);
    }

    @Override
    public void train(ClassificationDataSet dataSet)
    {
        if(dataSet.getPredicting().getNumOfCategories() !=2)
            throw new RuntimeException("Current SDCA implementation only support binary classification problems");
        
        double[] targets = new double[dataSet.size()];
        for(int i = 0; i < targets.length; i++)
            targets[i] = dataSet.getDataPointCategory(i)*2-1;
        
        trainProxSDCA(dataSet, targets, null);
    }
    
    @Override
    public void train(ClassificationDataSet dataSet, Classifier warmSolution, boolean parallel)
    {
        train(dataSet, warmSolution);
    }

    @Override
    public void train(ClassificationDataSet dataSet, Classifier warmSolution)
    {
        if(warmSolution == null || !(warmSolution instanceof SDCA))
            throw new FailedToFitException("SDCA implementation can only be warm-started from another instance of SDCA");
        
        if(dataSet.getPredicting().getNumOfCategories() !=2)
            throw new RuntimeException("Current SDCA implementation only support binary classification problems");
        
        double[] targets = new double[dataSet.size()];
        for(int i = 0; i < targets.length; i++)
            targets[i] = dataSet.getDataPointCategory(i)*2-1;
        
        trainProxSDCA(dataSet, targets, ((SDCA)warmSolution).dual_alphas);
    }

    @Override
    public void train(RegressionDataSet dataSet, boolean parallel)
    {
        train(dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        double[] targets = new double[dataSet.size()];
        for(int i = 0; i < targets.length; i++)
            targets[i] = dataSet.getTargetValue(i);
        
        trainProxSDCA(dataSet, targets, null);
    }

    @Override
    public void train(RegressionDataSet dataSet, Regressor warmSolution, boolean parallel)
    {
        train(dataSet, warmSolution);
    }

    @Override
    public void train(RegressionDataSet dataSet, Regressor warmSolution)
    {
        double[] targets = new double[dataSet.size()];
        for(int i = 0; i < targets.length; i++)
            targets[i] = dataSet.getTargetValue(i);
        
        trainProxSDCA(dataSet, targets, ((SDCA)warmSolution).dual_alphas);
    }
    
    private void trainProxSDCA(DataSet dataSet, double[] targets, double[] warm_alphas)
    {
        final int N = dataSet.size();
        final int D = dataSet.getNumNumericalVars();
        
        ws = new Vec[]{new DenseVector(D)};
        DenseVector v = new DenseVector(D);
        bs = new double[1];
        
        final double[] x_norms = new double[N];
        double scaling = 1;
        /*
         * SDCA seems scale sensative for classification, but  insensative for 
         * regression. In fact, re-scaling is breaking regression... so lets 
         * just not scale when doing regression! Weird, should dig in more later. 
         */
        final boolean is_regression = dataSet instanceof RegressionDataSet;
        for(int i = 0; i < N; i++)
        {
            x_norms[i] = dataSet.getDataPoint(i).getNumericalValues().pNorm(2);
            //Scaling seems to muck up regresion... so dont!
            if(!is_regression)
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
                y_bar += loss.getLoss(0.0, targets[i]);
            y_bar /= N;
            
            sigma_p = lambda;
            lambda_effective = tol * Math.pow(lambda / Math.max(y_bar, 1e-7), 2) ;
            tol_effective = tol/2;
        }
        else
        {
            lambda_effective = lambda;
            sigma_p = (alpha/(1-alpha));
            tol_effective = tol;
        }
        
        //set up the weight vector used during training. If using elatic-net, we will do lazy-updates of the values
        //otherswise, we can just re-use v
        final double[] w_lazy_backing;
        final DenseVector w_lazy;
        if(alpha > 0)
        {
            //We need a lazily updated w to keep our work sparse! 
            w_lazy_backing = new double[D];
            w_lazy = new DenseVector(w_lazy_backing);
        }
        else//alpha = 0, we can just re-use v! 
        {
            w_lazy_backing =null;
            w_lazy = v;
        }

        //init dual alphas, and any warm-start on the solutions
        if (warm_alphas == null)
            dual_alphas = new double[N];
        else
        {
            if (N != warm_alphas.length)
                throw new FailedToFitException("SDCA only supports warm-start training from the same dataset. A dataset of side " + N + " was given for training, but the warm solution was trained on " + warm_alphas.length + " points.");
            this.dual_alphas = Arrays.copyOf(warm_alphas, warm_alphas.length);

            for(int i = 0; i < N; i++)
            {
                v.mutableAdd(dual_alphas[i], dataSet.getDataPoint(i).getNumericalValues());
                if (useBias)
                    bs[0] += dual_alphas[i];
            }
            //noramlize 
            v.mutableDivide(scaling * lambda_effective * N);
            bs[0] /= (scaling * lambda_effective * N);
        }
        
        Random rand = RandomUtil.getRandom();
        
        double gamma = loss.lipschitz();
        
        IntList epoch_order = new IntList(N);
        ListUtils.addRange(epoch_order, 0, N, 1);
        
        
        epochs_taken = 0;
        int primal_converg_check = 0;
        
        for(int epoch = 0; epoch < max_epochs; epoch++)
        {
            double prevPrimal = Double.POSITIVE_INFINITY;
            epochs_taken++;
            double dual_loss_est = 0;
            double primal_loss_est = 0;
            Collections.shuffle(epoch_order, rand);
            for(int i : epoch_order)
            {

                double alpha_i_prev = dual_alphas[i];

                Vec x = dataSet.getDataPoint(i).getNumericalValues();
                double y = targets[i];
                if(alpha > 0)//lets lazily determine what w should look like! 
                    for(IndexValue iv : x)
                    {
                        final int j = iv.getIndex();
                        final double v_j = v.get(j);
                        final double v_j_sign = Math.signum(v_j);
                        final double v_j_abs = Math.abs(v_j);
                        w_lazy_backing[j] = v_j_sign * Math.max(v_j_abs-sigma_p, 0.0);
                    }
                //else, w_lazy points to v, which has the correct values
                final double raw_score = w_lazy.dot(x)/scaling+bs[0];

                //Option II 
                final double lossD = loss.getDeriv(raw_score, y);
                double u = -lossD;
                double q = u - alpha_i_prev;//also called z in older paper
                double q_sqrd = q*q;
                if(q_sqrd <= 1e-32)
                    continue;//avoid a NaN from div by zero
                //Option III
                double phi_i = loss.getLoss(raw_score, y);
                double conjg = loss.getConjugate(-alpha_i_prev, raw_score, y);
                double x_norm = x_norms[i];
                double x_norm_sqrd = x_norm*x_norm;

                double denom = q_sqrd*(gamma+x_norm_sqrd/(lambda_effective*N));
                double s = (phi_i + conjg + raw_score*alpha_i_prev + gamma*q_sqrd/2)/denom;
                s = Math.min(1, s);
                
                //for convergence check at end of epoch, record point estiamte of average primal and dual losses
                primal_loss_est += phi_i;
                if(!Double.isInfinite(conjg))
                    dual_loss_est += -conjg;
                
                
                if(s == 0)
                    continue;

                double alpha_i_delta = s*q;

                //α(t)_i ←α(t−1)_i +∆α_i
                dual_alphas[i] += alpha_i_delta;
                //v^(t) ←v^(t−1) +(λn)^-1 X_i ∆α_i
                v.mutableAdd(alpha_i_delta/(scaling*lambda_effective*N), x);
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
            //Sometime's gap dosn't work well when alphas hit weird ranges
            //lets check if the primal hasn't changed much in a while
            if(prevPrimal-primal_loss_est/N < tol_effective/5)
            {
                if(primal_converg_check++ > 10)
                    break;
            }
            else
                primal_converg_check = 0;
            
            prevPrimal = primal_loss_est/N;
        }
        
        //apply full sparsity patternt to w
        for(int j = 0; j < D; j++)
        {
            final double v_j = v.get(j);
            final double v_j_sign = Math.signum(v_j);
            final double v_j_abs = Math.abs(v_j);
            ws[0].set(j, v_j_sign * Math.max(v_j_abs - sigma_p, 0.0)/scaling);
        }
//        System.out.println(ws[0].nnz() + " " + lambda  + " " +  sigma_p + " " + ws[0]);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public SDCA clone()
    {
        return new SDCA(this);
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

    @Override
    public boolean warmFromSameDataOnly()
    {
        return true;
    }
    
    /**
     * Guess the distribution to use for the regularization term
     * {@link #setLambda(double) lambda}. 
     *
     * @param d the data set to get the guess for
     * @return the guess for the lambda parameter 
     */
    public static Distribution guessLambda(DataSet d)
    {
        int N = d.size();
        return new LogUniform(1.0/(N*50), Math.min(1.0/(N/50), 1.0));
    }
    
    
    /**
     * Guess the distribution to use for the regularization balance
     * {@link #setAlpha(double) alpha}. 
     *
     * @param d the data set to get the guess for
     * @return the guess for the lambda parameter 
     */
    public static Distribution guessAlpha(DataSet d)
    {
        return new Uniform(0.0, 0.5);
    }
}
