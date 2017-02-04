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
package jsat.classifiers.svm.extended;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.Uniform;
import jsat.exceptions.FailedToFitException;
import jsat.linear.*;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.FakeExecutor;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * This class implements the Convex Polytope Machine (CPM), which is an
 * extension of the Linear SVM. It is a binary classifier that has training time
 * proportionate to the linear case, but can obtain accuracies closer to that of
 * a kernelized SVM.<br>
 * Similar to the {@link AMM AMM} classifier, CPM uses multiple linear
 * hyper-planes to create a non-linear classifier. Increasing the number of
 * hyper-planes increases training/prediction time, but also increases the
 * amount of non-linearity the model can tolerate.<br>
 *
 *
 * <br>See: Kantchelian, A., Tschantz, M. C., Huang, L., Bartlett, P. L.,
 * Joseph, A. D., & Tygar, J. D. (2014). Large-margin Convex Polytope Machine.
 * In Proceedings of the 27th International Conference on Neural Information
 * Processing Systems (pp. 3248–3256). Cambridge, MA, USA: MIT Press. Retrieved
 * from <a href="http://dl.acm.org/citation.cfm?id=2969033.2969189">here</a>
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class CPM implements Classifier, Parameterized
{
    private static final long serialVersionUID = 3171068484917637037L;
    
    private int epochs;
    private double lambda;
    private int K;
    private double entropyThreshold;
    private double h;
    
    private Matrix Wp;
    private Matrix Wn;
    private Vec bp;
    private Vec bn;
    
    /**
     * Creates a new CPM classifier, with default parameters that should work
     * well for most cases.
     */
    public CPM()
    {
        this(1.0);
    }
    
    /**
     * Creates a new CPM classifier
     * @param K the number of hyper-planes to learn with. 
     */
    public CPM(int K)
    {
        this(1.0, K);
    }
    
    /**
     * Creates a new CPM classifier
     * @param lambda the regularization parameter
     */
    public CPM(double lambda)
    {
        this(lambda, 16);
    }
    
    
    /**
     * Creates a new CPM classifier
     * @param lambda the regularization parameter
     * @param K the number of hyper-planes to learn with. 
     */
    public CPM(double lambda, int K)
    {
        this(lambda, K, 3.0);
    }

    /**
     * Creates a new CPM classifier
     * @param lambda the regularization parameter
     * @param K the number of hyper-planes to learn with. 
     * @param entropyThreshold the parameter that encourages non-linearity to be exploited
     */
    public CPM(double lambda, int K, double entropyThreshold)
    {
        this(lambda, K, entropyThreshold, 50);
    }
    
    /**
     * Creates a new CPM classifier
     * @param lambda the regularization parameter
     * @param K the number of hyper-planes to learn with. 
     * @param entropyThreshold the parameter that encourages non-linearity to be exploited
     * @param epochs the number of training iterations
     */
    public CPM(double lambda, int K, double entropyThreshold, int epochs)
    {
        setEpochs(epochs);
        setLambda(lambda);
        setK(K);
        setEntropyThreshold(entropyThreshold);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public CPM(CPM toCopy)
    {
        this.epochs = toCopy.epochs;
        this.lambda = toCopy.lambda;
        this.K = toCopy.K;
        this.entropyThreshold = toCopy.entropyThreshold;
        this.h = toCopy.h;
        
        if(toCopy.Wp != null)
            this.Wp = toCopy.Wp.clone();
        if(toCopy.Wn != null)
            this.Wn = toCopy.Wn.clone();
        if(toCopy.bp != null)
            this.bp = toCopy.bp.clone();
        if(toCopy.bn != null)
            this.bn = toCopy.bn.clone();
    }
    
    

    /**
     * Sets the entropy threshold used for training. It ensures a diversity of
     * hyper-planes are used, where larger values encourage using more of the
     * hyper planes.<br>
     * <br>
     * This method is adjusted from the paper's definition so that the input can
     * be any non-negative value. It is recommended to try values in the range
     * of [0, 10]
     *
     * @param entropyThreshold the non-negative parameter for hyper-plane diversity
     */
    public void setEntropyThreshold(double entropyThreshold)
    {
        if(entropyThreshold < 0 || Double.isNaN(entropyThreshold) || Double.isInfinite(entropyThreshold))
            throw new IllegalArgumentException("Entropy threshold must be non-negative, not " + entropyThreshold);
        this.entropyThreshold = entropyThreshold;
        set_h_properly();
    }

    private void set_h_properly()
    {
        h = Math.log(entropyThreshold * K / 10.0)/Math.log(2);
        if(h <= 0)
            h = 0;
    }

    /**
     * 
     * @return the non-negative parameter for hyper-plane diversity
     */
    public double getEntropyThreshold()
    {
        return entropyThreshold;
    }

    /**
     * Sets the regularization parameter &lambda; to use. Larger values penalize
     * model complexity. This value is adjusted from the form in the original
     * paper so that you do not need to consider the number of epochs
     * explicitly. The effective regularization will be divided by the total
     * number of training updates.
     *
     * @param lambda the regularization parameter value to use, the recommended
     * range range is (0, 10<sup>4</sup>]
     */
    public void setLambda(double lambda)
    {
        this.lambda = lambda;
    }

    /**
     * 
     * @return the regularization parameter value
     */
    public double getLambda()
    {
        return lambda;
    }

    /**
     * Sets the number of hyper planes to use when training. A normal linear
     * model is equivalent to using only 1 hyper plane. The more hyper planes
     * used, the more modeling capacity the algorithm has, but the slower it
     * will run.
     *
     * @param K the number of hyper planes to use. 
     */
    public void setK(int K)
    {
        this.K = K;
        set_h_properly();
    }

    /**
     * 
     * @return the number of hyper planes to use. 
     */
    public int getK()
    {
        return K;
    }
    
    /**
     * Sets the number of whole iterations through the training set that will be
     * performed for training
     * @param epochs the number of whole iterations through the data set
     */
    public void setEpochs(int epochs)
    {
        if(epochs < 1)
            throw new IllegalArgumentException("epochs must be a positive value");
        this.epochs = epochs;
    }

    /**
     * Returns the number of epochs used for training
     * @return the number of epochs used for training
     */
    public int getEpochs()
    {
        return epochs;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        Vec x = data.getNumericalValues();
        
        double pos_score = Wp.multiply(x).add(bp).max();
        double neg_score = Wn.multiply(x).add(bn).max();
        
        CategoricalResults cr = new CategoricalResults(2);
        if(neg_score > 0 && pos_score > 0)//ambigious case, lets go with larger magnitude
        {
            if(neg_score > pos_score)
                cr.setProb(0, 1.0);
            else
                cr.setProb(1, 1.0);
        }
        else if(neg_score > 0)
            cr.setProb(0, 1.0);
        else if(pos_score > 0)
            cr.setProb(1, 1.0);
        else if(neg_score > pos_score )//not actually how describes in paper, but its ambigious - so lets use larger to tie break
            //ambig b/c if no model claims ownership, we get a score of 0
            cr.setProb(0, 1.0);
        else
            cr.setProb(1, 1.0);
        return cr;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
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
    
    /**
     * 
     * @param dots dot product between the input and each of the k hyper planes
     * @param owned a count of how many data points are assigned to this hyper plane
     * @param assignments maps each data point to the hyper plane that owns it. May have negative values for points not yet assigned
     * @param assigned_positive_instances the number of <bold>positive</bold> instances taht have been assigned to a hyper plane
     */
    private int ASSIGN(Vec dots, int indx, int k_true_max, int[] owned, int[] assignments, int assigned_positive_instances)
    {
        //Done outside this function 
//        int k_true_max = 0;
//        for(int i = 1; i < dots.length(); i++)
//            if(dots.get(i) > dots.get(k_true_max))
//                k_true_max = i;
        
        int old_owner = assignments[indx];
        
        double cur_entropy = 0;
        double new_entropy = Double.POSITIVE_INFINITY;
        int max_owned = 0;
        if(assigned_positive_instances > K*10)//we have enough assignments to start estimating entropy
        {
            new_entropy = 0;
            for(int i = 0; i < K; i++)
            {
                max_owned = Math.max(max_owned, owned[i]);//used later
                
//                double p_i = owned[i]/(double)assigned_positive_instances;
                double numer = owned[i];
                double denom = assigned_positive_instances;
                if(numer > 0 )
//                    cur_entropy += -p_i * Math.log(p_i)/Math.log(2);
                    cur_entropy += -numer*(Math.log(numer)-Math.log(denom))/(Math.log(2)*denom);
                
                //now calculate for new_entropy
                if(old_owner < 0)//every point has a differnt value, b/c denominator changes
                {
                    denom++;
                    if(i == k_true_max)//numer changes here too
                        numer++;
                    if(numer > 0 )
                        new_entropy += -numer*(Math.log(numer)-Math.log(denom))/(Math.log(2)*denom);
                }
                else if(old_owner == k_true_max)//no change in ownership, means no change in entropy
                {
                    new_entropy = cur_entropy;
                }
                else//change in ownership, denom remains the same, numer may change
                {
                    if(i == k_true_max)
                        numer++;
                    else if(i == old_owner)
                        numer--;
                    
                    if(numer > 0 )
                        new_entropy += -numer*(Math.log(numer)-Math.log(denom))/(Math.log(2)*denom);
                }
            }
            
            new_entropy += cur_entropy;//new was calcualted as a delta from cur, so by adding we get the correct value
        }
        
        if(new_entropy >= h)//if ENTROPY(UNADJ +(x, kunadj)) ≥ h then
            return k_true_max;
        //else
        //find max that would result in an increase in entropy
        
        int k_inc_max = 0;
        if (old_owner >= 0)//don't need to compute entropy, moving to any position that owns fewer would increase entropy
        {
            for (int i = 1; i < dots.length(); i++)
                if (owned[old_owner] > owned[i] && dots.get(i) > dots.get(k_inc_max))
                    k_inc_max = i;
        }
        else//not assigned, assign to anyone owns less than the most to improve
        {
            double best_score = Double.NEGATIVE_INFINITY;
            for (int i = 1; i < dots.length(); i++)
                if (max_owned > owned[i] && dots.get(i) > best_score)
                {
                    k_inc_max = i;
                    best_score = dots.get(i);
                }
            
            if(Double.isInfinite(best_score))//why couldn't we find someone? Bail out 
                return k_true_max;//Lets just give the original max
        }
        return k_inc_max;
    }

    /**
     * Training procedure that can be applied to each version of the CPM
     * sub-problem.
     *
     * @param D the dataset to train on
     * @param W the weight matrix of vectors to use
     * @param b a vector that stores the associated bias terms for each weigh
     * vector.
     * @param sign_mul Either positive or negative 1. Controls whether or not
     * the positive or negative class is to be enveloped by the polytype
     */
    private void sgdTrain(ClassificationDataSet D, MatrixOfVecs W, Vec b, int sign_mul)
    {
        IntList order = new IntList(D.getSampleSize());
        ListUtils.addRange(order, 0, D.getSampleSize(), 1);
        
        final double lambda_adj = lambda/(D.getSampleSize()*epochs);
        
        int[] owned = new int[K];//how many points does thsi guy own?
        int assigned_positive_instances = 0;//how many points in the positive class have been assigned?
        int[] assignments = new int[D.getSampleSize()];//who owns each data point
        Arrays.fill(assignments, -1);//Starts out that no one is assigned! 
        
        long t = 0;
        for(int epoch = 0; epoch < epochs; epoch++)
        {
            Collections.shuffle(order);
            for(int i : order)
            {
                t++;
                double eta = 1/(lambda_adj*t);
                Vec x_i = D.getDataPoint(i).getNumericalValues();
                int y_i = (D.getDataPointCategory(i)*2-1)*sign_mul;
                
                Vec dots = W.multiply(x_i);
                dots.mutableAdd(b);
                
                if(y_i == -1)
                {
                    for(int k = 0; k < K; k++)
                        if(dots.get(k) > -1)
                        {
                            W.getRowView(k).mutableSubtract(eta, x_i);
                            b.increment(k, -eta);
                        }
                }
                else//y_i == 1
                {
                    int k_true_max = 0;
                    for(int k = 1; k < dots.length(); k++)
                        if(dots.get(k) > dots.get(k_true_max))
                            k_true_max = k;
                    
                    if(dots.get(k_true_max) < 1)
                    {
                        int z = ASSIGN(dots, i, k_true_max, owned, assignments, assigned_positive_instances);
                        W.getRowView(z).mutableAdd(eta, x_i);
                        b.increment(z, eta);
                        
                        //book keeping
                        if(assignments[i] < 0)//first assignment, inc counter
                            assigned_positive_instances++;
                        else//change owner, decrement ownership count
                            owned[assignments[i]]--;
                        owned[z]++;
                        assignments[i] = z;
                        
                    }
                }
                
//                W.mutableMultiply(1-eta*lambda);
                //equivalent form, more stable
                W.mutableMultiply(1-1.0/t);
                b.mutableMultiply(1-1.0/t);
            }
        }
    }
    
    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        if(dataSet.getPredicting().getNumOfCategories() > 2)
            throw new FailedToFitException("CPM is a binary classifier, it can not be trained on a dataset with " + dataSet.getPredicting().getNumOfCategories() + " classes");
        final int d = dataSet.getNumNumericalVars();
        List<Vec> Wv_p = new ArrayList<Vec>(K);
        List<Vec> Wv_n = new ArrayList<Vec>(K);
        bp = new DenseVector(K);
        bn = new DenseVector(K);
        for(int i = 0; i < K; i++)
        {
            Wv_p.add(new ScaledVector(new DenseVector(d)));
            Wv_n.add(new ScaledVector(new DenseVector(d)));
        }
        MatrixOfVecs W_p = new MatrixOfVecs(Wv_p);
        MatrixOfVecs W_n = new MatrixOfVecs(Wv_n);
        
        sgdTrain(dataSet, W_p, bp, +1);
        sgdTrain(dataSet, W_n, bn, -1);
        
        this.Wp = new DenseMatrix(W_p);
        this.Wn = new DenseMatrix(W_n);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    @Override
    public CPM clone()
    {
        return new CPM(this);
    }

    /**
     * Provides a distribution of reasonable values for the
     * {@link #setLambda(double) &lambda;} parameter
     *
     * @param d the dataset to get the guess for
     * @return the distribution to search this parameter from
     */
    public static Distribution guessLambda(DataSet d)
    {
        return new LogUniform(1e-1, 1e4);
    }

    /**
     * Provides a distribution of reasonable values for the {@link #setEntropyThreshold(double)
     * } parameter
     *
     * @param d the dataset to get the guess for
     * @return the distribution to search this parameter from
     */
    public static Distribution guessEntropyThreshold(DataSet d)
    {
        return new Uniform(1e-1, 10);
    }
}
