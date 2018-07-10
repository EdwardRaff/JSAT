/*
 * Copyright (C) 2016 Edward Raff
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
package jsat.classifiers.svm;

import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.Vec;
import static java.lang.Math.*;
import java.util.Arrays;
import jsat.distributions.kernels.NormalizedKernel;
import jsat.utils.concurrent.AtomicDouble;
import jsat.utils.concurrent.ParallelUtils;

/**
 * This class implements a version of the Support Vector Machine without a bias
 * term. In addition, the current implementation requires that the Kernel Trick
 * used be a {@link KernelTrick#normalized() normalized} kernel. If the given
 * kernel is not normalized, this class will automatically wrap it to become
 * normalized.<br>
 * <br>
 * Because there is no bias term, this class should never be used with the
 * Linear kernel. But for the more common RBF kernel the lack of bias term
 * should have minimal impact on accuracy.<br>
 * <br>
 * See: Steinwart, I., Hush, D., & Scovel, C. (2011). <i>Training SVMs Without
 * Offset</i>. The Journal of Machine Learning Research, 12, 141–202.
 *
 * @author Edward Raff
 */
public class SVMnoBias extends SupportVectorLearner implements BinaryScoreClassifier
{
    
    private double C = 1;
    private double tolerance = 1e-3;
        
    /**
     * Stores the true label value (-1 or +1) of the data point
     */
    protected short[] label;
    /**
     * Weight values to apply to each data point
     */
    protected Vec weights;
    
    //Variables used during training
    private double T_a;
    private double S_a;

    
    /**
     * Creates a new SVM object that uses no cache mode. 
     * 
     * @param kf the kernel trick to use
     */
    public SVMnoBias(KernelTrick kf)
    {
        super(kf, SupportVectorLearner.CacheMode.NONE);
    }

    public SVMnoBias(SVMnoBias toCopy)
    {
        super(toCopy);
        if(toCopy.weights != null)
            this.weights = toCopy.weights.clone();
        if(toCopy.label != null)
            this.label = Arrays.copyOf(toCopy.label, toCopy.label.length);
        this.C = toCopy.C;
        this.tolerance = toCopy.tolerance;
    }

    @Override
    public void setKernel(KernelTrick kernel)
    {
        if(kernel.normalized())
            super.setKernel(kernel); 
        else
            super.setKernel(new NormalizedKernel(kernel));
    }
    
    @Override
    public double getScore(DataPoint dp)
    {
        return kEvalSum(dp.getNumericalValues());
    }

    @Override
    public SVMnoBias clone()
    {
        return new SVMnoBias(this);
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if (vecs == null)
            throw new UntrainedModelException("Classifier has yet to be trained");

        CategoricalResults cr = new CategoricalResults(2);

        double sum = getScore(data);

        if (sum > 0)
            cr.setProb(1, 1);
        else
            cr.setProb(0, 1);

        return cr;
    }

    @Override
    public void train(ClassificationDataSet dataSet, boolean parallel)
    {
        bookKeepingInit(dataSet);
        
        double[] nabla_W = procedure3_init();
        
        solver_1d(nabla_W, parallel);
        
        
        setCacheMode(null);
    }
    
    /**
     * 
     * @param dataSet the dataset to train on
     * @param warm_start Array of initial alpha values to use for support
     * vectors. The absolute value of the inputs will be used. may be longer
     * than the number of data points.
     */
    protected void train(ClassificationDataSet dataSet, double[] warm_start)
    {
        train(dataSet, warm_start, false);
    }
    
    protected void train(ClassificationDataSet dataSet, double[] warm_start, boolean parallel)
    {
        bookKeepingInit(dataSet);
        
        for(int i = 0; i < alphas.length; i++)
            alphas[i] = Math.abs(warm_start[i]);
        
        double[] nabla_W = procedure4m_init(parallel);
        
        solver_1d(nabla_W, parallel);
        
        setCacheMode(null);
    }

    private void solver_1d(final double[] nabla_W, boolean parallel)
    {
        final int N = alphas.length;
        final double lambda = 1/(2*C*N);
        
        //Algorithm 1 1D-SVM solver
        while(S_a > tolerance/(2*lambda))
        {
//            System.out.println(S_a + " > " + tolerance/(2*lambda));
            //Procedure 1 Calculate i∗ ∈ argmaxi=1,...,n δi · (∇Wi(α)−δi/2)
            double bestgain = -1;
            int i_max = -1;
            double best_delta = -1;
            for(int i = 0; i < N; i++)
            {
                double a_star_i = max(min(weights.get(i)*C, nabla_W[i]+alphas[i]), 0);
                double delta = a_star_i-alphas[i];
                ///gain←δ· (∇Wi(α)−δ/2)
                double gain = delta*(nabla_W[i]-delta/2);
                if(gain >= bestgain)
                {
                    bestgain = gain;
                    i_max = i;
                    best_delta = delta;
                }
            }
            
            //adjust alhpa
            alphas[i_max] += best_delta;
            //fuzzy clip to get hard 0/Cs
            if(alphas[i_max] + 1e-7 > weights.get(i_max)*C )//round to max
                alphas[i_max] = weights.get(i_max)*C;
            else if(alphas[i_max] - 1e-7 < 0)//round to 0
                alphas[i_max] = 0;
            
            final double delta = best_delta;
            final int i = i_max;
            
            //use Procedure 2 to update ∇W(α) in direction i∗ by δ and calculate S(α)
            //T(α)←T(α)−δ(2∇Wi(α)−1−δ)
            T_a -= best_delta*(2*nabla_W[i_max]-1-best_delta);
            final AtomicDouble E_a = new AtomicDouble(0.0);
            accessingRow(i);//hint to caching scheme
            ParallelUtils.run(parallel, N, (start, end) ->
            {
                double Ea_delta = 0;
                for (int j = start; j < end; j++)
                {
                    nabla_W[j] -= delta * label[i] * label[j] * kEval(i, j);
                    Ea_delta += weights.get(j) * C * min(max(0, nabla_W[j]), 2);
                }
                E_a.addAndGet(Ea_delta);
            });
            
            S_a = T_a + E_a.get();
        }
        accessingRow(-1);//no more row accesses

        //collapse label into signed alphas
        for(int i = 0; i < label.length; i++)
            alphas[i] *= label[i];
    }

    private double[] procedure3_init()
    {
        int N = alphas.length;
        //Procedure 3 Initialize by αi←0 and compute ∇W(α), S(α), and T(α).
        T_a = 0;
        S_a = 0;
        double[] nabla_W = new double[N];
        for(int i = 0; i < N; i++)
        {
            nabla_W[i] = 1;
            S_a += weights.get(i)*C;
        }
        return nabla_W;
    }
    
    private double[] procedure4m_init(boolean parallel)
    {
        final int N = alphas.length;
        //Procedure 3 Initialize by αi←0 and compute ∇W(α), S(α), and T(α).
        T_a = 0;
        final AtomicDouble E_a = new AtomicDouble(0.0);
        final AtomicDouble T_a_accum = new AtomicDouble(0.0);
        final double[] nabla_W = new double[N];
        
        ParallelUtils.run(parallel, N, (start, end)->
        {
            double Ta_delta = 0;
            double Ea_delta = 0;
            for(int i = start; i < end; i++)
            {
                nabla_W[i] = 1;
                double nabla_Wi_delta = 0;
                for(int j = 0; j < N; j++)
                {
                    if(alphas[j] == 0)
                        continue;
                    //We call k instead of kEval b/c we are accing most 
                    //of the n^2 values, so nothing will get to stay in 
                    //cache. Unless we are using FULL cacheing, in which
                    //case we will get re-use. 
                    //Using k avoids LRU overhead which can be significant
                    //for fast to evaluate datasets
                    double k_ij;
                    if(getCacheMode() == CacheMode.FULL)
                        k_ij = kEval(i, j);
                    else
                        k_ij = k(i, j);

                    nabla_Wi_delta -= alphas[j] * label[i] * label[j] * k_ij;
                }
                nabla_W[i] += nabla_Wi_delta;

                Ta_delta -= alphas[i]*nabla_W[i];
                Ea_delta += weights.get(i)*C*min(max(nabla_W[i], 0), 2);
            }

            E_a.addAndGet(Ea_delta);
            T_a_accum.addAndGet(Ta_delta);
        });
        
        T_a = T_a_accum.get();
        S_a = T_a + E_a.get(); 
        return nabla_W;
    }

    private void bookKeepingInit(ClassificationDataSet dataSet)
    {
        final int N = dataSet.size();
        vecs = dataSet.getDataVectors();
        weights = dataSet.getDataWeights();
        label = new short[N];
        for(int i = 0; i < N; i++)
            label[i] = (short) (dataSet.getDataPointCategory(i)*2-1);
        setCacheMode(getCacheMode());//Initiates the cahce
        //initialize alphas array to all zero
        alphas = new double[N];//zero is default value
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }
    
    /**
     * Sets the complexity parameter of SVM. The larger the C value the harder 
     * the margin SVM will attempt to find. Lower values of C allow for more 
     * misclassification errors. 
     * @param C the soft margin parameter
     */
//    @Parameter.WarmParameter(prefLowToHigh = true)
    public void setC(double C)
    {
        if(C <= 0)
            throw new ArithmeticException("C must be a positive constant");
        this.C = C;
    }

    /**
     * Returns the soft margin complexity parameter of the SVM
     * @return the complexity parameter of the SVM
     */
    public double getC()
    {
        return C;
    }
    
    /**
     * Sets the tolerance for the solution. Higher values converge to worse 
     * solutions, but do so faster
     * @param tolerance the tolerance for the solution
     */
    public void setTolerance(double tolerance)
    {
        this.tolerance = tolerance;
    }

    /**
     * Returns the solution tolerance 
     * @return the solution tolerance 
     */
    public double getTolerance()
    {
        return tolerance;
    }
}
