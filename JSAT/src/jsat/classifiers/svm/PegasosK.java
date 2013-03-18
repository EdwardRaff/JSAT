
package jsat.classifiers.svm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;

/**
 * Implements the kernelized version of the {@link Pegasos} algorithm for SVMs. 
 * Unlike standard SVM algorithms, this one relies on randomness and has no 
 * guarantee to reach the optimal solution, however it is very fast to train. 
 * Each iteration of the algorithm randomly selects one datapoint to potentially 
 * update the coefficient of. <br>
 * The resulting set of support vectors may be more or less sparse than a normal
 * SVM implementation. <br>
 * Because the Pegasos algorithm is stochastic and the kernelized updates on 
 * errors given regularization, the kernelized version may have more difficulty 
 * with noisy or overlapping class distributions. 
 * <br><br>
 * See: Shalev-Shwartz, S., Singer, Y., & Srebro, N. (2007). <i>Pegasos : Primal
 * Estimated sub-GrAdient SOlver for SVM</i>. 24th international conference on 
 * Machine learning (pp. 807–814). New York, NY: ACM. 
 * doi:10.1145/1273496.1273598
 * 
 * @author Edward Raff
 */
public class PegasosK extends SupportVectorMachine implements Parameterized
{
    private double regularization;
    private int iterations;
    
    private int[] sign;
    private int[] alpha;
    
    private List<Parameter> params = Parameter.getParamsFromMethods(this);
    private Map<String, Parameter> paramMap = Parameter.toParameterMap(params);

    /**
     * Creates a new kernelized Pegasos SVM solver 
     * 
     * @param regularization the amount of regularization to apply, normally a very small positive value
     * @param iterations the number of update iterations to perform
     * @param kernel the kernel to use
     */
    public PegasosK(double regularization, int iterations, KernelTrick kernel)
    {
        this(regularization, iterations, kernel, CacheMode.NONE);
    }
    
    /**
     * Creates a new kernelized Pegasos SVM solver 
     * 
     * @param regularization the amount of regularization to apply, normally a very small positive value
     * @param iterations the number of update iterations to perform
     * @param kernel the kernel to use
     * @param cacheMode what type of kernel caching to use
     */
    public PegasosK(double regularization, int iterations, KernelTrick kernel, CacheMode cacheMode)
    {
        super(kernel, cacheMode);
        setRegularization(regularization);
        setIterations(iterations);
    }

    /**
     * Sets the number of iterations of the algorithm to perform. Each iteration 
     * may or may not update a single coefficient for a specific data point.
     * 
     * @param iterations the number of learning iterations to perform
     */
    public void setIterations(int iterations)
    {
        this.iterations = iterations;
    }

    /**
     * Returns the number of iterations used during training
     * @return the number of iterations used in training
     */
    public int getIterations()
    {
        return iterations;
    }

    /**
     * Sets the amount of regularization to apply. The regularization must be a 
     * positive value
     * @param regularization the amount of regularization to apply
     */
    public void setRegularization(double regularization)
    {
        if(Double.isNaN(regularization) || Double.isInfinite(regularization) || regularization <= 0)
            throw new ArithmeticException("Regularization must be a positive constant, not " + regularization);
        this.regularization = regularization;
    }

    /**
     * Returns the amount of regularization used
     * @return the amount of regularization used
     */
    public double getRegularization()
    {
        return regularization;
    }
    
    @Override
    public Classifier clone()
    {
        PegasosK clone = new PegasosK(regularization, iterations, getKernel().clone(), getCacheMode());
        
        if(this.vecs != null)
        {
            clone.vecs = new Vec[this.vecs.length];
            clone.sign = new int[this.sign.length];
            clone.alpha = new int[this.alpha.length];
            for(int i = 0; i < this.vecs.length; i++)
            {
                clone.vecs[i] = this.vecs[i].clone();
                clone.sign[i] = this.sign[i];
                clone.alpha[i] = this.alpha[i];
            }
        }
        
        return clone;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(alpha == null)
            throw new UntrainedModelException("Model has not been trained");
        double sum = 0;
        CategoricalResults cr = new CategoricalResults(2);
        
        for (int i = 0; i < vecs.length; i++)
            sum += alpha[i] * sign[i] * kEval(vecs[i], data.getNumericalValues());


        //SVM only says yess / no, can not give a percentage
        if(sum > 0)
            cr.setProb(1, 1);
        else
            cr.setProb(0, 1);
        
        return cr;
    }

    /**
     * Does part of the run through the data to compute the predictoin
     */
    private class PredictPart implements Callable<Double>
    {
        int i;
        int start;
        int end;

        public PredictPart(int i, int start, int end)
        {
            this.i = i;
            this.start = start;
            this.end = end;
        }
        
        @Override
        public Double call() throws Exception
        {
            final double sign_i = sign[i];
            double val = 0;
            for(int j = start; j < end; j++)
            {
                if(j == i || alpha[j] == 0)
                    continue;
                val += alpha[j]*sign_i* kEval(i, j);
            }
            return val;
        }
        
    }
    
    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        if (dataSet.getClassSize() != 2)
            throw new FailedToFitException("Pegasos only supports binary classification problems");
        try
        {
            Random rand = new Random();
            final int m = dataSet.getSampleSize();

            alpha = new int[m];
            sign = new int[m];
            vecs = new Vec[m];
            for (int i = 0; i < dataSet.getSampleSize(); i++)
            {
                vecs[i] = dataSet.getDataPoint(i).getNumericalValues();
                sign[i] = dataSet.getDataPointCategory(i) == 1 ? 1 : -1;
            }

            List<Future<Double>> futures = new ArrayList<Future<Double>>(SystemInfo.LogicalCores);
            final int blockSize = m / SystemInfo.LogicalCores + (m % SystemInfo.LogicalCores == 0 ? 0 : 1);//1 extra if we want

            for (int t = 1; t <= iterations; t++)
            {

                final int i = rand.nextInt(m);
                final double sign_i = sign[i];
                double val = 0;
                
                //distribute work 
                futures.clear();
                int start = 0;
                while (start < m)
                {
                    futures.add(threadPool.submit(new PredictPart(i, start, Math.min(start + blockSize, m))));
                    start += blockSize;
                }
                //collect
                for(Future<Double> partialVal : futures)
                    val += partialVal.get();
                val *= sign_i / (regularization * t);

                if (val < 1)
                    alpha[i]++;

            }

            //Collect the non zero alphas
            int pos = 0;
            for (int i = 0; i < alpha.length; i++)
                if (alpha[i] != 0)
                {
                    alpha[pos] = alpha[i];
                    sign[pos] = sign[i];
                    vecs[pos] = vecs[i];
                    pos++;
                }
            
            alpha = Arrays.copyOf(alpha, pos);
            sign = Arrays.copyOf(sign, pos);
            vecs = Arrays.copyOf(vecs, pos);

        }
        catch (ExecutionException ex)
        {
            throw new FailedToFitException(ex);
        }
        catch (InterruptedException ex)
        {
            throw new FailedToFitException(ex);
        }
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public List<Parameter> getParameters()
    {
        List<Parameter> toRet = new ArrayList<Parameter>(params);
        toRet.addAll(getKernel().getParameters());
        return toRet;
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        Parameter toRet = paramMap.get(paramName);
        
        if(toRet == null)
            return getKernel().getParameter(paramName);
        
        return toRet;
    }
    
}
