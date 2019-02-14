package jsat.classifiers.linear;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.SimpleWeightVectorModel;
import jsat.classifiers.*;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.exceptions.FailedToFitException;
import jsat.linear.ConcatenatedVec;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.SubVector;
import jsat.linear.Vec;
import jsat.lossfunctions.LossC;
import jsat.lossfunctions.LossFunc;
import jsat.lossfunctions.LossMC;
import jsat.lossfunctions.LossR;
import jsat.lossfunctions.SoftmaxLoss;
import jsat.math.Function;
import jsat.math.FunctionVec;
import jsat.math.optimization.*;
import jsat.parameters.Parameterized;
import jsat.regression.*;
import jsat.utils.ListUtils;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.ParallelUtils;

/**
 * LinearBatch learns either a classification or regression problem depending on 
 * the {@link #setLoss(jsat.lossfunctions.LossFunc) loss function &#8467;(w,x)}
 * used. The solution attempts to minimize 
 * <big>&sum;</big><sub>i</sub> &#8467;(w,x<sub>i</sub>) + 
 * {@link #setLambda0(double) &lambda;<sub>0</sub>}/2 ||w||<sub>2</sub><sup>2</sup>, and is 
 * trained using a batch optimization method. <br>
 * <br>
 * LinearBatch can be warm started from any model implementing the
 * {@link SimpleWeightVectorModel} interface. 
 * <br>
 * <br>
 * Note: the current implementation does not currently use bias terms
 * @author Edward Raff
 */
public class LinearBatch implements Classifier, Regressor, Parameterized, SimpleWeightVectorModel, WarmClassifier, WarmRegressor
{

    private static final long serialVersionUID = -446156124954287580L;
    /**
     * Weight vectors 
     */
    private Vec[] ws;
    /**
     * bias terms for each weight vector
     */
    private double[] bs;
    private LossFunc loss;
    private double lambda0;
    private Optimizer optimizer;
    private double tolerance;
    private boolean useBiasTerm = true;

    /**
     * Creates a new Linear Batch learner for classification using a small 
     * regularization term
     */
    public LinearBatch()
    {
        this(new SoftmaxLoss(), 1e-6);
    }
    
    /**
     * Creates a new Linear Batch learner
     * @param loss the loss function to use
     * @param lambda0 the L<sub>2</sub> regularization term
     */
    public LinearBatch(LossFunc loss, double lambda0)
    {
        this(loss, lambda0, 1e-3);
    }
    
    /**
     * Creates a new Linear Batch learner
     * @param loss the loss function to use
     * @param lambda0 the L<sub>2</sub> regularization term
     * @param tolerance the threshold for convergence 
     */
    public LinearBatch(LossFunc loss, double lambda0, double tolerance)
    {
        this(loss, lambda0, tolerance, null);
    }
    
    /**
     * Creates a new Linear Batch learner
     * @param loss the loss function to use
     * @param lambda0 the L<sub>2</sub> regularization term
     * @param tolerance the threshold for convergence 
     * @param optimizer the batch optimization method to use
     */
    public LinearBatch(LossFunc loss, double lambda0, double tolerance, Optimizer optimizer)
    {
        setLoss(loss);
        setLambda0(lambda0);
        setOptimizer(optimizer);
        setTolerance(tolerance);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public LinearBatch(LinearBatch toCopy)
    {
        this(toCopy.loss.clone(), toCopy.lambda0, toCopy.tolerance, toCopy.optimizer == null ? null : toCopy.optimizer.clone());
        if(toCopy.ws != null)
        {
            this.ws = new Vec[toCopy.ws.length];
            for(int i = 0; i < toCopy.ws.length; i++)
                this.ws[i] = toCopy.ws[i].clone();
        }
        if(toCopy.bs != null)
            this.bs = Arrays.copyOf(toCopy.bs, toCopy.bs.length);
    }

    public void setUseBiasTerm(boolean useBiasTerm)
    {
        this.useBiasTerm = useBiasTerm;
    }

    public boolean isUseBiasTerm()
    {
        return useBiasTerm;
    }
    
    /**
     * &lambda;<sub>0</sub> controls the L<sub>2</sub> regularization penalty. 
     * @param lambda0 the L<sub>2</sub> regularization penalty to use
     */
    public void setLambda0(double lambda0)
    {
        if(lambda0 < 0 || Double.isNaN(lambda0) || Double.isInfinite(lambda0))
            throw new IllegalArgumentException("Lambda0 must be non-negative, not " + lambda0);
        this.lambda0 = lambda0;
    }

    /**
     * Returns the L<sub>2</sub> regularization term in use
     * @return the L<sub>2</sub> regularization term in use
     */
    public double getLambda0()
    {
        return lambda0;
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

    /**
     * Sets the method of batch optimization that will be used. {@code null} is 
     * valid for this value, in which case the implementation will attempt to 
     * select a reasonable optimizer automatically. <br>
     * <br>
     * NOTE: the current implementation requires the optimizer to work based off
     * only the function value and its derivative. 
     * 
     * @param optimizer the method to use for function minimization
     */
    public void setOptimizer(Optimizer optimizer)
    {
        this.optimizer = optimizer;
    }

    /**
     * Returns the optimization method in use, or {@code null}. 
     * @return the optimization method in use, or {@code null}. 
     */
    public Optimizer getOptimizer()
    {
        return optimizer;
    }
    
    /**
     * Sets the convergence tolerance to user for training. Smaller values reach
     * a more accuracy solution but may take longer to complete.<br>
     * While zero is a valid tolerance value, it is not usually useful in 
     * practice. Values in [10<sup>-4</sup>, 10<sup>-2</sup>] are usually more 
     * practical. 
     * 
     * @param tolerance the convergence tolerance
     */
    public void setTolerance(double tolerance)
    {
        if(tolerance < 0 || Double.isNaN(tolerance) || Double.isInfinite(tolerance))
            throw new IllegalArgumentException("Tolerance must be a non-negative constant, not " + tolerance);
        this.tolerance = tolerance;
    }

    /**
     * Returns the value of the convergence tolerance parameter
     * @return the convergence tolerance parameter
     */
    public double getTolerance()
    {
        return tolerance;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        final Vec x = data.getNumericalValues();
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
        final Vec x = data.getNumericalValues();
        return ((LossR)loss).getRegression(ws[0].dot(x)+bs[0]);
    }
    
   
    @Override
    public void train(final ClassificationDataSet D, final boolean parallel)
    {
        train(D, null, parallel);
    }
    
    @Override
    public void train(ClassificationDataSet D, Classifier warmSolution, boolean parallel)
    {
        if(D.getNumNumericalVars() <= 0)
            throw new FailedToFitException("LinearBath requires numeric features to work");
        if(!(loss instanceof LossC))
            throw new FailedToFitException("Loss function " + loss.getClass().getSimpleName() + " does not support classification");
        if(D.getClassSize() > 2)
            if (!(loss instanceof LossMC))
                throw new FailedToFitException("Loss function " + loss.getClass().getSimpleName() + " does not support multi-class classification");
            else
            {
                ws = new Vec[D.getClassSize()];
                bs = new double[ws.length];
            }
        else
        {
            ws = new Vec[1];
            bs = new double[1];
        }
        for (int i = 0; i < ws.length; i++)
            ws[i] = new DenseVector(D.getNumNumericalVars());

        Optimizer optimizerToUse;
        if(optimizer == null)
            optimizerToUse = new LBFGS(10);
        else
            optimizerToUse = optimizer.clone();
        
        doWarmStartIfNotNull(warmSolution);
        
        ExecutorService threadPool = ParallelUtils.getNewExecutor(parallel);
        
        if(ws.length == 1)
        {
            if(useBiasTerm)
            {
                //Special wrapper class that will handle it - tight coupling with the implementation of LossFun and GradFunc
                Vec w_tmp = new VecWithBias(ws[0], bs);
                optimizerToUse.optimize(tolerance, w_tmp, w_tmp, new LossFunction(D, loss), new GradFunction(D, loss), parallel);
            }
            else
                optimizerToUse.optimize(tolerance, ws[0], ws[0], new LossFunction(D, loss), new GradFunction(D, loss), parallel);
        }
        else
        {
            LossMC lossMC = (LossMC) loss;
            ConcatenatedVec wAll;
            if(useBiasTerm)//append bias terms and logic in the Loss and Grad functions wil handle it
            {
                ArrayList<Vec> vecs = new ArrayList<>(Arrays.asList(ws));
                vecs.add(DenseVector.toDenseVec(bs));
                wAll = new ConcatenatedVec(vecs);
            }
            else
                wAll = new ConcatenatedVec(Arrays.asList(ws));
            optimizerToUse.optimize(tolerance, wAll, new DenseVector(wAll), new LossMCFunction(D, lossMC), new GradMCFunction(D, lossMC), parallel);
        }
        
        threadPool.shutdownNow();
    }

    /**
     * Performs a warm start if the given object is of the appropriate class.
     * Nothing happens if input it null.
     *
     * @param warmSolution
     * @throws FailedToFitException
     */
    private void doWarmStartIfNotNull(Object warmSolution) throws FailedToFitException
    {
        if(warmSolution != null )
        {
            if(warmSolution instanceof SimpleWeightVectorModel)
            {
                SimpleWeightVectorModel warm = (SimpleWeightVectorModel) warmSolution;
                if(warm.numWeightsVecs() != ws.length)
                    throw new FailedToFitException("Warm solution has " + warm.numWeightsVecs() + " weight vectors instead of " + ws.length);
                for(int i = 0; i < ws.length; i++)
                {
                    warm.getRawWeight(i).copyTo(ws[i]);
                    if(useBiasTerm)
                        bs[i] = warm.getBias(i);
                }
            }
            else
                throw new FailedToFitException("Can not warm warm from " + warmSolution.getClass().getCanonicalName());
        }
    }
    
    @Override
    public void train(RegressionDataSet D, boolean parallel)
    {
        train(D, this, parallel);
    }

    @Override
    public void train(RegressionDataSet dataSet, Regressor warmSolution)
    {
        train(dataSet, warmSolution, false);
    }
    
    @Override
    public void train(RegressionDataSet D, Regressor warmSolution, boolean parallel)
    {
        if(D.getNumNumericalVars() <= 0)
            throw new FailedToFitException("LinearBath requires numeric features to work");
        if(!(loss instanceof LossR))
            throw new FailedToFitException("Loss function " + loss.getClass().getSimpleName() + " does not regression");
        ws = new Vec[]{ new DenseVector(D.getNumNumericalVars()) };
        bs = new double[1];
        
        Optimizer optimizerToUse;
        if(optimizer == null)
            optimizerToUse = new LBFGS(10);
        else
            optimizerToUse = optimizer.clone();
        
        doWarmStartIfNotNull(warmSolution);
        
        ExecutorService threadPool = ParallelUtils.getNewExecutor(parallel);
        
        if(useBiasTerm)
        {
            Vec w_tmp = new VecWithBias(ws[0], bs);
            optimizerToUse.optimize(tolerance, w_tmp, w_tmp, new LossFunction(D, loss), new GradFunction(D, loss), parallel);
        }
        else
            optimizerToUse.optimize(tolerance, ws[0], ws[0], new LossFunction(D, loss), new GradFunction(D, loss), parallel);
        
        threadPool.shutdownNow();
    }

    private static double getTargetY(DataSet D, int i)
    {
        double y;
        if (D instanceof ClassificationDataSet)
            y = ((ClassificationDataSet) D).getDataPointCategory(i) * 2 - 1;
        else
            y = ((RegressionDataSet) D).getTargetValue(i);
        return y;
    }

    @Override
    public boolean warmFromSameDataOnly()
    {
        return false;
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

    private class VecWithBias extends Vec
    {
        public Vec w;
        public double[] b;

        public VecWithBias(Vec w, double[] b)
        {
            this.w = w;
            this.b = b;
        }
        
        //2 hacks below to make the original code work with bias terms "transparently" This means we need to know which functions will be called with a miss-matched size

        @Override
        public double dot(Vec v)
        {
            if(v.length() == w.length())
                return w.dot(v)+b[0];
            return super.dot(v); 
        }

        @Override
        public void mutableAdd(double c, Vec b)
        {
            if(b.length() == w.length())
            {
                w.mutableAdd(c, b);
                this.b[0] += c;
            }
            else
                super.mutableAdd(c, b);
        }

        
        @Override
        public int length()
        {
            return w.length()+1;
        }

        @Override
        public double get(int index)
        {
            if(index < w.length())
                return w.get(index);
            else if (index == w.length())
                return b[0];
            else
                throw new IndexOutOfBoundsException();
        }

        @Override
        public void set(int index, double val)
        {
            if(index < w.length())
                w.set(index, val);
            else if (index == w.length())
                b[0] = val;
            else
                throw new IndexOutOfBoundsException();
        }

        @Override
        public boolean isSparse()
        {
            return w.isSparse();
        }

        @Override
        public Vec clone()
        {
            return new VecWithBias(w.clone(), Arrays.copyOf(b, b.length));
        }

        @Override
        public void setLength(int length)
        {
            throw new UnsupportedOperationException("Not supported yet."); 
        }
        
    }
    /**
     * Function for using the single weight vector loss functions related to 
     * {@link LossC} and {@link LossR}. 
     */
    public class LossFunction implements Function
    {
        private static final long serialVersionUID = -576682206943283356L;
        private final DataSet D;
        private final LossFunc loss;
        
        public LossFunction(DataSet D, LossFunc loss)
        {
            this.D = D;
            this.loss = loss;
        }
        
        @Override
        public double f(Vec w, boolean parallel)
        {
            DoubleAdder sum = new DoubleAdder();
            DoubleAdder weightSum = new DoubleAdder();
            
            ParallelUtils.run(parallel, D.size(), (start, end)->
            {
                for(int i = start; i < end; i++)
                {
                    DataPoint dp = D.getDataPoint(i);
                    Vec x = dp.getNumericalValues();
                    double y = getTargetY(D, i);
                    sum.add(loss.getLoss(w.dot(x), y)*D.getWeight(i));
                    weightSum.add(D.getWeight(i));
                }
            });
            
            if(lambda0 > 0)
                return sum.sum()/weightSum.sum() + lambda0*w.dot(w);
            else
                return sum.sum()/weightSum.sum();
        }
    }

    /**
     * Function for using the single weight vector loss functions related to 
     * {@link LossC} and {@link LossR}
     */
    public class GradFunction implements FunctionVec
    {
        private final DataSet D;
        private final LossFunc loss;
        private ThreadLocal<Vec> tempVecs;

        public GradFunction(DataSet D, LossFunc loss)
        {
            this.D = D;
            this.loss = loss;
        }
        
        @Override
        public Vec f(Vec w, Vec s, boolean parallel)
        {
            if(s == null)
                s = w.clone();
            s.zeroOut();
            DoubleAdder weightSum = new DoubleAdder();
            ThreadLocal<Vec> tl_s = ThreadLocal.withInitial(s::clone);
            
            ParallelUtils.run(parallel, D.size(), (start, end)->
            {
                Vec s_l = tl_s.get();
                
                for (int i = start; i < end; i++)
                {
                    DataPoint dp = D.getDataPoint(i);
                    Vec x = dp.getNumericalValues();
                    double y = getTargetY(D, i);
                    s_l.mutableAdd(loss.getDeriv(w.dot(x), y)*D.getWeight(i), x);
                    weightSum.add(D.getWeight(i));
                }
                
                return s_l;
            }, (a,b)->a.add(b))
                    .copyTo(s);
            
            s.mutableDivide(weightSum.sum());
            if(lambda0 > 0)
                s.mutableSubtract(lambda0, w);
            return s;
        }
    }
    
    public class LossMCFunction implements Function
    {
        private static final long serialVersionUID = -861700500356609563L;
        private final ClassificationDataSet D;
        private final LossMC loss;

        public LossMCFunction(ClassificationDataSet D, LossMC loss)
        {
            this.D = D;
            this.loss = loss;
        }
        
        @Override
        public double f(Vec w, boolean parallel)
        {
            DoubleAdder sum = new DoubleAdder();
            Vec pred = new DenseVector(D.getClassSize());//store the predictions in
            //bias terms are at the end, treat them seperate and special
            final int subWSize = (w.length() - (useBiasTerm ? bs.length : 0) )/D.getClassSize();
            DoubleAdder weightSum = new DoubleAdder();
            ParallelUtils.run(parallel, D.size(), (start, end)->
            {
                Vec pred_local = pred.clone();
                for (int i = start; i < end; i++)
                {
                    DataPoint dp = D.getDataPoint(i);
                    Vec x = dp.getNumericalValues();
                    for(int k = 0; k < pred_local.length(); k++)
                        pred_local.set(k, new SubVector(k*subWSize, subWSize, w).dot(x));
                    if(useBiasTerm)
                        pred_local.mutableAdd(new SubVector(w.length()-bs.length, bs.length, w));
                    loss.process(pred_local, pred_local);
                    int y = D.getDataPointCategory(i);
                    sum.add(loss.getLoss(pred_local, y)*D.getWeight(i));
                    weightSum.add(D.getWeight(i));
                }
            });
            if(lambda0 > 0 )
                return sum.sum()/weightSum.sum() + lambda0*w.dot(w);
            return sum.sum();
        }
    }
    
    private class GradMCFunction implements FunctionVec
    {
        private final ClassificationDataSet D;
        private final LossMC loss;
        private ThreadLocal<Vec> tempVecs;

        public GradMCFunction(ClassificationDataSet D, LossMC loss)
        {
            this.D = D;
            this.loss = loss;
        }

        @Override
        public Vec f(Vec w, Vec s, boolean parllel)
        {
            if(s == null)
                s = w.clone();
            s.zeroOut();
            
            ThreadLocal<Vec> tl_s = ThreadLocal.withInitial(s::clone);
            
            Vec pred = new DenseVector(D.getClassSize());//store the predictions in
            final int subWSize = (w.length() - (useBiasTerm ? bs.length : 0) )/D.getClassSize();
            DoubleAdder weightSum = new DoubleAdder();
            ParallelUtils.run(parllel, D.size(), (start, end)->
            {
                Vec s_l = tl_s.get();
                Vec pred_local = pred.clone();
                for (int i = start; i < end; i++)
                {
                    DataPoint dp = D.getDataPoint(i);
                    Vec x = dp.getNumericalValues();
                    for(int k = 0; k < pred_local.length(); k++)
                        pred_local.set(k, new SubVector(k*subWSize, subWSize, w).dot(x));
                    if(useBiasTerm)
                        pred_local.mutableAdd(new SubVector(w.length()-bs.length, bs.length, w));
                    loss.process(pred_local, pred_local);
                    int y = D.getDataPointCategory(i);
                    loss.deriv(pred_local, pred_local, y);
                    for(int k = 0; k < pred_local.length(); k++)
                        new SubVector(k*subWSize, subWSize, s_l).mutableAdd(pred_local.get(k)*D.getWeight(i), x);
                    weightSum.add(D.getWeight(i));
                }
                return s_l;
            }, (a,b)->a.add(b)).copyTo(s);
            s.mutableDivide(weightSum.sum());
            if(lambda0 > 0)
                s.mutableSubtract(lambda0, w);
            return s;
        }
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }

    @Override
    public LinearBatch clone()
    {
        return new LinearBatch(this);
    }
    
     /**
     * Guess the distribution to use for the regularization term
     * {@link #setLambda0(double) &lambda;<sub>0</sub>} .
     *
     * @param d the data set to get the guess for
     * @return the guess for the &lambda;<sub>0</sub> parameter
     */
    public static Distribution guessLambda0(DataSet d)
    {
        return new LogUniform(1e-7, 1e-2);
    }
    
}
