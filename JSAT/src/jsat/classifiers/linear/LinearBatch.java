package jsat.classifiers.linear;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;
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
import jsat.math.FunctionP;
import jsat.math.FunctionVec;
import jsat.math.optimization.*;
import jsat.parameters.Parameter;
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
    private Optimizer2 optimizer;
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
    public LinearBatch(LossFunc loss, double lambda0, double tolerance, Optimizer2 optimizer)
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
    public void setOptimizer(Optimizer2 optimizer)
    {
        this.optimizer = optimizer;
    }

    /**
     * Returns the optimization method in use, or {@code null}. 
     * @return the optimization method in use, or {@code null}. 
     */
    public Optimizer2 getOptimizer()
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
    public void trainC(ClassificationDataSet dataSet, Classifier warmSolution)
    {
        trainC(dataSet, warmSolution, null);
    }
    
    @Override
    public void trainC(final ClassificationDataSet D, final ExecutorService threadPool)
    {
        trainC(D, null, threadPool);
    }

    
    @Override
    public void trainC(ClassificationDataSet D, Classifier warmSolution, ExecutorService threadPool)
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

        Optimizer2 optimizerToUse;
        if(optimizer == null)
            optimizerToUse = new LBFGS(10);
        else
            optimizerToUse = optimizer.clone();
        
        doWarmStartIfNotNull(warmSolution);
        
        if(ws.length == 1)
        {
            if(useBiasTerm)
            {
                //Special wrapper class that will handle it - tight coupling with the implementation of LossFun and GradFunc
                Vec w_tmp = new VecWithBias(ws[0], bs);
                optimizerToUse.optimize(tolerance, w_tmp, w_tmp, new LossFunction(D, loss), new GradFunction(D, loss), null, threadPool);
            }
            else
                optimizerToUse.optimize(tolerance, ws[0], ws[0], new LossFunction(D, loss), new GradFunction(D, loss), null, threadPool);
        }
        else
        {
            LossMC lossMC = (LossMC) loss;
            ConcatenatedVec wAll;
            if(useBiasTerm)//append bias terms and logic in the Loss and Grad functions wil handle it
            {
                ArrayList<Vec> vecs = new ArrayList<Vec>(Arrays.asList(ws));
                vecs.add(DenseVector.toDenseVec(bs));
                wAll = new ConcatenatedVec(vecs);
            }
            else
                wAll = new ConcatenatedVec(Arrays.asList(ws));
            optimizerToUse.optimize(tolerance, wAll, new DenseVector(wAll), new LossMCFunction(D, lossMC), new GradMCFunction(D, lossMC), null, threadPool);
        }
        
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
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, (ExecutorService) null);
    }
    
    @Override
    public void train(RegressionDataSet D, ExecutorService threadPool)
    {
        train(D, null, threadPool);
    }

    @Override
    public void train(RegressionDataSet dataSet, Regressor warmSolution)
    {
        train(dataSet, warmSolution, null);
    }
    
    @Override
    public void train(RegressionDataSet D, Regressor warmSolution, ExecutorService threadPool)
    {
        if(D.getNumNumericalVars() <= 0)
            throw new FailedToFitException("LinearBath requires numeric features to work");
        if(!(loss instanceof LossR))
            throw new FailedToFitException("Loss function " + loss.getClass().getSimpleName() + " does not regression");
        ws = new Vec[]{ new DenseVector(D.getNumNumericalVars()) };
        bs = new double[1];
        
        Optimizer2 optimizerToUse;
        if(optimizer == null)
            optimizerToUse = new LBFGS(10);
        else
            optimizerToUse = optimizer.clone();
        
        doWarmStartIfNotNull(warmSolution);
        
        if(useBiasTerm)
        {
            Vec w_tmp = new VecWithBias(ws[0], bs);
            optimizerToUse.optimize(tolerance, w_tmp, w_tmp, new LossFunction(D, loss), new GradFunction(D, loss), null, threadPool);
        }
        else
            optimizerToUse.optimize(tolerance, ws[0], ws[0], new LossFunction(D, loss), new GradFunction(D, loss), null, threadPool);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, (ExecutorService)null);
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
        
    }
    /**
     * Function for using the single weight vector loss functions related to 
     * {@link LossC} and {@link LossR}. 
     */
    public class LossFunction implements FunctionP
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
        public double f(Vec w)
        {
            double sum = 0;
            double weightSum = 0;
            for (int i = 0; i < D.getSampleSize(); i++)
            {
                DataPoint dp = D.getDataPoint(i);
                Vec x = dp.getNumericalValues();
                double y = getTargetY(D, i);
                sum += loss.getLoss(w.dot(x), y)*dp.getWeight();
                weightSum += dp.getWeight();
            }
            if(lambda0 > 0)
                return sum/weightSum + lambda0*w.dot(w);
            else
                return sum/weightSum;
        }
            
        @Override
        public double f(final Vec w, ExecutorService ex)
        {
            final int N = D.getSampleSize();
            final int P = SystemInfo.LogicalCores;
            final double[] weightSums = new double[P];
            List<Future<Double>> partialSums = new ArrayList<Future<Double>>(P);
            for (int p = 0; p < SystemInfo.LogicalCores; p++)
            {
                final int ID = p;
                partialSums.add(ex.submit(new Callable<Double>()
                {
                    @Override
                    public Double call() throws Exception
                    {
                        double sum = 0;
                        double weightSum = 0;
                        for (int i = ParallelUtils.getStartBlock(N, ID, P); i < ParallelUtils.getEndBlock(N, ID, P); i++)
                        {
                            DataPoint dp = D.getDataPoint(i);
                            Vec x = dp.getNumericalValues();
                            double y = getTargetY(D, i);
                            sum += loss.getLoss(w.dot(x), y)*dp.getWeight();
                            weightSum += dp.getWeight();
                        }
                        weightSums[ID] = weightSum;
                        return sum;
                    }
                }));
            }
            double sum = 0;
            try
            {
                for (Double partial : ListUtils.collectFutures(partialSums))
                    sum += partial;
            }
            catch (ExecutionException ex1)
            {
                Logger.getLogger(LinearBatch.class.getName()).log(Level.SEVERE, null, ex1);
            }
            catch (InterruptedException ex1)
            {
                Logger.getLogger(LinearBatch.class.getName()).log(Level.SEVERE, null, ex1);
            }

            double weightSum = 0;
            for(double ws : weightSums)
                weightSum += ws;
            if(lambda0 > 0)
                return sum/weightSum + lambda0*w.dot(w);
            else
                return sum/weightSum;
        }

        @Override
        public double f(double... x)
        {
            return f(DenseVector.toDenseVec(x));
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
        public Vec f(double... x)
        {
            return f(DenseVector.toDenseVec(x));
        }

        @Override
        public Vec f(Vec w)
        {
            Vec s = w.clone();
            f(w, s);
            return s;
        }

        @Override
        public Vec f(Vec w, Vec s)
        {
            if(s == null)
                s = w.clone();
            s.zeroOut();
            double weightSum = 0;
            for (int i = 0; i < D.getSampleSize(); i++)
            {
                DataPoint dp = D.getDataPoint(i);
                Vec x = dp.getNumericalValues();
                double y = getTargetY(D, i);
                s.mutableAdd(loss.getDeriv(w.dot(x), y)*dp.getWeight(), x);
                weightSum += dp.getWeight();
            }
            s.mutableDivide(weightSum);
            if(lambda0 > 0)
                s.mutableSubtract(lambda0, w);
            return s;
        }

        @Override
        public Vec f(final Vec w, Vec s, ExecutorService ex)
        {
            if (s == null)
                s = w.clone();
            s.zeroOut();
            if (tempVecs == null)
                tempVecs = new ThreadLocal<Vec>()
                {
                    @Override
                    protected Vec initialValue()
                    {
                        return w.clone();
                    }
                };
            final Vec store = s;
            final int N = D.getSampleSize();
            final int P = SystemInfo.LogicalCores;
            final CountDownLatch latch = new CountDownLatch(P);
            final double[] weightSums = new double[P];
            for (int p = 0; p < SystemInfo.LogicalCores; p++)
            {
                final int ID = p;
                ex.submit(new Runnable()
                {
                    @Override
                    public void run()
                    {
                        Vec temp = tempVecs.get();
                        temp.zeroOut();
                        double weightSum = 0;
                        for (int i = ParallelUtils.getStartBlock(N, ID, P); i < ParallelUtils.getEndBlock(N, ID, P); i++)
                        {
                            DataPoint dp = D.getDataPoint(i);
                            Vec x = dp.getNumericalValues();
                            double y = getTargetY(D, i);
                            temp.mutableAdd(loss.getDeriv(w.dot(x), y)*dp.getWeight(), x);
                            weightSum += dp.getWeight();
                        }
                        synchronized (store)
                        {
                            store.mutableAdd(temp);
                        }
                        weightSums[ID] = weightSum;
                        latch.countDown();
                    }
                });
            }
            
            try
            {
                latch.await();
            }
            catch (InterruptedException ex1)
            {
                Logger.getLogger(LinearBatch.class.getName()).log(Level.SEVERE, null, ex1);
            }

            double weightSum = 0;
            for(double ws : weightSums)
                weightSum += ws;
            s.mutableDivide(weightSum);
            if(lambda0 > 0)
                s.mutableSubtract(lambda0, w);
            return s;
        }
    }
    
    public class LossMCFunction implements FunctionP
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
        public double f(Vec w)
        {
            double sum = 0;
            Vec pred = new DenseVector(D.getClassSize());//store the predictions in
            //bias terms are at the end, treat them seperate and special
            final int subWSize = (w.length() - (useBiasTerm ? bs.length : 0) )/D.getClassSize();
            double weightSum = 0;
            for (int i = 0; i < D.getSampleSize(); i++)
            {
                DataPoint dp = D.getDataPoint(i);
                Vec x = dp.getNumericalValues();
                for(int k = 0; k < pred.length(); k++)
                    pred.set(k, new SubVector(k*subWSize, subWSize, w).dot(x));
                if(useBiasTerm)
                    pred.mutableAdd(new SubVector(w.length()-bs.length, bs.length, w));
                loss.process(pred, pred);
                int y = D.getDataPointCategory(i);
                sum += loss.getLoss(pred, y)*dp.getWeight();
                weightSum += dp.getWeight();
            }
            if(lambda0 > 0 )
                return sum/weightSum + lambda0*w.dot(w);
            return sum;
        }

        @Override
        public double f(final Vec w, ExecutorService ex)
        {
            final int N = D.getSampleSize();
            final int P = SystemInfo.LogicalCores;
            final int subWSize = (w.length() - (useBiasTerm ? bs.length : 0) )/D.getClassSize();
            List<Future<Double>> partialSums = new ArrayList<Future<Double>>(P);
            final double[] weightSums = new double[P];
            for (int p = 0; p < SystemInfo.LogicalCores; p++)
            {
                final int ID = p;
                partialSums.add(ex.submit(new Callable<Double>()
                {
                    @Override
                    public Double call() throws Exception
                    {
                        double sum = 0;
                        Vec pred = new DenseVector(D.getClassSize());//store the predictions in
                        double weightSum = 0;
                        for (int i = ParallelUtils.getStartBlock(N, ID, P); i < ParallelUtils.getEndBlock(N, ID, P); i++)
                        {
                            DataPoint dp = D.getDataPoint(i);
                            Vec x = dp.getNumericalValues();
                            for(int k = 0; k < pred.length(); k++)
                                pred.set(k, new SubVector(k * subWSize, subWSize, w).dot(x));
                            if(useBiasTerm)
                                pred.mutableAdd(new SubVector(w.length()-bs.length, bs.length, w));
                            loss.process(pred, pred);
                            int y = D.getDataPointCategory(i);
                            sum += loss.getLoss(pred, y)*dp.getWeight();
                            weightSum += dp.getWeight();
                        }

                        weightSums[ID] = weightSum;
                        return sum;
                    }
                }));
            }
            double sum = 0;
            
            try
            {
                for (Double partial : ListUtils.collectFutures(partialSums))
                    sum += partial;
            }
            catch (ExecutionException ex1)
            {
                Logger.getLogger(LinearBatch.class.getName()).log(Level.SEVERE, null, ex1);
            }
            catch (InterruptedException ex1)
            {
                Logger.getLogger(LinearBatch.class.getName()).log(Level.SEVERE, null, ex1);
            }

            double weightSum = 0;
            for(double ws : weightSums)
                weightSum += ws;

            return sum/weightSum + lambda0*w.dot(w);
        }

        @Override
        public double f(double... x)
        {
            return f(DenseVector.toDenseVec(x));
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
        public Vec f(double... x)
        {
            return f(DenseVector.toDenseVec(x));
        }

        @Override
        public Vec f(Vec w)
        {
            Vec s = w.clone();
            f(w, s);
            return s;
        }

        @Override
        public Vec f(Vec w, Vec s)
        {
            if(s == null)
                s = w.clone();
            s.zeroOut();
            
            Vec pred = new DenseVector(D.getClassSize());//store the predictions in
            final int subWSize = (w.length() - (useBiasTerm ? bs.length : 0) )/D.getClassSize();
            double weightSum = 0;
            for (int i = 0; i < D.getSampleSize(); i++)
            {
                DataPoint dp = D.getDataPoint(i);
                Vec x = dp.getNumericalValues();
                for(int k = 0; k < pred.length(); k++)
                    pred.set(k, new SubVector(k*subWSize, subWSize, w).dot(x));
                if(useBiasTerm)
                    pred.mutableAdd(new SubVector(w.length()-bs.length, bs.length, w));
                loss.process(pred, pred);
                int y = D.getDataPointCategory(i);
                loss.deriv(pred, pred, y);
                for(int k = 0; k < pred.length(); k++)
                    new SubVector(k*subWSize, subWSize, s).mutableAdd(pred.get(k)*dp.getWeight(), x);
                weightSum += dp.getWeight();
            }
            s.mutableDivide(weightSum);
            if(lambda0 > 0)
                s.mutableSubtract(lambda0, w);
            return s;
        }

        @Override
        public Vec f(final Vec w, Vec s, ExecutorService ex)
        {
            if (s == null)
                s = w.clone();
            s.zeroOut();
            if (tempVecs == null)
                tempVecs = new ThreadLocal<Vec>()
                {
                    @Override
                    protected Vec initialValue()
                    {
                        return w.clone();
                    }
                };
            final Vec store = s;
            final int N = D.getSampleSize();
            final int P = SystemInfo.LogicalCores;
            final int subWSize = (w.length() - (useBiasTerm ? bs.length : 0) )/D.getClassSize();
            final CountDownLatch latch = new CountDownLatch(P);
            final double[] weightSums = new double[P];
            for (int p = 0; p < SystemInfo.LogicalCores; p++)
            {
                final int ID = p;
                ex.submit(new Runnable()
                {
                    @Override
                    public void run()
                    {
                        Vec temp = tempVecs.get();
                        temp.zeroOut();
                        Vec pred = new DenseVector(D.getClassSize());//store the predictions in
                        double weightSum = 0;
                        for (int i = ParallelUtils.getStartBlock(N, ID, P); i < ParallelUtils.getEndBlock(N, ID, P); i++)
                        {
                            DataPoint dp = D.getDataPoint(i);
                            Vec x = dp.getNumericalValues();
                            for (int k = 0; k < pred.length(); k++)
                                pred.set(k, new SubVector(k * subWSize, subWSize, w).dot(x));
                            if(useBiasTerm)
                                pred.mutableAdd(new SubVector(w.length()-bs.length, bs.length, w));
                            loss.process(pred, pred);
                            int y = D.getDataPointCategory(i);
                            loss.deriv(pred, pred, y);
                            for(IndexValue iv : pred)
                                new SubVector(iv.getIndex() * subWSize, subWSize, temp).mutableAdd(iv.getValue()*dp.getWeight(), x);
                            weightSum += dp.getWeight();
                        }
                        synchronized (store)
                        {
                            store.mutableAdd(temp);
                        }
                        weightSums[ID] = weightSum;
                        latch.countDown();
                    }
                });
            }
            
            try
            {
                latch.await();
            }
            catch (InterruptedException ex1)
            {
                Logger.getLogger(LinearBatch.class.getName()).log(Level.SEVERE, null, ex1);
            }
            
            double weightSum = 0;
            for(double ws : weightSums)
                weightSum += ws;

            s.mutableDivide(weightSum);
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
