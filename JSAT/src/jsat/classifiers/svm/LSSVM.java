package jsat.classifiers.svm;

import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.distributions.kernels.KernelTrick;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import static java.lang.Math.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.distributions.Distribution;
import jsat.distributions.kernels.LinearKernel;
import jsat.exceptions.FailedToFitException;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.parameters.Parameter.WarmParameter;
import jsat.regression.*;
import jsat.utils.FakeExecutor;
import jsat.utils.PairedReturn;
import jsat.utils.SystemInfo;
import static jsat.utils.concurrent.ParallelUtils.*;


/**
 * The Least Squares Support Vector Machine (LS-SVM) is an alternative to the 
 * standard SVM classifier for regression and binary classification problems. It
 * can be faster to train, but is usually significantly slower to perform 
 * predictions with. This is because the LS-SVM solution is dense, so all 
 * training points become support vectors. <br>
 * <br>
 * The LS-SVM algorithm may be warm started only from another LS-SVM object 
 * trained on the same data set. <br>
 * <br>
 * NOTE: A SMO implementation similar to {@link PlattSMO} is used. This is done 
 * because is can easily operate without explicitly forming the whole kernel 
 * matrix. However it is recommended to use the LS-SVM when the problem size is 
 * small enough such that {@link SupportVectorLearner.CacheMode#FULL} can be 
 * used. <br>
 * <br>
 * If <i>N</i> is the number of data points:<br>
 * <ul>
 * <li>Training complexity is roughly O(n^3), but can be lower for small C</li>
 * <li>Prediction complexity is O(n)</li>
 * <li>This implementation is multi-threaded, but scales best when there are 
 * several thousand data points per core. For smaller problems, especially when 
 * full cache mode can be used, there may be negative speedups when using the 
 * parallel training methods</li>
 * </ul>
 * <br>
 * See: <br>
 * <ul>
 * <li>Suykens, J.,&amp;Vandewalle, J. (1999). <i>Least Squares Support Vector 
 * Machine Classifiers</i>. Neural processing letters, 9(3), 293–298. 
 * doi:10.1023/A:1018628609742</li>
 * <li>Keerthi, S. S.,&amp;Shevade, S. K. (2003). <i>SMO algorithm for Least 
 * Squares SVM</i>. In Proceedings of the International Joint Conference on 
 * Neural Networks (Vol. 3, pp. 2088–2093). IEEE. doi:10.1109/IJCNN.2003.1223730
 * </li>
 * </ul>
 * 
 * 
 * @author Edward Raff
 */
public class LSSVM extends SupportVectorLearner implements BinaryScoreClassifier, Regressor, Parameterized, WarmRegressor, WarmClassifier
{

    private static final long serialVersionUID = -7569924400631719451L;
    protected double b = 0, b_low, b_up;
    private double C = 1;

    private int i_up, i_low;
    private double[] fcache;
    private double dualObjective;
    
    private static double epsilon = 1e-12;
    private static double tol = 1e-3;
    
    /**
     * Creates a new LS-SVM learner that uses a linear model and does not use a
     * cache
     */
    public LSSVM()
    {
        this(new LinearKernel());
    }
    
    /**
     * Creates a new LS-SVM learner that does not use a cache
     * @param kernel the kernel method to use
     */
    public LSSVM(KernelTrick kernel)
    {
        this(kernel, CacheMode.NONE);
    }
    
    /**
     * Creates a new LS-SVM learner
     * @param kernel the kernel method to use
     * @param cacheMode the caching scheme to use for kernel evaluations
     */
    public LSSVM(KernelTrick kernel, CacheMode cacheMode)
    {
        super(kernel, cacheMode);
    }

    /**
     * Creates a deep copy of another LS-SVM 
     * @param toCopy the object to copy
     */
    public LSSVM(LSSVM toCopy)
    {
        super(toCopy.getKernel().clone(), toCopy.getCacheMode());
        this.b_low = toCopy.b_low;
        this.b_up = toCopy.b_up;
        this.i_up = toCopy.i_up;
        this.i_low = toCopy.i_low;
        this.C = toCopy.C;
        if(toCopy.alphas != null)
            this.alphas = Arrays.copyOf(toCopy.alphas, toCopy.alphas.length);
        if(toCopy.fcache != null)
            this.fcache = Arrays.copyOf(toCopy.fcache, toCopy.fcache.length);
    }
    

    /**
     * Sets the regularization constant when training. Lower values correspond 
     * to higher amounts of regularization. 
     * @param C the positive regularization parameter
     */
    @WarmParameter(prefLowToHigh = true)
    public void setC(double C)
    {
        if(C <= 0 || Double.isNaN(C) || Double.isInfinite(C))
            throw new IllegalArgumentException("C must be in (0, Infty), not " + C);
        this.C = C;
    }

    /**
     * Returns the regularization parameter value used
     * @return the regularization parameter value
     */
    public double getC()
    {
        return C;
    }


    private boolean takeStep(int i1, int i2, ExecutorService ex, int P) throws InterruptedException, ExecutionException
    {
        //these 2 will hold the old values
        final double alph1 = alphas[i1];
        final double alph2 = alphas[i2];
        double F1 = fcache[i1];
        double F2 = fcache[i2];
        double gamma = alph1+alph2;
        
        final double k11 = kEval(i1, i1);
        final double k12 = kEval(i2, i1);
        final double k22 = kEval(i2, i2);
        
        final double eta = 2*k12-k11-k22;
        final double a2 = alph2-(F1-F2)/eta;
        if(abs(a2-alph2) < epsilon*(a2+alph2+epsilon))
            return false;
        final double a1 = gamma-a2;
        alphas[i1] = a1;
        alphas[i2] = a2;
        
        //Update the DualObjectiveFunction using (4.11)
        double t = (F1-F2)/eta;
        dualObjective -= eta/2*t*t;
        
        //2 steps done in the same loop
        b_up = Double.NEGATIVE_INFINITY;
        b_low = Double.POSITIVE_INFINITY;
        
        //Update Fcache[i] for all i in I using (4.10)
        //Compute (i_low, b_low) and (i_up, b_up) using (3.4)
        
        List<Future<PairedReturn<Integer, Integer>>> futures = new ArrayList<Future<PairedReturn<Integer, Integer>>>(P);
        for(int id = 0; id < P; id++)
        {
            int from = getStartBlock(fcache.length, id, P);
            int to = getEndBlock(fcache.length, id, P);
            futures.add(ex.submit(new TakeStepLoop(from, to, i1, i2, alph1, alph2)));
        }
        for(Future<PairedReturn<Integer, Integer>> fpr : futures)
        {
            PairedReturn<Integer, Integer> pr = fpr.get();
            int i_up_cand = pr.getFirstItem();
            int i_low_cand = pr.getSecondItem();
            if(fcache[i_up_cand] > b_up)
            {
                b_up = fcache[i_up_cand];
                i_up = i_up_cand;
            }
            
            if(fcache[i_low_cand] < b_low)
            {
                b_low = fcache[i_low_cand];
                i_low = i_low_cand;
            }
        }
        
        return true;
    }

    @Override
    public boolean warmFromSameDataOnly()
    {
        return true;
    }
    
    private class TakeStepLoop implements Callable<PairedReturn<Integer, Integer>>
    {
        int from, to;
        int i1, i2;
        double alph1, alph2;
        int i_low_p;
        int i_up_p;

        public TakeStepLoop(int from, int to, int i1, int i2, double alph1, double alph2)
        {
            this.from = from;
            this.to = to;
            this.i1 = i1;
            this.i2 = i2;
            this.alph1 = alph1;
            this.alph2 = alph2;
        }

        @Override
        public PairedReturn<Integer, Integer> call() throws Exception
        {
            final double a1 = alphas[i1], a2 = alphas[i2];
            double b_up_p = Double.NEGATIVE_INFINITY, b_low_p = Double.POSITIVE_INFINITY;
            for (int i = from; i < to; i++)
            {
                final double k_i1 = kEval(i1, i);
                final double k_i2 = kEval(i2, i);
                final double Fi = (fcache[i] += (a1 - alph1) * k_i1 + (a2 - alph2) * k_i2);
                if(Fi > b_up_p)
                {
                    b_up_p = Fi;
                    i_up_p = i;
                }

                if(Fi < b_low_p)
                {
                    b_low_p = Fi;
                    i_low_p = i;
                }
            }
            
            return new PairedReturn<Integer, Integer>(i_up_p, i_low_p);
        }
    }
    
    private class BiasGapCallable implements Callable<Double>
    {
        int from, to;

        public BiasGapCallable(int from, int to)
        {
            this.from = from;
            this.to = to;
        }
        
        @Override
        public Double call() throws Exception
        {
            double B = 0;
            for(int i = from; i < to; i++)
                B += fcache[i]-alphas[i]/C;
            return B;
        }
    }
    
    private class DualityGapCallable implements Callable<Double>
    {
        int from, to;

        public DualityGapCallable(int from, int to)
        {
            this.from = from;
            this.to = to;
        }

        @Override
        public Double call() throws Exception
        {
            double gap = 0;
            for(int i = from; i < to; i++)
            {
                final double x_i = b + alphas[i]/C - fcache[i];
                gap += alphas[i]*(fcache[i]-(0.5*alphas[i]/C)) + C*x_i*x_i/2;
            }
            return gap;
        }
    }

    private double computeDualityGap(boolean fast, ExecutorService ex, int P) throws InterruptedException, ExecutionException
    {
        double gap = 0;
        //set b using (3.16) or (3.17)
        if(fast)
            b = (b_up+b_low)/2;
        else
        {
            b = 0;
            List<Future<Double>> bParts = new ArrayList<Future<Double>>(P);
            for(int id = 0; id < P; id++)
                bParts.add(ex.submit(new BiasGapCallable(getStartBlock(alphas.length, id, P), getEndBlock(alphas.length, id, P))));
            for(Future<Double> bPart : bParts)
                b += bPart.get();
            b /= alphas.length;
        }
        
        List<Future<Double>> gapParts = new ArrayList<Future<Double>>(P);
        for (int id = 0; id < P; id++)
            gapParts.add(ex.submit(new DualityGapCallable(getStartBlock(alphas.length, id, P), getEndBlock(alphas.length, id, P))));
        for(Future<Double> gapPart : gapParts)
            gap += gapPart.get();

        return gap;
    }
    
    private void initializeVariables(double[] targets, LSSVM warmSolution, DataSet data)
    {
        alphas = new double[targets.length];
        fcache = new double[targets.length];
        dualObjective = 0;
        if(warmSolution != null)
        {
            if(warmSolution.alphas.length != this.alphas.length)
                throw new FailedToFitException("Warm LS-SVM solution could not have been trained on the sama data, different number of alpha values present");
            double C_ratio = this.C/warmSolution.C;
            for(int i = 0; i < targets.length; i++)
            {
                alphas[i] = warmSolution.alphas[i];
                fcache[i] = warmSolution.fcache[i]-(C_ratio-1)*warmSolution.alphas[i]/(this.C);
                dualObjective += alphas[i]*(targets[i]-fcache[i]);
            }
            dualObjective /= 2;
        }
        else
        {
            for(int i = 0; i < targets.length; i++)
                fcache[i] = -targets[i];
        }
        
        //Compute (i_low, b_low) and (i_up, b_up) using (3.4)
        b_up = Double.NEGATIVE_INFINITY;
        b_low = Double.POSITIVE_INFINITY;
        
        //Update Fcache[i] for all i in I using (4.10)
        //Compute (i_low, b_low) and (i_up, b_up) using (3.4)
        for (int i = 0; i < fcache.length; i++)
        {
            final double Fi = fcache[i];
            if(Fi > b_up)
            {
                b_up = Fi;
                i_up = i;
            }
            
            if(Fi < b_low)
            {
                b_low = Fi;
                i_low = i;
            }
        }
        setCacheMode(getCacheMode());//Initializes the cahce
    }
    
    
    @Override
    public double getScore(DataPoint dp)
    {
        return regress(dp);
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(2);
        if(regress(data) > 0)
            cr.setProb(1, 1.0);
        else
            cr.setProb(0, 1.0);
        return cr;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet, null, threadPool);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, (ExecutorService)null);
    }
    
    @Override
    public void train(RegressionDataSet dataSet, Regressor warmSolution, ExecutorService threadPool)
    {
        if(warmSolution != null && !(warmSolution instanceof LSSVM))
            throw new FailedToFitException("Warm solution must be an implementation of LS-SVM, not " + warmSolution.getClass());
        double[] targets = dataSet.getTargetValues().arrayCopy();
        mainLoop(dataSet, (LSSVM)warmSolution, targets, threadPool);
    }

    @Override
    public void train(RegressionDataSet dataSet, Regressor warmSolution)
    {
        train(dataSet, warmSolution, null);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, Classifier warmSolution, ExecutorService threadPool)
    {
        if(dataSet.getClassSize() != 2)
            throw new FailedToFitException("LS-SVM only supports binary classification problems");
        if(warmSolution != null && !(warmSolution instanceof LSSVM))
            throw new FailedToFitException("Warm solution must be an implementation of LS-SVM, not " + warmSolution.getClass());
        double[] targets = new double[dataSet.getSampleSize()];
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            targets[i] = dataSet.getDataPointCategory(i)*2-1;
        mainLoop(dataSet, (LSSVM) warmSolution , targets, threadPool);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, Classifier warmSolution)
    {
        trainC(dataSet, warmSolution, null);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public double regress(DataPoint data)
    {
        return kEvalSum(data.getNumericalValues())-b;
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        train(dataSet, null, threadPool);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, (ExecutorService)null);
    }

    @Override
    public LSSVM clone()
    {
        return new LSSVM(this);
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

    private void mainLoop(DataSet dataSet, LSSVM warmSolution, double[] targets, ExecutorService ex)
    {
        final int P;
        if(ex == null || ex instanceof FakeExecutor)
        {
            ex = new FakeExecutor();
            P = 1;
        }
        else
            P = SystemInfo.LogicalCores;
        
        try
        {
            vecs = dataSet.getDataVectors();
            initializeVariables(targets, warmSolution, dataSet);
            
            boolean change = true;
            double dualityGap = computeDualityGap(true, ex, P);
            int iter = 0;
            while (dualityGap > tol * dualObjective && change)
            {
                change = takeStep(i_up, i_low, ex, P);
                dualityGap = computeDualityGap(true, ex, P);
                iter++;
            }
            
            setCacheMode(null);
            setAlphas(alphas);
        }
        catch (InterruptedException interruptedException)
        {
            throw new FailedToFitException(interruptedException);
        }
        catch (ExecutionException executionException)
        {
            throw new FailedToFitException(executionException);
        }
    }
    
    /**
     * Guess the distribution to use for the regularization term
     * {@link #setC(double) C} in a LS-SVM.
     *
     * @param d the data set to get the guess for
     * @return the guess for the C parameter in the LS-SVM
     */
    public static Distribution guessC(DataSet d)
    {
        return PlattSMO.guessC(d);//LS-SVM isn't technically the same algo, but still a good search
    }
    
}
