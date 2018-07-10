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
import java.util.stream.IntStream;
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.distributions.Distribution;
import jsat.distributions.kernels.LinearKernel;
import jsat.exceptions.FailedToFitException;
import jsat.parameters.Parameterized;
import jsat.parameters.Parameter.WarmParameter;
import jsat.regression.*;
import jsat.utils.FakeExecutor;
import jsat.utils.PairedReturn;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.ParallelUtils;
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


    private boolean takeStep(int i1, int i2, ExecutorService ex, boolean parallel) throws InterruptedException, ExecutionException
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
        ParallelUtils.run(parallel, fcache.length, (from, to) -> 
        {
            int i_low_cand = from;
            int i_up_cand = from;
            double b_up_p = Double.NEGATIVE_INFINITY, b_low_p = Double.POSITIVE_INFINITY;
            for (int i = from; i < to; i++)
            {
                final double k_i1 = kEval(i1, i);
                final double k_i2 = kEval(i2, i);
                final double Fi = (fcache[i] += (a1 - alph1) * k_i1 + (a2 - alph2) * k_i2);
                if(Fi > b_up_p)
                {
                    b_up_p = Fi;
                    i_up_cand = i;
                }

                if(Fi < b_low_p)
                {
                    b_low_p = Fi;
                    i_low_cand = i;
                }
            }
       
            synchronized (fcache)
            {
                if (fcache[i_up_cand] > b_up)
                {
                    b_up = fcache[i_up_cand];
                    i_up = i_up_cand;
                }

                if (fcache[i_low_cand] < b_low)
                {
                    b_low = fcache[i_low_cand];
                    i_low = i_low_cand;
                }
            }
        }, ex);
        
        return true;
    }

    @Override
    public boolean warmFromSameDataOnly()
    {
        return true;
    }

    private double computeDualityGap(boolean fast, boolean parallel) throws InterruptedException, ExecutionException
    {
        //Below we use the IntStream rather than parallelUtil's range b/c the sequence should be long enough to actually get parallelism, and it will use less memory
        double gap = 0;
        //set b using (3.16) or (3.17)
        if(fast)
            b = (b_up+b_low)/2;
        else
        {
            b = ParallelUtils.streamP(IntStream.range(0, alphas.length), parallel).mapToDouble( i ->
            {
                return fcache[i]-alphas[i]/C;
            }).sum();
            b /= alphas.length;
        }

        gap = ParallelUtils.streamP(IntStream.range(0, alphas.length), parallel).mapToDouble( i ->
        {
            final double x_i = b + alphas[i]/C - fcache[i];
            return alphas[i]*(fcache[i]-(0.5*alphas[i]/C)) + C*x_i*x_i/2;
        }).sum();

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
    public void train(ClassificationDataSet dataSet, boolean parallel)
    {
        train(dataSet, null, parallel);
    }
    
    @Override
    public void train(RegressionDataSet dataSet, Regressor warmSolution, boolean parallel)
    {
        if(warmSolution != null && !(warmSolution instanceof LSSVM))
            throw new FailedToFitException("Warm solution must be an implementation of LS-SVM, not " + warmSolution.getClass());
        double[] targets = dataSet.getTargetValues().arrayCopy();
        mainLoop(dataSet, (LSSVM)warmSolution, targets, parallel);
    }

    @Override
    public void train(RegressionDataSet dataSet, Regressor warmSolution)
    {
        train(dataSet, warmSolution, false);
    }

    @Override
    public void train(ClassificationDataSet dataSet, Classifier warmSolution, boolean parallel)
    {
        if(dataSet.getClassSize() != 2)
            throw new FailedToFitException("LS-SVM only supports binary classification problems");
        if(warmSolution != null && !(warmSolution instanceof LSSVM))
            throw new FailedToFitException("Warm solution must be an implementation of LS-SVM, not " + warmSolution.getClass());
        double[] targets = new double[dataSet.size()];
        for(int i = 0; i < dataSet.size(); i++)
            targets[i] = dataSet.getDataPointCategory(i)*2-1;
        mainLoop(dataSet, (LSSVM) warmSolution , targets, parallel);
    }

    @Override
    public void train(ClassificationDataSet dataSet, Classifier warmSolution)
    {
        train(dataSet, warmSolution, false);
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
    public void train(RegressionDataSet dataSet, boolean parallel)
    {
        train(dataSet, null, parallel);
    }

    @Override
    public LSSVM clone()
    {
        return new LSSVM(this);
    }

    private void mainLoop(DataSet dataSet, LSSVM warmSolution, double[] targets, boolean parallel)
    {
        try
        {
            ExecutorService ex = ParallelUtils.getNewExecutor(parallel);
            vecs = dataSet.getDataVectors();
            initializeVariables(targets, warmSolution, dataSet);
            
            boolean change = true;
            double dualityGap = computeDualityGap(true, parallel);
            int iter = 0;
            while (dualityGap > tol * dualObjective && change)
            {
                change = takeStep(i_up, i_low, ex, parallel);
                dualityGap = computeDualityGap(true, parallel);
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
