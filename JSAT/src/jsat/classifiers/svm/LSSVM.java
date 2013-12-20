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
import java.util.Map;
import jsat.DataSet;
import jsat.exceptions.FailedToFitException;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;

/**
 * The Least Squares Support Vector Machine (LS-SVM) is an alternative to the 
 * standard SVM classifier for regression and binary classification problems. It
 * can be faster to train, but is usually significantly slower to perform 
 * predictions with. This is because the LS-SVM solution is dense, so all 
 * training points become support vectors. <br>
 * <br>
 * NOTE: A SMO implementation similar to {@link PlatSMO} is used. This is done 
 * because is can easily operate without explicitly forming the whole kernel 
 * matrix. However it is recommended to use the LS-SVM when the problem size is 
 * small enough such that {@link SupportVectorLearner.CacheMode#FULL} can be 
 * used. <br>
 * <br>
 * See: <br>
 * <ul>
 * <li>Suykens, J., & Vandewalle, J. (1999). <i>Least Squares Support Vector 
 * Machine Classifiers</i>. Neural processing letters, 9(3), 293–298. 
 * doi:10.1023/A:1018628609742</li>
 * <li>Keerthi, S. S., & Shevade, S. K. (2003). <i>SMO algorithm for Least 
 * Squares SVM</i>. In Proceedings of the International Joint Conference on 
 * Neural Networks (Vol. 3, pp. 2088–2093). IEEE. doi:10.1109/IJCNN.2003.1223730
 * </li>
 * </ul>
 * 
 * 
 * @author Edward Raff
 */
public class LSSVM extends SupportVectorLearner implements BinaryScoreClassifier, Regressor, Parameterized
{
    protected double b = 0, b_low, b_up;
    private double C = 1;
    
    private int i_up, i_low;
    private double[] fcache;
    private double dualObjective;
    
    private List<Parameter> params = Parameter.getParamsFromMethods(this);
    private Map<String, Parameter> paramMap = Parameter.toParameterMap(params);
    
    private static double epsilon = 1e-12;
    private static double tol = 1e-6;
    
    public LSSVM(KernelTrick kernel, CacheMode cacheMode)
    {
        super(kernel, cacheMode);
    }

    public LSSVM(LSSVM toCopy)
    {
        super(toCopy.getKernel().clone(), toCopy.getCacheMode());
        this.b_low = toCopy.b_low;
        this.b_up = toCopy.b_up;
        this.i_up = toCopy.i_up;
        this.i_low = toCopy.i_low;
        this.C = toCopy.C;
        if(toCopy.alphas != null)
        {
            this.alphas = Arrays.copyOf(toCopy.alphas, toCopy.alphas.length);
        }
    }
    

    public void setC(double C)
    {
        this.C = C;
    }

    public double getC()
    {
        return C;
    }

    
    private boolean takeStep(int i1, int i2)
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
        for (int i = 0; i < fcache.length; i++)
        {
            final double k_i1 = kEval(i1, i);
            final double k_i2 = kEval(i2, i);
            final double Fi = (fcache[i] += (a1 - alph1) * k_i1 + (a2 - alph2) * k_i2);
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
        
        return true;
    }
    
    private double computeDualityGap(boolean fast)
    {
        double gap = 0;
        //set b using (3.16) or (3.17)
        if(fast)
            b = (b_up+b_low)/2;
        else
        {
            b = 0;
            for(int i = 0; i < alphas.length; i++)
                b += fcache[i]-alphas[i]/C;
            b /= alphas.length;
        }
        
        for(int i = 0; i < alphas.length; i++)
        {
            final double x_i = b + alphas[i]/C - fcache[i];
            gap += alphas[i]*(fcache[i]-(0.5*alphas[i]/C)) + C*x_i*x_i/2;
        }
        return gap;
    }
    
    private void initializeVariables(double[] targets)
    {
        alphas = new double[targets.length];
        fcache = new double[targets.length];
        dualObjective = 0;
        for(int i = 0; i < targets.length; i++)
            fcache[i] = -targets[i];
        
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
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        if(dataSet.getClassSize() != 2)
            throw new FailedToFitException("LS-SVM only supports binary classification problems");
        double[] targets = new double[dataSet.getSampleSize()];
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            targets[i] = dataSet.getDataPointCategory(i)*2-1;
        mainLoop(dataSet, targets);
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
        train(dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        double[] targets = dataSet.getTargetValues().arrayCopy();
        mainLoop(dataSet, targets);        
    }

    @Override
    public LSSVM clone()
    {
        return new LSSVM(this);
    }

    @Override
    public List<Parameter> getParameters()
    {
        return params;
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return paramMap.get(paramName);
    }

    private void mainLoop(DataSet dataSet, double[] targets)
    {
        vecs = new ArrayList<Vec>(dataSet.getSampleSize());
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            vecs.add(dataSet.getDataPoint(i).getNumericalValues());
        initializeVariables(targets);
        
        boolean change = true;
        double dualityGap = computeDualityGap(true);
        int iter = 0;
        while(dualityGap > tol*dualObjective && change)
        {
            change = takeStep(i_up, i_low);
            dualityGap = computeDualityGap(true);
            iter++;
        }
        
        setCacheMode(null);
        setAlphas(alphas);
    }
    
}
