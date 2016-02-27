package jsat.classifiers.linear;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.SingleWeightVectorModel;
import jsat.classifiers.*;
import jsat.classifiers.svm.PlattSMO;
import jsat.distributions.Distribution;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.lossfunctions.LogisticLoss;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * This provides an implementation of regularized logistic regression using Dual
 * Coordinate Descent. This algorithm works well on both dense and sparse large
 * data sets. 
 * <br><br>
 * The regularized problem is of the form:<br>
 * C <big>&Sigma;</big> log(1+exp(-y<sub>i</sub>w<sup>T</sup>x<sub>i</sub>)) + w<sup>T</sup>w/2
 * <br><br>
 * See:<br>
 * Yu, H.-F., Huang, F.-L.,&amp;Lin, C.-J. (2010). <i>Dual Coordinate Descent 
 * Methods for Logistic Regression and Maximum Entropy Models</i>. Machine 
 * Learning, 85(1-2), 41–75. doi:10.1007/s10994-010-5221-8
 * 
 * @author Edward Raff
 */
public class LogisticRegressionDCD implements Classifier, Parameterized, SingleWeightVectorModel
{

    private static final long serialVersionUID = -5813704270903243462L;
    private static final double eps_1 = 1e-3;
    private static final double eps_2 = 1e-8;
    
    private Vec w;
    private double bias;
    private boolean useBias;
    
    private double C;
    private int maxIterations;
    
    /**
     * Creates a new Logistic Regression learner that does no more than 100
     * training iterations with a default regularization tradeoff of C = 1
     */
    public LogisticRegressionDCD()
    {
        this(1.0);
    }
    
    /**
     * Creates a new Logistic Regression learner that does no more than 100
     * training iterations. 
     * @param C the regularization tradeoff term
     */
    public LogisticRegressionDCD(double C)
    {
        this(C, 100);
    }

    /**
     * Creates a new Logistic Regression learner
     * @param C the regularization tradeoff term
     * @param maxIterations the maximum number of iterations through the data set
     */
    public LogisticRegressionDCD(double C, int maxIterations)
    {
        setC(C);
        setMaxIterations(maxIterations);
    }

    /**
     * Copy constructor 
     * @param toCopy the object to copy
     */
    protected LogisticRegressionDCD(LogisticRegressionDCD toCopy)
    {
        this(toCopy.C, toCopy.maxIterations);
        if(toCopy.w != null)
            this.w = toCopy.w.clone();
        this.bias = toCopy.bias;
        this.useBias = toCopy.useBias;
    }

    /**
     * Sets the regularization trade-off term. larger values reduce the amount 
     * of regularization, and smaller values increase the regularization. 
     * 
     * @param C the positive regularization tradeoff value 
     */
    public void setC(double C)
    {
        if(C <= 0 || Double.isInfinite(C) || Double.isNaN(C))
            throw new IllegalArgumentException("C must be a positive constant, not " + C);
        this.C = C;
    }

    /**
     * Returns the regularization tradeoff parameter
     * @return the regularization tradeoff parameter
     */
    public double getC()
    {
        return C;
    }
    
    /**
     * Sets the maximum number of iterations the algorithm is allowed to run 
     * for. 
     * @param maxIterations the maximum number of iterations
     */
    public void setMaxIterations(int maxIterations)
    {
        if(maxIterations < 1)
            throw new IllegalArgumentException("iterations must be a positive value, not " + maxIterations);
        this.maxIterations = maxIterations;
    }

    /**
     * Returns the maximum number of iterations the algorithm is allowed to run
     * @return the maximum number of iterations the algorithm is allowed to run
     */
    public int getMaxIterations()
    {
        return maxIterations;
    }

    /**
     * Sets whether or not an implicit bias term should be added to the model. 
     * @param useBias {@code true} to add a bias term, {@code false} to exclude
     * the bias term. 
     */
    public void setUseBias(boolean useBias)
    {
        this.useBias = useBias;
    }

    /**
     * Returns {@code true} if a bias term is in use, {@code false} otherwise. 
     * @return {@code true} if a bias term is in use, {@code false} otherwise. 
     */
    public boolean isUseBias()
    {
        return useBias;
    }

    @Override
    public Vec getRawWeight()
    {
        return w;
    }

    @Override
    public double getBias()
    {
        return bias;
    }
    
    @Override
    public Vec getRawWeight(int index)
    {
        if(index < 1)
            return getRawWeight();
        else
            throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }

    @Override
    public double getBias(int index)
    {
        if (index < 1)
            return getBias();
        else
            throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }
    
    @Override
    public int numWeightsVecs()
    {
        return 1;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        return LogisticLoss.classify(w.dot(data.getNumericalValues())+bias);
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
            throw new FailedToFitException("Logistic Regression is a binary classifier, can can not handle " + dataSet.getClassSize() + " class problems");
        final int N = dataSet.getSampleSize();
        List<Vec> x = dataSet.getDataVectors();
        double[] alpha = new double[N];
        double[] alphaPrime = new double[N];
        double[] Q_ii = new double[N];
        int[] y = new int[N];
        
        /*
         * All points start will small eps, because LR dosn't tend to zero out
         * coefficients. But we expect a few alphas to quickly go to larger
         * values.  
         */
        Arrays.fill(alpha, Math.min(eps_1*C, eps_2));
        Arrays.fill(alphaPrime, C-alpha[0]);
        w = new DenseVector(dataSet.getNumNumericalVars());
        bias = 0;
        
        for(int i = 0; i < N; i++)
        {
            y[i] = dataSet.getDataPointCategory(i)*2-1;
            Vec x_i = x.get(i);
            Q_ii[i] = x_i.dot(x_i);
            w.mutableAdd(alpha[0]*y[i], x_i);//all alpha are the same right now
            if(useBias)
                bias += alpha[0]*y[i];
        }
        
        IntList permutation = new IntList(N);
        ListUtils.addRange(permutation, 0, N, 1);
        
        for(int iter = 0; iter < maxIterations; iter++)
        {
            Collections.shuffle(permutation);
            
            double maxChange = 0;
            
            for(int i : permutation)
            {
                Vec x_i = x.get(i);
                //Step 1.
                final double c1 = alpha[i], c2 = alphaPrime[i];
                double a = Q_ii[i], b = y[i] * (w.dot(x_i) + bias);
                double z_m = (c2 - c1) / 2, s = c1 + c2;
                boolean case1 = z_m >= -b / a;

                double z;//see eq (35)
                if (case1)
                {
                    if (c1 >= s / 2)
                        z = 0.1 * c1;
                    else
                        z = c1;
                }
                else
                {
                    if (c2 >= s / 2)
                        z = 0.1 * c2;
                    else
                        z = c2;
                }

                //what if z is very small? Leave it alone..
                if(z < 1e-20)
                    continue;

                //Step 2.
                //Algorithm 4 solving equation (18)
                //would it really take more than 100 iterations?
                for(int subIter = 0; subIter < 100; subIter++)
                {
                    double gP = Math.log(z/(C-z));
                    if(case1)
                        gP += a*(z-c1)+ b;
                    else
                        gP += a*(z-c2)-b;
                    //check if "0"
                    if(Math.abs(gP) < 1e-6)
                        break;
                    
                    double gPP= a + s/(z*(s-z));
                    double d = -gP/gPP;
                    
                    if(z + d <= 0)
                        z *= 0.1;//unsepcified shrinkage term: just use 0.1
                    else
                        z += d;   
                }
                
                //Step 4. alpha_i  = Z1, alpha'_i = Z2.
                if(case1)
                {
                    alpha[i] = z;
                    alphaPrime[i] = C-z;
                }
                else
                {
                    alpha[i] = C-z;
                    alphaPrime[i] = z;
                }
                
                //Step 3. w = w + (Z1 −alpha_i) yi xi
                double change = (alpha[i]-c1);
                w.mutableAdd(change*y[i], x_i);
                if(useBias)
                    bias += change*y[i];
                maxChange = Math.max(maxChange, change);
            }
            
            //Convergence check
            if(Math.abs(maxChange) < 1e-4)
                return;
        }
        
    }
    
    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public Classifier clone()
    {
        return new LogisticRegressionDCD(this);
    }
    
    /**
     * Returns the weight vector used to compute results via a dot product. <br>
     * Do not modify this value, or you will alter the results returned.
     * @return the learned weight vector for prediction
     */
    public Vec getWeightVec()
    {
        return w;
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
     * Guess the distribution to use for the regularization term
     * {@link #setC(double) C} in Logistic Regression.
     *
     * @param d the data set to get the guess for
     * @return the guess for the C parameter 
     */
    public static Distribution guessC(DataSet d)
    {
        return PlattSMO.guessC(d);
    }
}
