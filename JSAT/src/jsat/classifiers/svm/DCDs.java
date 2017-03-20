package jsat.classifiers.svm;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.SingleWeightVectorModel;
import jsat.classifiers.*;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.*;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.random.XORWOW;
import java.util.*;
import jsat.DataSet;
import jsat.distributions.Distribution;
import jsat.distributions.Exponential;
import jsat.utils.random.RandomUtil;

/**
 * Implements Dual Coordinate Descent with shrinking (DCDs) training algorithms
 * for a Linear L<sup>1</sup> or L<sup>2</sup> Support Vector Machine for binary 
 * classification and regression.
 * NOTE: While this implementation makes use of the dual formulation only the linear 
 * kernel is ever used. The algorithm also uses the primal representation and uses 
 * the explicit formulation of <i>w</i> in training and classification. As such, 
 * the support vectors found are not necessary once training is complete - 
 * and will be discarded.<br>
 * <br>
 * DCDs man be warm started by other DCDs models trained on the same data set. 
 * <br><br>
 * See: 
 * <ul>
 * <li>
 * Hsieh, C.-J., Chang, K.-W., Lin, C.-J., Keerthi, S. S., &amp; Sundararajan, S. 
 * (2008). <i>A Dual Coordinate Descent Method for Large-scale Linear SVM</i>. 
 * Proceedings of the 25th international conference on Machine learning - ICML
 * ’08 (pp. 408–415). New York, New York, USA: ACM Press. 
 * doi:10.1145/1390156.1390208
 * </li>
 * <li>
 * Ho, C.-H., &amp; Lin, C.-J. (2012). <i>Large-scale Linear Support Vector 
 * Regression</i>. Journal of Machine Learning Research, 13, 3323–3348. 
 * Retrieved from <a href="http://ntu.csie.org/~cjlin/papers/linear-svr.pdf">
 * here</a>
 * </ul>
 * @author Edward Raff
 * @see DCD
 */
public class DCDs implements BinaryScoreClassifier, Regressor, Parameterized, SingleWeightVectorModel, WarmClassifier, WarmRegressor
{

    private static final long serialVersionUID = -1686294187234524696L;
    private int maxIterations;
    private double tolerance;
    private Vec[] vecs;
    private double[] alpha;
    private double[] y;
    private double bias;
    private Vec w;
    private double C;
    private boolean useL1;
    private double eps = 0.001;
    
    private boolean useBias = true;
    
    private final List<Parameter> params = Collections.unmodifiableList(Parameter.getParamsFromMethods(this));
    private final Map<String, Parameter> paramMap = Parameter.toParameterMap(params);

    /**
     * Creates a new DCDL2 SVM object
     */
    public DCDs()
    {
        this(10000, false);
    }

    /**
     * Creates a new DCD SVM object
     * @param maxIterations the maximum number of training iterations
     * @param useL1 whether or not to use L1 or L2 form
     */
    public DCDs(int maxIterations, boolean useL1)
    {
        this(maxIterations, 1e-3, 1, useL1);
    }
    
    /**
     * Creates a new DCD SVM object
     * @param maxIterations the maximum number of training iterations
     * @param tolerance the tolerance value for early stopping
     * @param C the misclassification penalty
     * @param useL1 whether or not to use L1 or L2 form
     */
    public DCDs(int maxIterations, double tolerance, double C, boolean useL1)
    {
        setMaxIterations(maxIterations);
        setTolerance(tolerance);
        setC(C);
        setUseL1(useL1);
    }
    
    /**
     * Sets the penalty parameter for misclassifications. The recommended value 
     * is 1, and values larger than 4 are not normally needed according to the 
     * original paper. 
     * 
     * @param C the penalty parameter in (0, Inf)
     */
    public void setC(double C)
    {
        if(Double.isNaN(C) || Double.isInfinite(C) || C <= 0)
            throw new ArithmeticException("Penalty parameter must be a positive value, not " + C);
        this.C = C;
    }

    /**
     * Returns the penalty parameter for misclassifications.
     * @return the penalty parameter for misclassifications.
     */
    public double getC()
    {
        return C;
    }

    /**
     * Sets the {@code eps} used in the epsilon insensitive loss function used 
     * when performing regression. Errors in the output that less than 
     * {@code eps} during training are treated as correct. 
     * <br>
     * This parameter has no impact on classification problems. 
     * 
     * @param eps the non-negative value to use as the error tolerance in regression
     */
    public void setEps(double eps)
    {
        if(Double.isNaN(eps) || eps < 0 || Double.isInfinite(eps))
            throw new IllegalArgumentException("eps must be non-negative, not "+eps);
        this.eps = eps;
    }

    /**
     * Returns the epsilon insensitivity parameter used in regression problems. 
     * @return the epsilon insensitivity parameter used in regression problems.
     */
    public double getEps()
    {
        return eps;
    }

    /**
     * Sets the tolerance for the stopping condition when training, a small value near 
     * zero allows training to stop early when little to no additional convergence 
     * is possible. 
     * 
     * @param tolerance the tolerance value to use to stop early
     */
    public void setTolerance(double tolerance)
    {
        this.tolerance = tolerance;
    }

    /**
     * Returns the tolerance value used to terminate early
     * @return the tolerance value used to terminate early 
     */
    public double getTolerance()
    {
        return tolerance;
    }

    /**
     * Determines whether or not to use the L<sup>1</sup> or L<sup>2</sup> SVM
     * @param useL1 <tt>true</tt> to use the L<sup>1</sup> form, <tt>false</tt> to use the L<sup>2</sup> form. 
     */
    public void setUseL1(boolean useL1)
    {
        this.useL1 = useL1;
    }

    /**
     * Returns <tt>true</tt> if the L<sup>1</sup> form is in use
     * @return <tt>true</tt> if the L<sup>1</sup> form is in use 
     */
    public boolean isUseL1()
    {
        return useL1;
    }

    /**
     * Sets the maximum number of iterations allowed through the whole training 
     * set. 
     * @param maxIterations the maximum number of training epochs
     */
    public void setMaxIterations(int maxIterations)
    {
        if(maxIterations <= 0)
            throw new IllegalArgumentException("Number of iterations must be positive, not " + maxIterations);
        this.maxIterations = maxIterations;
    }

    /**
     * Returns the maximum number of allowed training epochs
     * @return the maximum number of allowed training epochs
     */
    public int getMaxIterations()
    {
        return maxIterations;
    }

    /**
     * Sets whether or not an implicit bias term should be added to the inputs. 
     * @param useBias {@code true} to add an implicit bias term to inputs, 
     * {@code false} to use the input data as provided. 
     */
    public void setUseBias(boolean useBias)
    {
        this.useBias = useBias;
    }

    /**
     * Returns {@code true} if an implicit bias term is in use, or {@code false}
     * if not. 
     * @return {@code true} if an implicit bias term is in use, or {@code false}
     * if not. 
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
        if(w == null)
            throw new UntrainedModelException("The model has not been trained");
        CategoricalResults cr = new CategoricalResults(2);
        
        if(getScore(data) < 0)
            cr.setProb(0, 1.0);
        else
            cr.setProb(1, 1.0);
        
        return cr;
    }

    @Override
    public double getScore(DataPoint dp)
    {
        return w.dot(dp.getNumericalValues())+bias;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, (Classifier)null);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, Classifier warmSolution, ExecutorService threadPool)
    {
        trainC(dataSet, warmSolution);
    }
            
    @Override
    public void trainC(ClassificationDataSet dataSet, Classifier warmSolution)
    {
        if(dataSet.getClassSize() != 2)
            throw new FailedToFitException("SVM only supports binary classificaiton problems");
        vecs = new Vec[dataSet.getSampleSize()];
        alpha = new double[vecs.length];
        y = new double[vecs.length];
        bias = 0;
        final double[] Qhs = new double[vecs.length];//Q hats
        
        final double[] U = new double[vecs.length], D = new double[vecs.length];
        
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            final DataPoint dp = dataSet.getDataPoint(i);
            vecs[i] = dp.getNumericalValues();
            y[i] = dataSet.getDataPointCategory(i)*2-1;
            U[i] = getU(dp.getWeight());
            D[i] = getD(dp.getWeight());
            Qhs[i] = vecs[i].dot(vecs[i])+D[i];
            if(useBias)//+1 for implicit bias term
                Qhs[i]++;
        }
        w = new DenseVector(vecs[0].length());
        
        List<Integer> A = new IntList(vecs.length);
        ListUtils.addRange(A, 0, vecs.length, 1);
        
        if(warmSolution != null)
        {
            //TODO the below code works OK for warm starting classification problems, but we also need code that works well for warm starting the regression problems to meet the API contract. Having more difficulty with that one. 
//            if (warmSolution instanceof SimpleWeightVectorModel)
//            {
//                SimpleWeightVectorModel swvm = (SimpleWeightVectorModel) warmSolution;
//                if (swvm.numWeightsVecs() != 1)
//                    throw new FailedToFitException("Can not warm start from given solution, it has more than 1 weight vector");
//
//                Vec w_warm = swvm.getRawWeight(0);
//                double b_warm = useBias ? swvm.getBias(0) : 0;
//                //we can't just copy the values in b/c we need the solution to always be a linear combination of the training data
//                //we we use it to guess at alpha values
//                Iterator<Integer> iter = A.iterator();
//                while (iter.hasNext())
//                {
//                    int i = iter.next();
//                    double error = max(1 - y[i] * (vecs[i].dot(w_warm) + b_warm), 0);
//                    if (!useL1)
//                        error *= error;
//                    error = min(C*error, U[i]) * y[i];
//                    alpha[i] = abs(error);
//                    if(error != 0)
//                    {
//                        w.mutableAdd(error, vecs[i]);
//                        bias += error;
//                    }
//                }
//            }
            if(warmSolution instanceof DCDs)
            {
                DCDs other = (DCDs) warmSolution;
                if (this.alpha != null && other.alpha.length != this.alpha.length)
                    throw new FailedToFitException("Warm solution could not have been trained on the same data set");

                double C_mul = this.C/other.C;
                other.w.copyTo(this.w);
                this.w.mutableMultiply(C);
                this.bias = other.bias*C_mul;
                System.arraycopy(other.alpha, 0, this.alpha, 0, this.alpha.length);
                for(int i = 0; i < this.alpha.length; i++)
                    this.alpha[i] *= C_mul;
            }
            else 
                throw new FailedToFitException("Warm solution can not be used for warm start");
        }
        
        double M = Double.NEGATIVE_INFINITY;
        double m = Double.POSITIVE_INFINITY;
        boolean noShrinking = false;
        
        /*
         * From profling Shufling & RNG generation takes a suprising amount of 
         * time on some data sets, so use one of our fast ones
         */
        Random rand = RandomUtil.getRandom();

        for(int t = 0; t < maxIterations; t++ )
        {
            Collections.shuffle(A, rand);
            M = Double.NEGATIVE_INFINITY;
            m = Double.POSITIVE_INFINITY;
            Iterator<Integer> iter = A.iterator();
            while(iter.hasNext())//2. 
            {
                int i = iter.next();
                //a
                final double G = y[i]*(w.dot(vecs[i])+bias)-1+D[i]*alpha[i];//bias will be zero if usebias is off
                //b
                double PG = 0;
                if(alpha[i] == 0)
                {
                    if(G > M && !noShrinking)
                        iter.remove();
                    if(G < 0)
                        PG = G;
                }
                else if(alpha[i] == U[i])
                {
                    if(G < m && !noShrinking)
                        iter.remove();
                    if(G > 0)
                        PG = G;
                }
                else
                    PG = G;
                //c
                M = Math.max(M, PG);
                m = Math.min(m, PG);
                //d
                if(PG != 0)
                {
                    double alphaOld = alpha[i];
                    alpha[i] = Math.min(Math.max(alpha[i]-G/Qhs[i], 0), U[i]);
                    double scale = (alpha[i]-alphaOld)*y[i];
                    w.mutableAdd(scale, vecs[i]);
                    if(useBias)
                        bias += scale;
                }
            }
            
            if(M - m < tolerance)//3.
            {
                //a
                if(A.size() == alpha.length)
                    break;//We have converged
                else //repeat without shrinking
                {
                    A.clear();
                    ListUtils.addRange(A, 0, vecs.length, 1);
                    noShrinking = true;
                }
            }
            else if(M <= 0 || m >= 0)//technically less agressive then the original paper
                noShrinking = true;
            else
                noShrinking = false;
        }
        
        //dual problem variables are no longer needed
        vecs = null;
        y = null;
        //don't delete alpha incase we want to warm start from it
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }
    
    @Override
    public boolean warmFromSameDataOnly()
    {
        return true;
    }

    @Override
    public DCDs clone()
    {
        DCDs clone = new DCDs(maxIterations, tolerance, C, useL1);
        clone.bias = this.bias;
        clone.useBias = this.useBias;
        
        if(this.w != null)
            clone.w = this.w.clone();
        if(this.alpha != null)
            clone.alpha = Arrays.copyOf(this.alpha, this.alpha.length);
        
        return clone;
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

    @Override
    public double regress(DataPoint data)
    {
        return w.dot(data.getNumericalValues())+bias;
    }

    @Override
    public void train(RegressionDataSet dataSet, Regressor warmSolution, ExecutorService threadPool)
    {
        train(dataSet, warmSolution);
    }
    
    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        train(dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, (Regressor) null);
    }
    
    @Override
    public void train(RegressionDataSet dataSet, Regressor warmSolution)
    {
        vecs = new Vec[dataSet.getSampleSize()];
        /**
         * Makes the Beta vector in the Algo 4 description 
         */
        alpha = new double[vecs.length];
        y = new double[vecs.length];
        bias = 0;
        final double[] Qhs = new double[vecs.length];//Q hats
        
        final double[] U = new double[vecs.length], lambda = new double[vecs.length];
        double v_0 = 0;
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            final DataPoint dp = dataSet.getDataPoint(i);
            vecs[i] = dp.getNumericalValues();
            y[i] = dataSet.getTargetValue(i);
            U[i] = getU(dp.getWeight());
            lambda[i] = getD(dp.getWeight());
            Qhs[i] = vecs[i].dot(vecs[i])+lambda[i];
            if (useBias)
                Qhs[i] += 1.0;
            v_0 += Math.abs(eq24(0, -y[i]-eps, -y[i]+eps, U[i]));
        }
        w = new DenseVector(vecs[0].length());
        
        IntList activeSet = new IntList(2*vecs.length);
        ListUtils.addRange(activeSet, 0, vecs.length, 1);
        
        if(warmSolution != null)
        {
            if(warmSolution instanceof DCDs)
            {
                DCDs other = (DCDs) warmSolution;
                if (this.alpha != null && other.alpha.length != this.alpha.length)
                    throw new FailedToFitException("Warm solution could not have been trained on the same data set");

                double C_mul = this.C/other.C;
                other.w.copyTo(this.w);
                this.w.mutableMultiply(C);
                this.bias = other.bias*C_mul;
                System.arraycopy(other.alpha, 0, this.alpha, 0, this.alpha.length);
                for(int i = 0; i < this.alpha.length; i++)
                    this.alpha[i] *= C_mul;
            }
            else 
                throw new FailedToFitException("Warm solution can not be used for warm start");
        }
        
        /*
         * From profling Shufling & RNG generation takes a suprising amount of 
         * time on some data sets, so use one of our fast ones
         */
        Random rand = RandomUtil.getRandom();
        
        double M = Double.POSITIVE_INFINITY;

        for(int iteration = 0; iteration < maxIterations; iteration++)
        {
            double maxVk = Double.NEGATIVE_INFINITY;
            double vKSum = 0;
            //6.1 Randomly permute T
            Collections.shuffle(activeSet, rand);

            //6.2 For i in T
            Iterator<Integer> iter = activeSet.iterator();
            while(iter.hasNext())
            {
                final int i = iter.next();
                final double y_i = y[i];
                final Vec x_i = vecs[i];
                final double wDotX = w.dot(x_i)+bias;
                final double g = -y_i + wDotX + lambda[i] * alpha[i];
                final double gP = g + eps;
                final double gN = g - eps;
                
                final double v_i = eq24(alpha[i], gN, gP, U[i]);
                maxVk = Math.max(maxVk, v_i);
                vKSum += Math.abs(v_i);
                
                //6.2.3 shrinking work
                //eq (26) beta_i = 0 and g'n(βi) < −M < 0 <M < g'p(βi)
                boolean shrink = false;
                if(alpha[i] == 0 && gN < -M && -M < 0 && M < gP)
                    shrink = true;
                if( (alpha[i] == U[i] &&  gP < -M) || (alpha[i] == -U[i] && gN > M))
                    shrink = true;
                
                if(shrink)
                    iter.remove();
                
                //eq (22)
                final double Q_ii = Qhs[i];
                final double d;
                if (gP < Q_ii * alpha[i])
                    d = -gP / Q_ii;
                else if (gN > Q_ii * alpha[i])
                    d = -gN / Q_ii;
                else
                    d = -alpha[i];

                if (Math.abs(d) < 1e-14)
                    continue;
                
                //s = max(−U, min(U,beta_i +d))     eq (21) 
                final double s = Math.max(-U[i], Math.min(U[i], alpha[i]+d));
                
                w.mutableAdd(s-alpha[i], x_i);
                if(useBias)
                    bias += (s-alpha[i]);
                alpha[i] = s;
            }
            
            //convergence check
            if(vKSum/v_0 < tolerance)//converged
            {
                if(activeSet.size() == vecs.length)//we converged on all the data
                    break;
                else//reset to do a pass through the whole data set
                {
                    activeSet.clear();
                    ListUtils.addRange(activeSet, 0, vecs.length, 1);
                    M = Double.POSITIVE_INFINITY;
                }
            }
            else
            {
                M = maxVk;
            }
            
        }
        
        y = null;
        vecs = null;
    }
    
    private double getU(double w)
    {
        if(useL1)
            return C*w;
        else
            return Double.POSITIVE_INFINITY;
    }
    
    private double getD(double w)
    {
        if(useL1)
            return 0;
        else
            return 1/(2*C*w);
    }

    /**
     * returns the result of evaluation equation 24 of an individual index
     * @param beta_i the weight coefficent value
     * @param gN the g'<sub>n</sub>(beta_i) value
     * @param gP the g'<sub>p</sub>(beta_i) value
     * @param U the upper bound value obtained from {@link #getU(double) }
     * @return the result of equation 24
     */
    protected static double eq24(final double beta_i, final double gN, final double gP, final double U)
    {
        //6.2.2
        double vi = 0;//Used as "other" value
        
        if(beta_i == 0)//if beta_i = 0 ...
        {
            //if beta_i = 0 and g'n(beta_i) >= 0
            if(gN >= 0)
                vi = gN;
            else if(gP <= 0) //if beta_i = 0 and g'p(beta_i) <= 0
                vi = -gP;
        }
        else//beta_i is non zero
        {
            //Two cases
            //if beta_i in (−U, 0), or 
            //beta_i = −U and g'n(beta_i) <= 0
            //then v_i =  |g'n|
            
            //if beta_i in (0,U), or 
            //beta_i = U and g'p(βi) >= 0
            //then v_i = |g'p|
            
            if(beta_i < 0)//first set of cases
            {
                if(beta_i > -U || (beta_i == -U && gN <= 0))
                    vi = Math.abs(gN);
            }
            else//second case
            {
                if(beta_i < U || (beta_i == U && gP >= 0))
                    vi = Math.abs(gP);
            }
        }
        
        return vi;
    }
    
    /**
     * Guess the distribution to use for the regularization term
     * {@link #setC(double) C} in a SVM.
     *
     * @param d the data set to get the guess for
     * @return the guess for the C parameter in the SVM 
     */
    public static Distribution guessC(DataSet d)
    {
        return PlattSMO.guessC(d);
    }
}
