
package jsat.classifiers.svm;

import java.util.*;
import java.util.concurrent.ExecutorService;

import jsat.SingleWeightVectorModel;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import static jsat.classifiers.svm.DCDs.eq24;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.random.RandomUtil;

/**
 * Implements Dual Coordinate Descent (DCD) training algorithms for a Linear 
 * L<sup>1</sup> or L<sup>2</sup> Support Vector Machine for binary 
 * classification and regression.
 * NOTE: While this implementation makes use of the dual formulation only the linear 
 * kernel is ever used. The algorithm also uses the primal representation and uses 
 * the explicit formulation of <i>w</i> in training and classification. As such, 
 * the support vectors found are not necessary once training is complete - 
 * and will be discarded.
 * <br><br>
 * See: 
 * <ul>
 * <li>
 * Hsieh, C.-J., Chang, K.-W., Lin, C.-J., Keerthi, S. S.,&amp;Sundararajan, S. 
 * (2008). <i>A Dual Coordinate Descent Method for Large-scale Linear SVM</i>. 
 * Proceedings of the 25th international conference on Machine learning - ICML
 * ’08 (pp. 408–415). New York, New York, USA: ACM Press. 
 * doi:10.1145/1390156.1390208
 * </li>
 * <li>
 * Ho, C.-H.,&amp;Lin, C.-J. (2012). <i>Large-scale Linear Support Vector 
 * Regression</i>. Journal of Machine Learning Research, 13, 3323–3348. 
 * Retrieved from <a href="http://ntu.csie.org/~cjlin/papers/linear-svr.pdf">
 * here</a>
 * </ul>
 * @author Edward Raff
 * @see DCDs
 */
public class DCD implements BinaryScoreClassifier, Regressor, Parameterized, SingleWeightVectorModel
{

	private static final long serialVersionUID = -1489225034030922798L;
	private int maxIterations;
    private Vec[] vecs;
    private double[] alpha;
    private double[] y;
    private double bias;
    private Vec w;
    private double C;
    private boolean useL1;
    private boolean onlineVersion = false;
    private double eps = 0.001;
    private boolean useBias = true;
    
    private final List<Parameter> params = Collections.unmodifiableList(Parameter.getParamsFromMethods(this));
    private final Map<String, Parameter> paramMap = Parameter.toParameterMap(params);

    /**
     * Creates a new DCDL2 SVM object
     */
    public DCD()
    {
        this(10000, false);
    }

    /**
     * Creates a new DCD SVM object. The default C value of 1 is 
     * used as suggested in the original paper. 
     * @param maxIterations the maximum number of training iterations
     * @param useL1 whether or not to use L1 or L2 form
     */
    public DCD(int maxIterations, boolean useL1)
    {
        this(maxIterations, 1, useL1);
    }
    
    /**
     * Creates a new DCD SVM object
     * @param maxIterations the maximum number of training iterations
     * @param C the misclassification penalty
     * @param useL1 whether or not to use L1 or L2 form
     */
    public DCD(int maxIterations, double C, boolean useL1)
    {
        this.maxIterations = maxIterations;
        this.C = C;
        this.useL1 = useL1;
    }

    /**
     * By default, Algorithm 1 is used. Algorithm 2 is an "online" version 
     * that updates the dual form by only one data point at a time. This
     * controls which version is used. 
     * @param onlineVersion <tt>false</tt> to use algorithm 1, <tt>true</tt>
     * to use algorithm 2
     */
    public void setOnlineVersion(boolean onlineVersion)
    {
        this.onlineVersion = onlineVersion;
    }

    /**
     * Returns whether or not the online version of the algorithm, 
     * algorithm 2 is in use. 
     * @return <tt>true</tt> if algorithm 2 is in use, <tt>false</tt> if
     * algorithm 1
     */
    public boolean isOnlineVersion()
    {
        return onlineVersion;
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
        if (w == null)
            throw new UntrainedModelException("The model has not been trained");
        CategoricalResults cr = new CategoricalResults(2);

        if (getScore(data) < 0)
            cr.setProb(0, 1.0);
        else
            cr.setProb(1, 1.0);

        return cr;
    }

    @Override
    public double getScore(DataPoint dp)
    {
        return w.dot(dp.getNumericalValues()) + bias;
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
            throw new FailedToFitException("SVM only supports binary classificaiton problems");
        vecs = new Vec[dataSet.getSampleSize()];
        alpha = new double[vecs.length];
        y = new double[vecs.length];
        bias = 0;
        final double[] Qhs = new double[vecs.length];//Q hats
        
        final double U = getU(), D = getD();
        
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            vecs[i] = dataSet.getDataPoint(i).getNumericalValues();
            y[i] = dataSet.getDataPointCategory(i)*2-1;
            Qhs[i] = vecs[i].dot(vecs[i])+D;
            if(useBias)
                Qhs[i] += 1.0;
        }
        w = new DenseVector(vecs[0].length());
        
        List<Integer> A = new IntList(vecs.length);
        ListUtils.addRange(A, 0, vecs.length, 1);
        
        Random rand = RandomUtil.getRandom();
        for(int t = 0; t < maxIterations; t++ )
        {
            if(onlineVersion)
            {
                int i = rand.nextInt(vecs.length);
                performUpdate(i, D, U, Qhs[i]);
            }
            else
            {
                Collections.shuffle(A, rand);
                for(int i : A)
                    performUpdate(i, D, U, Qhs[i]);
            }
        }
        
    }
    
    /**
     * Performs steps a, b, and c of the DCD algorithms 1 and 2
     * @param i the index to update
     * @param D the value of D
     * @param U the value of U
     * @param Qh_ii the Q hat value that will be used in this update. 
     */
    private void performUpdate(final int i, final double D, final double U, final double Qh_ii)
    {
        //a
        final double G = y[i]*(w.dot(vecs[i])+bias)-1+D*alpha[i];
        //b
        final double PG;
        if(alpha[i] == 0)
            PG = Math.min(G, 0);
        else if(alpha[i] == U)
            PG = Math.max(G, 0);
        else
            PG = G;
        //c
        if(PG != 0)
        {
            final double alphaOld = alpha[i];
            alpha[i] = Math.min(Math.max(alpha[i]-G/Qh_ii, 0), U);
            final double scale = (alpha[i]-alphaOld)*y[i];
            w.mutableAdd(scale, vecs[i]);
            if(useBias)
                bias += scale;
        }
    }
    
    @Override
    public double regress(DataPoint data)
    {
        return w.dot(data.getNumericalValues())+bias;
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        train(dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        vecs = new Vec[dataSet.getSampleSize()];
        /**
         * Makes the Beta vector in the Algo 4 description 
         */
        alpha = new double[vecs.length];
        y = new double[vecs.length];
        bias = 0;
        final double[] Qhs = new double[vecs.length];//Q hats
        
        final double U = getU(), lambda = getD();
        double v_0 = 0;
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            vecs[i] = dataSet.getDataPoint(i).getNumericalValues();
            y[i] = dataSet.getTargetValue(i);
            Qhs[i] = vecs[i].dot(vecs[i])+lambda;
            if (useBias)
                Qhs[i] += 1.0;
            v_0 += Math.abs(eq24(0, -y[i]-eps, -y[i]+eps, U));
        }
        w = new DenseVector(vecs[0].length());
        
        IntList activeSet = new IntList(vecs.length);
        ListUtils.addRange(activeSet, 0, vecs.length, 1);
        
        @SuppressWarnings("unused")
		double M = Double.POSITIVE_INFINITY;

        for(int iteration = 0; iteration < maxIterations; iteration++)
        {
            double maxVk = Double.NEGATIVE_INFINITY;
            double vKSum = 0;
            //6.1 Randomly permute T
            Collections.shuffle(activeSet);

            //6.2 For i in T
            Iterator<Integer> iter = activeSet.iterator();
            while(iter.hasNext())
            {
                final int i = iter.next();
                final double y_i = y[i];
                final Vec x_i = vecs[i];
                final double wDotX = w.dot(x_i)+bias;
                final double g = -y_i + wDotX + lambda * alpha[i];
                final double gP = g + eps;
                final double gN = g - eps;
                
                final double v_i = eq24(alpha[i], gN, gP, U);
                maxVk = Math.max(maxVk, v_i);
                vKSum += Math.abs(v_i);

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
                final double s = Math.max(-U, Math.min(U, alpha[i]+d));
                
                w.mutableAdd(s-alpha[i], x_i);
                if(useBias)
                    bias += (s-alpha[i]);
                alpha[i] = s;
            }
            
            //convergence check
            if(vKSum/v_0 < 1e-4)//converged
                break;
            else
                M = maxVk;
        }
    }
    
    private double getU()
    {
        if(useL1)
            return C;
        else
            return Double.POSITIVE_INFINITY;
    }
    
    private double getD()
    {
        if(useL1)
            return 0;
        else
            return 1/(2*C);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public DCD clone()
    {
        DCD clone = new DCD(maxIterations, C, useL1);
        clone.onlineVersion = this.onlineVersion;
        clone.bias = this.bias;
        clone.useBias = this.useBias;
        
        if(this.w != null)
            clone.w = this.w.clone();
        
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
}
