
package jsat.classifiers.svm;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * Implements Algorithms 1 and 2 for Dual Coordinate Descent with shrinking (DCDs)
 * method for a linear L<sup>1</sup> or L<sup>2</sup> Support Vector Machine.
 * This implementation uses an implicit bias term.<br>
 * NOTE: While this implementation makes use of the dual formulation only the linear 
 * kernel is ever used. The algorithm also uses the primal representation and uses 
 * the explicit formulation of <i>w></i> in training and classification. As such, 
 * the support vectors found are not necessary once training is complete - 
 * and will be discarded.
 * <br>
 * See: Hsieh, C.-J., Chang, K.-W., Lin, C.-J., Keerthi, S. S., & Sundararajan, S. (2008). 
 * <i>A Dual Coordinate Descent Method for Large-scale Linear SVM</i>. 
 * Proceedings of the 25th international conference on Machine learning - ICML  ’08 (pp. 408–415). 
 * New York, New York, USA: ACM Press. doi:10.1145/1390156.1390208
 * 
 * @author Edward Raff
 * @see DCDs
 */
public class DCD implements BinaryScoreClassifier, Parameterized
{
    private int maxIterations;
    private Vec[] vecs;
    private double[] alpha;
    private double[] y;
    private double bias;
    private Vec w;
    private double C;
    private boolean useL1;
    private boolean onlineVersion = false;
    
    private final List<Parameter> params = Collections.unmodifiableList(Parameter.getParamsFromMethods(this));
    private final Map<String, Parameter> paramMap = Parameter.toParameterMap(params);

    /**
     * Creates a new DCDL2 SVM object
     */
    public DCD()
    {
        this(1000, false);
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
        
        final double U, D;
        if(useL1)
        {
            U = C;
            D = 0;
        }
        else
        {
            U = Double.POSITIVE_INFINITY;
            D = 1.0/(2*C);
        }
        
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            vecs[i] = dataSet.getDataPoint(i).getNumericalValues();
            y[i] = dataSet.getDataPointCategory(i)*2-1;
            Qhs[i] = vecs[i].dot(vecs[i])+1.0+D;//+1 for implicit bias term
        }
        w = new DenseVector(vecs[0].length());
        
        List<Integer> A = new IntList(vecs.length);
        ListUtils.addRange(A, 0, vecs.length, 1);
        
        Random rand = new Random();
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
            bias += scale;
        }
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
