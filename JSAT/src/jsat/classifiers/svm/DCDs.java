package jsat.classifiers.svm;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * Implements Algorithm 3 for Dual Coordinate Descent with shrinking (DCDs)
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
 * @author Edward Raff
 */
public class DCDs implements Classifier, Parameterized
{
    private int maxIterations;
    private double tolerance;
    private Vec[] vecs;
    private double[] alpha;
    private double[] y;
    private double bias;
    private Vec w;
    private double C;
    private boolean useL1;
    
    private final List<Parameter> params = Collections.unmodifiableList(Parameter.getParamsFromMethods(this));
    private final Map<String, Parameter> paramMap = Parameter.toParameterMap(params);

    /**
     * Creates a new DCDL2 SVM object
     */
    public DCDs()
    {
        this(1000, false);
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
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.C = C;
        this.useL1 = useL1;
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
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(w == null)
            throw new UntrainedModelException("The model has not been trained");
        CategoricalResults cr = new CategoricalResults(2);
        
        if(w.dot(data.getNumericalValues())+bias < 0)
            cr.setProb(0, 1.0);
        else
            cr.setProb(1, 1.0);
        
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
            throw new FailedToFitException("SVM only supports binary classificaiton problems");
        vecs = new Vec[dataSet.getSampleSize()];
        alpha = new double[vecs.length];
        y = new double[vecs.length];
        
        
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            vecs[i] = dataSet.getDataPoint(i).getNumericalValues();
            y[i] = dataSet.getDataPointCategory(i)*2-1;
        }
        
        w = new DenseVector(vecs[0].length());
        
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
        
        List<Integer> A = new IntList(vecs.length);
        ListUtils.addRange(A, 0, vecs.length, 1);
        
        double M = Double.NEGATIVE_INFINITY;
        double m = Double.POSITIVE_INFINITY;
        boolean noShrinking = false;
        
        for(int t = 0; t < maxIterations; t++ )
        {
            Collections.shuffle(A);
            M = Double.NEGATIVE_INFINITY;
            m = Double.POSITIVE_INFINITY;
            Iterator<Integer> iter = A.iterator();
            while(iter.hasNext())//2. 
            {
                int i = iter.next();
                //a
                final double G = y[i]*(w.dot(vecs[i])+bias)-1+D*alpha[i];
                //b
                double PG = 0;
                if(alpha[i] == 0)
                {
                    if(G > M && !noShrinking)
                        iter.remove();
                    if(G < 0)
                        PG = G;
                }
                else if(alpha[i] == U)
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
                    alpha[i] = Math.min(Math.max(alpha[i]-G/(getQ(i)+D), 0), U);
                    double scale = (alpha[i]-alphaOld)*y[i];
                    w.mutableAdd(scale, vecs[i]);
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
        alpha = null;
        y = null;
    }
    
    private double getQ(int i)
    {
        return vecs[i].dot(vecs[i])+1;//+1 for implicit bias term
    }
    
    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public Classifier clone()
    {
        DCDs clone = new DCDs(maxIterations, tolerance, C, useL1);
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
