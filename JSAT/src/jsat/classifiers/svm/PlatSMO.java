
package jsat.classifiers.svm;

import static java.lang.Math.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.*;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.Vec;
import jsat.parameters.*;
import jsat.utils.IntSetFixedSize;

/**
 * An implementation of SVMs using Plat's Sequential Minimum Optimization
 * <br><br>
 * See:<br>
 * <ul>
 * <li>Platt, J. C. (1998). <i>Sequential Minimal Optimization: A Fast Algorithm
 * for Training Support Vector Machines</i>. Advances in kernel methods 
 * (pp. 185 – 208). Retrieved from <a href="http://www.bradblock.com/Sequential_Minimal_Optimization_A_Fast_Algorithm_for_Training_Support_Vector_Machine.pdf">here</a></li>
 * <li>Keerthi, S. S., Shevade, S. K., Bhattacharyya, C., & Murthy, K. R. K. 
 * (2001). <i>Improvements to Platt’s SMO Algorithm for SVM Classifier Design
 * </i>. Neural Computation, 13(3), 637–649. doi:10.1162/089976601300014493</li>
 * </ul>
 * 
 * @author Edward Raff
 */
public class PlatSMO extends SupportVectorMachine implements Parameterized
{
    /**
     * Bias
     */
    protected double b = 0, b_low, b_up;
    private double C = 0.05;
    private double tolerance = 1e-4;
    private double epsilon = 1e-3;
    
    private int maxIterations = 10000;
    private boolean modificationOne = false;
    
    /**
     * During training contains alphas in [0, C]. After training they are given 
     * the sign of the label so that it need not be kept. 
     */
    protected double[] alpha;
    protected double[] fcache;
    
    private int i_up, i_low;
    
    /* NOTE: Only I_0 needs to be iterated over, so make it a set to iterate 
     * quickly. All others only need set/check, so just use a boolean array. 
     * This saves memory. (bools default false, so they start out all 'empty')
     */
    
    /**
     * i : 0 < a_i < C
     */
    Set<Integer> I0;
    /**
     * i: y_i = 1 AND  a_i = 0
     */
    boolean[] I1;
    /**
     * i: y_i = -1 AND a_i = C
     */
    boolean[] I2;
    /**
     * i: y_i = 1 AND a_i = C
     */
    boolean[] I3;
    /**
     * i: y_i = -1 AND a_i = 0
     */
    boolean[] I4;
    
    protected double[] label;
    
    /**
     * Creates a new SVM object that uses the fill cache mode. 
     * 
     * @param kf 
     * @see CacheMode
     */
    public PlatSMO(KernelTrick kf)
    {
        super(kf, SupportVectorMachine.CacheMode.FULL);
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(vecs == null)
            throw new UntrainedModelException("Classifier has yet to be trained");
        
        double sum = 0;
        CategoricalResults cr = new CategoricalResults(2);
        
        for (int i = 0; i < vecs.length; i++)
            sum += alpha[i] * kEval(vecs[i], data.getNumericalValues());


        //SVM only says yess / no, can not give a percentage
        if(sum > b)
            cr.setProb(1, 1);
        else
            cr.setProb(0, 1);
        
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
            throw new FailedToFitException("SVM does not support non binary decisions");
        //First we need to set up the vectors array

        vecs = new Vec[dataSet.getSampleSize()];
        label = new double[vecs.length];
        b = 0;
        for(int i = 0; i < vecs.length; i++)
        {
            DataPoint dataPoint = dataSet.getDataPoint(i);
            vecs[i] = dataPoint.getNumericalValues();
            if(dataSet.getDataPointCategory(i) == 0)
                label[i] = -1;
            else
                label[i] = 1;
        }
        
        setCacheMode(getCacheMode());//Initiates the cahce
        
        I0 = new IntSetFixedSize(vecs.length);
        I1 = new boolean[vecs.length];
        I2 = new boolean[vecs.length];
        I3 = new boolean[vecs.length];
        I4 = new boolean[vecs.length];
        
        
        //initialize alpha array to all zero
        alpha = new double[vecs.length];//zero is default value
        fcache = new double[vecs.length];
        
        i_up = i_low = -1;//giberish for init
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            if(dataSet.getDataPointCategory(i) == 0)
            {
                label[i] = -1;
                i_low = i;
                I4[i] = true;
            }
            else
            {
                label[i] = 1;
                i_up = i;
                I1[i] = true;
            }
        
        b_up  = -1;
        fcache[i_up]  = -1;
        b_low =  1;
        fcache[i_low] =  1;

        int numChanged = 0;
        boolean examinAll = true;

        int examinAllCount = 0;
        int iter = 0;
        while( (examinAll || numChanged > 0) && iter < maxIterations )
        {
            iter++;
            numChanged = 0;
            if (examinAll)
            {
                //loop I over all training examples
                for (int i = 0; i < vecs.length; i++)
                    numChanged += examineExample(i);
                examinAllCount++;
            }
            else
            {
                if (modificationOne)
                {
                    for (int i : I0)
                    {
                        numChanged += examineExample(i);
                        if (b_up > b_low - 2 * tolerance)
                        {
                            numChanged = 0;//causes examinAll to become true
                            break;
                        }
                    }
                }
                else
                {
                    boolean inner_loop_success = true;

                    while (b_up < b_low - 2 * tolerance && inner_loop_success)
                        if (inner_loop_success = takeStep(i_up, i_low))
                            numChanged++;
                }
                
                numChanged = 0;
            }

            if(examinAll)
                examinAll = false;
            else if(numChanged == 0)
                examinAll = true;
        }
        
        b = (b_up+b_low)/2;

        //SVMs are usualy sparse, we dont need to keep all the original vectors!
        //collapse label into signed alphas
        for(int i = 0; i < label.length; i++)
            alpha[i] *= label[i];
        
        int supportVectorCount = 0;
        for(int i = 0; i < vecs.length; i++)
            if(alpha[i] > 0 || alpha[i] < 0)//Its a support vector
            {
                vecs[supportVectorCount] = vecs[i];
                alpha[supportVectorCount++] = alpha[i];
            }

        vecs = Arrays.copyOfRange(vecs, 0, supportVectorCount);
        alpha = Arrays.copyOfRange(alpha, 0, supportVectorCount);
        label = null;
        
        fcache = null;
        I0 = null;
        I1 = I2 = I3 = I4 = null;
    }
    
    /**
     * Updates the index set I0 
     * @param i1 the value of i1
     * @param a1 the value of a1
     */
    private void updateSet(int i1, double a1)
    {
        if(a1 > 0 && a1 < C)
            I0.add(i1);
        else
            I0.remove(i1);
    }
    
    /**
     * Updates the index sets 
     * @param i1 the index to update for
     * @param a1 the alpha value for the index
     */
    private void updateSetsLabeled(int i1, double a1)
    {
        if(label[i1] == 1)
        {
            if(a1 == 0)
            {
                I1[i1] = true;
                I3[i1] = false;
            }
            else if(a1 == C)
            {
                I1[i1] = false;
                I3[i1] = true;
            }
            else
            {
                I1[i1] = false;
                I3[i1] = false;
            }
        }
        else
        {
            if(a1 == 0)
            {
                I4[i1] = true;
                I2[i1] = false;
            }
            else if(a1 == C)
            {
                I4[i1] = false;
                I2[i1] = true;
            }
            else
            {
                I4[i1] = false;
                I2[i1] = false;
            }
        }
    }
    
    protected boolean takeStep(int i1, int i2)
    {
        if(i1 == i2)
            return false;
        //alph1 = Lagrange multiplier for i
        double alpha1 = alpha[i1], alpha2 = alpha[i2];
        //y1 = target[i1]
        double y1 = label[i1], y2 = label[i2];
        double F1 = fcache[i1];
        double F2 = fcache[i2];

        //s = y1*y2
        double s = y1*y2;

        //Compute L, H : see smo-book, page 46
        double L, H;
        if(y1 != y2)
        {
            L = max(0, alpha2-alpha1);
            H = min(C, C+alpha2-alpha1);
        }
        else
        {
            L = max(0, alpha1+alpha2-C);
            H = min(C, alpha1+alpha2);
        }

        if (L == H)
            return false;

        double a1;//new alpha1
        double a2;//new alpha2

        /*
         * k11 = kernel(point[i1],point[i1])
         * k12 = kernel(point[i1],point[i2])
         * k22 = kernel(point[i2],point[i2]
         */
        double k11 = kEval(i1, i1);
        double k12 = kEval(i1, i2);
        double k22 = kEval(i2, i2);
        //eta = 2*k12-k11-k22
        double eta = 2*k12 - k11 - k22;

        if (eta < 0)
        {
            a2 = alpha2 - y2 * (F1 - F2) / eta;
            if (a2 < L)
                a2 = L;
            else if (a2 > H)
                a2 = H;
        }
        else
        {
            /*
             * Lobj = objective function at a2=L
             * Hobj = objective function at a2=H
             */
            
            double L1 = alpha1 + s * (alpha2 - L);
            double H1 = alpha1 + s * (alpha2 - H);
            double f1 = y1 * F1 - alpha1 * k11 - s * alpha2 * k12;
            double f2 = y2 * F2 - alpha2 * k22 - s * alpha1 * k12;
            double Lobj = -0.5 * L1 * L1 * k11 - 0.5 * L * L * k22 - s * L * L1 * k12 - L1 * f1 - L * f2;
            double Hobj = -0.5 * H1 * H1 * k11 - 0.5 * H * H * k22 - s * H * H1 * k12 - H1 * f1 - H * f2;

            if(Lobj > Hobj + epsilon)
                a2 = L;
            else if(Lobj < Hobj - epsilon)
                a2 = H;
            else
                a2 = alpha2;
        }

        if(a2 < 1e-8)
            a2 = 0;
        else if (a2 > C - 1e-8)
            a2 = C;

        if(abs(a2 - alpha2) < epsilon*(a2+alpha2+epsilon))
            return false;

        a1 = alpha1 + s *(alpha2-a2);

        if (a1 < 0)
        {
            a2 += s * a1;
            a1 = 0;
        }
        else if (a1 > C)
        {
            double t = a1 - C;
            a2 += s * t;
            a1 = C;
        }

        double newF1C = F1 + y1*(a1-alpha1)*k11 + y2*(a2-alpha2)*k12;
        double newF2C = F2 + y1*(a1-alpha1)*k12 + y2*(a2-alpha2)*k22;
        
        if(abs(newF1C-fcache[i1]) < 1e-10 && abs(newF2C-fcache[i2]) < 1e-10)
            return false;
        
        updateSet(i1, a1);
        updateSet(i2, a2);
        
        updateSetsLabeled(i1, a1);
        updateSetsLabeled(i2, a2);
        
        fcache[i1] = newF1C;
        fcache[i2] = newF2C;
        
        b_low = Double.NEGATIVE_INFINITY;
        b_up = Double.POSITIVE_INFINITY;
        i_low = -1;
        i_up = -1;
        
        for (int i : I0)
        {
            double bCand = fcache[i];
            if (bCand > b_low)
            {
                i_low = i;
                b_low = bCand;
            }

            if (bCand < b_up)
            {
                i_up = i;
                b_up = bCand;
            }
        }

        //Store a1 in the alpha array
        alpha[i1] = a1;
        //Store a2 in the alpha arra
        alpha[i2] = a2;

        return true;
    }
    
    private int examineExample(int i2)
    {
        //y2 = target[i2]
        double y2 = label[i2];
        
        double F2;
        if(I0.contains(i2))
            F2 = fcache[i2];
        else
        {
            fcache[i2] = F2 = decisionFunction(i2) - y2;
            //update (b_low, i_low) or (b_up, i_up) using (F2, i2)
            if( (I1[i2] || I2[i2] ) && (F2 < b_up)  )
            {
                b_up = F2;
                i_up = i2;
            }
            else if( (I3[i2] || I4[i2]) && (F2 > b_low) )
            {
                b_low = F2;
                i_low = i2;
            }
        }
        
        //check optimality using current b_low and b_up and, if violated, find 
        //an index i1 to do joint optimization ith i2
        boolean optimal = true;
        int i1 = -1;//giberish init value will not get used, but makes compiler smile
        
        final boolean I0_contains_i2 = I0.contains(i2);
        if(I0_contains_i2 || I1[i2] || I2[i2])
        {
            if(b_low - F2 > tolerance*2)
            {
                optimal = false;
                i1 = i_low;
            }
        }
        
        if(I0_contains_i2 || I3[i2] || I4[i2])
        {
            if(F2-b_up > tolerance*2)
            {
                optimal = false;
                i1 = i_up;
            }
        }
        
        if(optimal)//no changes if optimal
            return 0;

        //for i2 in I0 choose the better i1
        if(I0_contains_i2)
        {
            if(b_low-F2 > F2-b_up)
                i1 = i_low;
            else 
                i1 = i_up;
        }

        if(takeStep(i1, i2))
            return 1;
        else
            return 0;
    }
    
    /**
     * Returns the local decision function for training purposes without the bias temr
     * @param v the index of the point to select
     * @return the decision function output sans bias
     */
    protected double decisionFunction(int v)
    {
        double sum = 0;
        for(int i = 0; i < vecs.length; i++)
            if(alpha[i] > 0)
                sum += alpha[i] * label[i] * kEval(i, v);

        return sum;
    }

    @Override
    public Classifier clone()
    {
        PlatSMO copy = new PlatSMO(this.getKernel().clone());
        
        copy.C = this.C;
        if(this.alpha != null)
            copy.alpha = Arrays.copyOf(this.alpha, this.alpha.length);
        copy.b = this.b;
        copy.epsilon = this.epsilon;
        if(this.label != null)
            copy.label = Arrays.copyOf(this.label, this.label.length);
        copy.tolerance = this.tolerance;
        if(this.vecs != null)
            copy.vecs = Arrays.copyOf(this.vecs, this.vecs.length);
        
        return copy;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    /**
     * Sets the complexity parameter of SVM. The larger the C value the harder 
     * the margin SVM will attempt to find. Lower values of C allow for more 
     * misclassification errors. 
     * @param C the soft margin parameter
     */
    public void setC(double C)
    {
        if(C <= 0)
            throw new ArithmeticException("C must be a positive constant");
        this.C = C;
    }

    /**
     * Returns the soft margin complexity parameter of the SVM
     * @return the complexity parameter of the SVM
     */
    public double getC()
    {
        return C;
    }

    /**
     * Sets the maximum number of iterations to perform of the training loop. 
     * This is important for cases with a C value that is to large for a non 
     * linear problem, which can result in SVM failing to converge. 
     * @param maxIterations the maximum number of main iteration loops
     */
    public void setMaxIterations(int maxIterations)
    {
        this.maxIterations = maxIterations;
    }

    /**
     * Returns the maximum number of iterations
     * @return the maximum number of iterations
     */
    public int getMaxIterations()
    {
        return maxIterations;
    }

    /**
     * Sets where or not modification one or two should be used when training. 
     * Modification two is more aggressive, but often results in less kernel 
     * evaluations. 
     * 
     * @param modificationOne {@code true} to us modificaiotn one, {@code false}
     * to use modification two. 
     */
    public void setModificationOne(boolean modificationOne)
    {
        this.modificationOne = modificationOne;
    }

    /**
     * Returns true if modification one is in use
     * @return true if modification one is in use
     */
    public boolean isModificationOne()
    {
        return modificationOne;
    }

    /**
     * Sets the tolerance for the solution. Higher values converge to worse 
     * solutions, but do so faster
     * @param tolerance the tolerance for the solution
     */
    public void setTolerance(double tolerance)
    {
        this.tolerance = tolerance;
    }

    /**
     * Returns the solution tolerance 
     * @return the solution tolerance 
     */
    public double getTolerance()
    {
        return tolerance;
    }
    
    
    private List<Parameter> params = Parameter.getParamsFromMethods(this);
    private Map<String, Parameter> paramMap = Parameter.toParameterMap(params);
    
    @Override
    public List<Parameter> getParameters()
    {
        List<Parameter> retParams = new ArrayList<Parameter>(params);
        retParams.addAll(getKernel().getParameters());
        return retParams;
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        Parameter toRet = paramMap.get(paramName);
        if(toRet == null)
            toRet = getKernel().getParameter(paramName);
        return toRet;
    }

}
