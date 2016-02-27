
package jsat.classifiers.svm;

import static java.lang.Math.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.distributions.Distribution;
import jsat.distributions.Exponential;
import jsat.distributions.LogUniform;
import jsat.distributions.kernels.KernelTrick;
import jsat.distributions.kernels.LinearKernel;
import jsat.distributions.kernels.RBFKernel;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.ConstantVector;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.parameters.*;
import jsat.parameters.Parameter.WarmParameter;
import jsat.regression.*;
import jsat.utils.ListUtils;

/**
 * An implementation of SVMs using Platt's Sequential Minimum Optimization (SMO) 
 * for both Classification and Regression problems. <br>
 * <br>
 * This algorithm can be warm started for classification problems by any
 * algorithm implementing the {@link BinaryScoreClassifier} interface. For
 regression any algorithm can be used as a warms start. For best results, warm
 starts should be from algorithms that will have a similar solution to
 PlattSMO.
 <br><br>
 * See:<br>
 * <ul>
 * <li>Platt, J. C. (1998). <i>Sequential Minimal Optimization: A Fast Algorithm
 * for Training Support Vector Machines</i>. Advances in kernel methods 
 * (pp. 185 – 208). Retrieved from <a href="http://www.bradblock.com/Sequential_Minimal_Optimization_A_Fast_Algorithm_for_Training_Support_Vector_Machine.pdf">here</a></li>
 * <li>Keerthi, S. S., Shevade, S. K., Bhattacharyya, C.,&amp;Murthy, K. R. K. 
 * (2001). <i>Improvements to Platt’s SMO Algorithm for SVM Classifier Design
 * </i>. Neural Computation, 13(3), 637–649. doi:10.1162/089976601300014493</li>
 * <li>Smola, A. J.,&amp;Schölkopf, B. (2004). <i>A tutorial on support vector 
 * regression</i>. Statistics and Computing, 14(3), 199–222. 
 * doi:10.1023/B:STCO.0000035301.49549.88</li>
 * <li>Shevade, S. K., Keerthi, S. S., Bhattacharyya, C.,&amp;Murthy, K. K. (1999)
 * . <i>Improvements to the SMO algorithm for SVM regression</i>. Control D
 * ivision, Dept. of Mechanical Engineering CD-99–16. Control Division, Dept. of
 * Mechanical Engineering. doi:10.1109/72.870050</li>
 * <li>Shevade, S. K., Keerthi, S. S., Bhattacharyya, C.,&amp;Murthy, K. K. (2000)
 * . <i>Improvements to the SMO algorithm for SVM regression</i>. IEEE 
 * transactions on neural networks / a publication of the IEEE Neural Networks 
 * Council, 11(5), 1188–93. doi:10.1109/72.870050</li>
 * </ul>
 * 
 * @author Edward Raff
 */
public class PlattSMO extends SupportVectorLearner implements BinaryScoreClassifier, WarmRegressor, Parameterized, WarmClassifier
{

    private static final long serialVersionUID = 1533410993462673127L;
    /**
     * Bias
     */
    protected double b = 0, b_low, b_up;
    private double C = 1;
    private double tolerance = 1e-3;
    private double eps = 1e-7;
    private double epsilon = 1e-2;

    private int maxIterations = 10000;
    private boolean modificationOne = true;
    
    protected double[] fcache;
    
    private int i_up, i_low;
    
    /* NOTE: Only I_0 needs to be iterated over, so make it a set to iterate 
     * quickly. All others only need set/check, so just use a boolean array. 
     * This saves memory. (bools default false, so they start out all 'empty')
     */
    
    /*
     * used only in regression 
     */
    private double[] alpha_s;
    /**
     * i : 0 &lt; a_i &lt; C
     * <br>
     * For regression this contains both of I0_a and I0_b
     */
    private boolean[] I0;
    /**
     * Indicates if the value that is currently in I0 is also in I0_a. If its
     * not in I0, the value in this array is false 
     */
    private boolean[] I0_a;
    /**
     * Indicates if the value that is currently in I0 is also in I0_b. If its
     * not in I0, the value in this array is false
     */
    private boolean[] I0_b;
    /**
     * i: y_i = 1 AND  a_i = 0
     */
    private boolean[] I1;
    /**
     * i: y_i = -1 AND a_i = C
     */
    private boolean[] I2;
    /**
     * i: y_i = 1 AND a_i = C
     */
    private boolean[] I3;
    /**
     * i: y_i = -1 AND a_i = 0
     */
    private boolean[] I4;
    
    /**
     * Stores the true value of the data point
     */
    protected double[] label;
    /**
     * Weight values to apply to each data point
     */
    protected Vec weights;
    
    /**
     * Creates a new SVM object with a {@link LinearKernel} that uses no cache
     * mode.
     *
     */
    public PlattSMO()
    {
        this(new LinearKernel());
    }
    
    /**
     * Creates a new SVM object that uses no cache mode. 
     * 
     * @param kf the kernel trick to use
     */
    public PlattSMO(KernelTrick kf)
    {
        super(kf, SupportVectorLearner.CacheMode.NONE);
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(vecs == null)
            throw new UntrainedModelException("Classifier has yet to be trained");
        
        CategoricalResults cr = new CategoricalResults(2);
        
        double sum = getScore(data);

        if(sum > 0)
            cr.setProb(1, 1);
        else
            cr.setProb(0, 1);
        
        return cr;
    }

    @Override
    public double getScore(DataPoint dp)
    {
        return kEvalSum(dp.getNumericalValues())-b;
    }
    
    @Override
    public void trainC(ClassificationDataSet dataSet, Classifier warmSolution, ExecutorService threadPool)
    {
        trainC(dataSet, warmSolution);
    }
    
    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, Classifier warmSolution)
    {
        trainC_warm_and_normal(dataSet, warmSolution);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC_warm_and_normal(dataSet, null);
    }
    
    private void trainC_warm_and_normal(ClassificationDataSet dataSet, Classifier warmSolution)
    {
        if(dataSet.getClassSize() != 2)
            throw new FailedToFitException("SVM does not support non binary decisions");
        //First we need to set up the vectors array

        final int N = dataSet.getSampleSize();
        vecs = new ArrayList<Vec>(N);
        label = new double[N];
        weights = new DenseVector(N);
        b = 0;
        i_up = i_low = -1;//giberish for init
        I0 = new boolean[N];
        I1 = new boolean[N];
        I2 = new boolean[N];
        I3 = new boolean[N];
        I4 = new boolean[N];
        
        boolean allWeightsAreOne = true;
        for(int i = 0; i < N; i++)
        {
            DataPoint dataPoint = dataSet.getDataPoint(i);
            vecs.add(dataPoint.getNumericalValues());
            weights.set(i, dataPoint.getWeight());
            if(dataPoint.getWeight() != 1)
                allWeightsAreOne = false;
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
        }
        if(allWeightsAreOne)//if everything == 1, don't waste the memory storying it
            weights = new ConstantVector(1.0, N);
        
        setCacheMode(getCacheMode());//Initiates the cahce
        
        //initialize alphas array to all zero
        alphas = new double[N];//zero is default value
        fcache = new double[N];
        
        b_up  = -1;
        fcache[i_up]  = -1;
        b_low =  1;
        fcache[i_low] =  1;
        boolean examinAll = true;
        
        //Now lets try and do some warm starting if applicable
        if(warmSolution instanceof PlattSMO || warmSolution instanceof BinaryScoreClassifier)
        {
            examinAll = false;
            
            WarmScope://We need to use one of the methods to detemrine if the model is a fit
            {
                if (warmSolution instanceof PlattSMO)
                {
                    //if this SMO object was learend on the same data, we can get a very good guess on C
                    PlattSMO warmSMO = (PlattSMO) warmSolution;

                    //first, we need to make sure we were actually trained on the same data
                    //TODO find a better way to ensure this is true, it is POSSIBLE that we could have same labels and different data
                    if(warmSMO.alphas == null)//whats going on? just break out
                        break WarmScope;
                    boolean sameData = alphas.length == warmSMO.alphas.length;
                    if(sameData)
                        for(int i = 0; i < this.label.length && sameData; i++)
                            //copy sign used so that -0.0 gets picked up as -1.0 and 0.0 as 1.0
                            if(this.label[i] != Math.copySign(1.0, warmSMO.alphas[i]))
                                sameData = false;
                        
                    if(sameData)
                    {
                        double C_prev = warmSMO.C;
                        double multiplier = this.C / C_prev;
                        for (int i = 0; i < vecs.size(); i++)
                            this.alphas[i] = fuzzyClamp(multiplier * Math.abs(warmSMO.alphas[i]), this.C);
                        break WarmScope;//init sucessful
                    }
                    //else, fall through and let 2nd case cick in
                }

                //last case, should be true
                if (warmSolution instanceof BinaryScoreClassifier)
                {
                    //In this case we can take a decent guess at the values of alpha
                    BinaryScoreClassifier warmSC = (BinaryScoreClassifier) warmSolution;

                    for (int i = 0; i < vecs.size(); i++)
                    {
                        //get the loss, normaly wrapped by max(x, 0), but that will be handled by the clamp
                        double guess = 1 - label[i] * warmSC.getScore(dataSet.getDataPoint(i));
                        this.alphas[i] = fuzzyClamp(C * guess, C);
                    }
                }
                else//how did this happen?
                {
                    throw new FailedToFitException("BUG: Should not have been able to reach");
                }
            }
            fcache[i_up]  = 0;
            fcache[i_low] = 0;
            for (int i = 0; i < vecs.size(); i++)
            {

                //we can't skip a_i == 0 b/c we still need to make the contribution to w_dot_x
                final double a_i = this.alphas[i];
                for (int j = i; j < vecs.size(); j++)
                {
                    final double a_j = this.alphas[j];
                    if (a_j == 0 && a_i == 0)
                        continue;

                    double dot = kEval(i, j);

                    if (i != j)//avoid double counting
                        fcache[j] += a_i * label[i] * dot;
                    fcache[i] += a_j * label[j] * dot;

                }

            }

            //determine i_up and i_low based on equations (11a) and (11b) in Keerthi et al
            for (int i = 0; i < vecs.size(); i++)
            {
                fcache[i] -= label[i];
                
                updateSet(i, alphas[i], C*weights.get(i));
                updateSetsLabeled(i, alphas[i], C*weights.get(i));
                
                if(label[i] == -1)
                    if(I0[i] && (i_low == -1 || fcache[i] > fcache[i_low]) )
                    {
                        i_low = i;
                        b_low = fcache[i];
                    }
                else
                    if(I0[i] && (i_low == -1 || fcache[i] > fcache[i_up]) )
                    {
                        i_up = i;
                        b_up = fcache[i];
                    }
            }
        }

        int numChanged = 0;
        
        int examinAllCount = 0;
        int iter = 0;
        while( (examinAll || numChanged > 0) && iter < maxIterations )
        {
            iter++;
            numChanged = 0;
            if (examinAll)
            {
                //loop I over all training examples
                for (int i = 0; i < N; i++)
                    numChanged += examineExample(i);
                examinAllCount++;
            }
            else
            {
                if (modificationOne)
                {
                    for(int i = 0; i < I0.length; i++)
                    {
                        if(!I0[i])
                            continue;
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
                    
                    numChanged = 0;
                }
                
            }

            if(examinAll)
                examinAll = false;
            else if(numChanged == 0)
                examinAll = true;
        }
        
        if (iter >= maxIterations)
        {//1 extra pass to get a better guess on bUp & bLow since we quit early
            for (int i = 0; i < N; i++)
                numChanged += examineExample(i);
        }
        b = (b_up+b_low)/2;
        
        //collapse label into signed alphas
        for(int i = 0; i < label.length; i++)
            alphas[i] *= label[i];
        
//        sparsify();
        label = null;
        
        fcache = null;
        I0 = I1 = I2 = I3 = I4 = null;
        weights = null;
        
        setCacheMode(null);
        setAlphas(alphas);
    }
    
    /**
     * Updates the index set I0 
     * @param i1 the value of i1
     * @param a1 the value of a1
     * @param C the regularization value to use for this datum
     */
    private void updateSet(int i1, double a1, double C )
    {
        I0[i1] = a1 > 0 && a1 < C;
    }
    
    private double fuzzyClamp(double val, double max)
    {
        return fuzzyClamp(val, max, max*1e-7);
    }
    
    private double fuzzyClamp(double val, double max, double e)
    {
        if(val > max-e)
            return max;
        if(val < e)
            return 0;
        return val;
    }
    
    private void updateSetR(int i, double C)
    {
        /**
         * See page 5 of he pseudo code paper version of "Improvements to the SMO algorithm for SVM regression."
         */
        //TODO, this can be done with less work.. but I messed that up
        double a_i = alphas[i];
        double as_i = alpha_s[i];
        I0_a[i] = 0 < a_i && a_i < C;
        I0_b[i] = 0 < as_i && as_i < C;
        I0[i] = I0_a[i] || I0_b[i];
        I1[i] = a_i == 0 && as_i == 0;
        I2[i] = a_i == 0 && as_i == C;
        I3[i] = a_i == C && as_i == 0;
    }

    /**
     * Updates the index sets 
     * @param i1 the index to update for
     * @param a1 the alphas value for the index
     * @param C the regularization value to use for this datum
     */
    private void updateSetsLabeled(int i1, final double a1, final double C)
    {
        final double y_i = label[i1];
        I1[i1] = a1 == 0 && y_i == 1;
        I2[i1] = a1 == C && y_i == -1;
        I3[i1] = a1 == C && y_i == 1;
        I4[i1] = a1 == 0 && y_i == -1;
    }
    
    protected boolean takeStep(int i1, int i2)
    {
        if(i1 == i2)
            return false;
        //alph1 = Lagrange multiplier for i
        double alpha1 = alphas[i1], alpha2 = alphas[i2];
        //y1 = target[i1]
        double y1 = label[i1], y2 = label[i2];
        double F1 = fcache[i1];
        double F2 = fcache[i2];
        final double C1 = C*weights.get(i1);
        final double C2 = C*weights.get(i2);

        //s = y1*y2
        double s = y1*y2;

        //Compute L, H : see smo-book, page 46
        //also "A tutorial on support vector regression" page 30
        double L, H;
        if(y1 != y2)
        {
            L = max(0, alpha2-alpha1);
            H = min(C2, C1+alpha2-alpha1);
        }
        else
        {
            L = max(0, alpha1+alpha2-C1);
            H = min(C2, alpha1+alpha2);
        }

        if (L >= H)//>= instead of == incase of numerical issues
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
            
            if(Lobj > Hobj + eps)
                a2 = L;
            else if(Lobj < Hobj - eps)
                a2 = H;
            else
                a2 = alpha2;
        }

        a2 = fuzzyClamp(a2, C2);

        if(abs(a2 - alpha2) < eps*(a2+alpha2+eps))
            return false;
        
        a1 = alpha1 + s *(alpha2-a2);
        a1 = fuzzyClamp(a1, C1);

        double newF1C = F1 + y1*(a1-alpha1)*k11 + y2*(a2-alpha2)*k12;
        double newF2C = F2 + y1*(a1-alpha1)*k12 + y2*(a2-alpha2)*k22;
        
        updateSet(i1, a1, C1);
        updateSet(i2, a2, C2);
        
        updateSetsLabeled(i1, a1, C1);
        updateSetsLabeled(i2, a2, C2);
        
        fcache[i1] = newF1C;
        fcache[i2] = newF2C;
        
        b_low = Double.NEGATIVE_INFINITY;
        b_up = Double.POSITIVE_INFINITY;
        i_low = -1;
        i_up = -1;
        
        //"Update fcache[i] for i in I_0 using new Lagrange multipliers", done inside loop check for new bounds
        for(int i = 0; i < I0.length; i++)
        {
            if(!I0[i])
                continue;
            if (i != i1 && i != i2)
                fcache[i] += y1 * (a1 - alpha1) * kEval(i1, i) + y2 * (a2 - alpha2) * kEval(i2, i);
            
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
        
        //case where i1 & i2 are not in I0
        for(int i : new int[]{i1, i2})
        {
            if(I3[i] || I4[i])
            {
                double bCand = fcache[i];
                if (bCand > b_low )
                {
                    i_low = i;
                    b_low = bCand;
                }
            }
            if(I1[i] || I2[i])
            {
                double bCand = fcache[i];
                if (bCand < b_up )
                {
                    i_up = i;
                    b_up = bCand;
                }
            }
        }
        
        //Store a1 in the alphas array
        alphas[i1] = a1;
        //Store a2 in the alphas arra
        alphas[i2] = a2;

        return true;
    }
    
    protected boolean takeStepR(int i1, int i2)
    {
        if(i1 == i2)
            return false;
        //alph1 = Lagrange multiplier for i
        double alpha1 = alphas[i1], alpha2 = alphas[i2];
        double alpha1_S = alpha_s[i1], alpha2_S = alpha_s[i2];
        double F1 = fcache[i1];//phi1 in paper 
        double F2 = fcache[i2];
        final double C1 = C*weights.get(i1);
        final double C2 = C*weights.get(i2);

        /*
         * k11 = kernel(point[i1],point[i1])
         * k12 = kernel(point[i1],point[i2])
         * k22 = kernel(point[i2],point[i2]
         */
        double k11 = kEval(i1, i1);
        double k12 = kEval(i2, i1);
        double k22 = kEval(i2, i2);
        //eta = -2*k12+k11+k22
        double eta = -2*k12 + k11 + k22;
        if(eta < 0)
            eta = 0;//lets just assume it was a numerical issue... (dirty NPSD kernels)
        
        //gamma = alpha1-alpha1*+alpha2-alpha2*
        double gamma = alpha1-alpha1_S+alpha2-alpha2_S;
        
        boolean case1, case2, case3, case4, finished;
        case1 = case2 = case3 = case4 = finished = false;
        double alpha1_old = alpha1, alpha1_oldS = alpha1_S;
        double alpha2_old = alpha2, alpha2_oldS = alpha2_S;
        double deltaPhi = F1-F2;

        double L, H;
        while(!finished)//occurs at most 3 times
        {
            if(!case1 &&
                    (alpha1 > 0 || (alpha1_S == 0 && deltaPhi > 0) ) &&
                    (alpha2 > 0 || (alpha2_S == 0 && deltaPhi < 0) ) )
            {
                //compute L, H, (wrt. alpha1, alpha2)
                L = max(0, gamma-C1);
                H = min(C2, gamma);
                if(L < H)
                {
                    double a2 = max(L, min(alpha2 - deltaPhi/eta, H));
                    a2 = fuzzyClamp(a2, C2);
                    double a1 = alpha1 - (a2 - alpha2);
                    a1 = fuzzyClamp(a1, C1);
                    if(abs(alpha1-a1) > 1e-10 || abs(a2-alpha2) > 1e-10)
                    {
                        deltaPhi += (a2-alpha2)*eta;
                        alpha1 = a1;
                        alpha2 = a2;
                    }
                }
                else 
                    finished = true;
                case1 = true;
            }
            else if(!case2 && 
                    (alpha1 > 0 || (alpha1_S == 0 && deltaPhi > 2*epsilon)) && 
                    (alpha2_S > 0 || (alpha2 == 0 && deltaPhi > 2*epsilon)))
            {
                //compute L, H, (wrt. alpha1, alpha2*)
                L = max(0, -gamma);
                H = min(C2, -gamma+C1);
                if(L < H)
                {
                    double a2 = max(L, min(alpha2_S + (deltaPhi-2*epsilon)/eta, H));
                    a2 = fuzzyClamp(a2, C2);
                    double a1 = alpha1 + (a2 - alpha2_S);
                    a1 = fuzzyClamp(a1, C1);
                    if(abs(alpha1-a1) > 1e-10 || abs(alpha2_S-a2) > 1e-10)
                    {
                        deltaPhi += (alpha2_S-a2)*eta;
                        alpha1 = a1;
                        alpha2_S = a2;
                    }
                }
                else
                    finished = true;
                case2 = true;
            }
            else if(!case3 && 
                    (alpha1_S > 0 || (alpha1 == 0 && deltaPhi < -2*epsilon)) &&
                    (alpha2 > 0 || (alpha2_S == 0 && deltaPhi < -2*epsilon)))
            {
                //compute L, H, (wrt. alpha1*, alpha2)
                L = max(0, gamma);
                H = min(C2, C1+gamma);
                if(L < H)
                {
                    double a2 = max(L, min(alpha2 - (deltaPhi+2*epsilon)/eta, H));
                    a2 = fuzzyClamp(a2, C2);
                    double a1 = alpha1_S + (a2 - alpha2);
                    a1 = fuzzyClamp(a1, C1);
                    if(abs(alpha1_S-a1) > 1e-10 || abs(alpha2-a2) > 1e-10)
                    {
                        deltaPhi += (a2-alpha2)*eta;
                        alpha1_S = a1;
                        alpha2 = a2;
                    }
                }
                else
                    finished = true;
                case3 = true;
            }
            else if(!case4 &&
                    (alpha1_S > 0 || (alpha1 == 0 && deltaPhi < 0)) &&
                    (alpha2_S > 0 || (alpha2 == 0 && deltaPhi > 0)))
            {
                //compute L, H, (wrt. alpha1*, alpha2*)
                L = max(0, -gamma-C1);
                H = min(C2, -gamma);
                if(L < H)
                {
                    double a2 = max(L, min(alpha2_S + deltaPhi/eta, H));
                    a2 = fuzzyClamp(a2, C2);
                    double a1 = alpha1_S - (a2 - alpha2_S);
                    a1 = fuzzyClamp(a1, C1);
                    if(abs(alpha1_S-a1) > 1e-10 || abs(alpha2_S-a2) > 1e-10)
                    {
                        deltaPhi += (alpha2_S-a2)*eta;
                        alpha1_S = a1;
                        alpha2_S = a2;
                    }
                }
                else
                    finished = true;
                case4 = true;
            }
            else
            {
                finished = true;
            }
        }
        //TODO do a check for numerical issues
        
        //end of the while loop, did we change anything?
        if(alpha1 == alpha1_old && alpha1_S == alpha1_oldS &&
                alpha2 == alpha2_old && alpha2_S == alpha2_oldS)
        {
            return false;
        }
        alphas[i1] = alpha1;
        alphas[i2] = alpha2;
        alpha_s[i1] = alpha1_S;
        alpha_s[i2] = alpha2_S;
        
        //Update error cache using new Lagrange multipliers
        double ceof1 = alpha1 - alpha1_old - (alpha1_S - alpha1_oldS);
        double ceof2 = alpha2 - alpha2_old - (alpha2_S - alpha2_oldS);

        for(int i = 0; i < I0.length; i++)
            if(I0[i] && i != i1 && i != i2)
                fcache[i] -= ceof1 * kEval(i1, i) + ceof2 * kEval(i2, i);
        fcache[i1] -= ceof1 * k11 + ceof2 * k12;
        fcache[i2] -= ceof1 * k12 + ceof2 * k22;
        updateSetR(i1, C1);
        updateSetR(i2, C2);
        
        //Update threshold to reflect change in Lagrange multipliers Update
        b_low = Double.NEGATIVE_INFINITY;
        b_up = Double.POSITIVE_INFINITY;
        i_low = -1;
        i_up = -1;
        
        for(int i = 0; i < I0.length; i++)
            if(I0[i])
                updateThreshold(i);
        //may duplicate work... who cares? its just 2 constant time checks
        updateThreshold(i1);
        updateThreshold(i2);

        //These SHOULD ALWAYS BE NON NEGATIVE, i1 and i2 should have a valid update
        if(i_low == -1 || i_up == -1)
            throw new FailedToFitException("BUG: Imposible code block reached. Please report");
        

        return true;
    }
    
    /**
     * Updates the threshold for regression based off of
     * "using only i1, i2, and indices in I_0"
     * @param i the index to update from that MUST have a value in {@link #fcache}
     */
    private void updateThreshold(int i)
    {
        double Fi = fcache[i];
        double F_tilde_i = b_low;
        
        if (I0_b[i] || I2[i])
            F_tilde_i = Fi + epsilon;
        else if (I0_a[i] || I1[i])
            F_tilde_i = Fi - epsilon;
        
        double F_bar_i = b_up;
        if (I0_a[i] || I3[i])
            F_bar_i = Fi - epsilon;
        else if (I0_b[i] || I1[i])
            F_bar_i = Fi + epsilon;
        
        //update the bounds
        
        if (b_low < F_tilde_i)
        {
            b_low = F_tilde_i;
            i_low = i;
        }
        
        if (b_up > F_bar_i)
        {
            b_up = F_bar_i;
            i_up = i;
        }
    }
    
    private int examineExample(int i2)
    {
        //y2 = target[i2]
        double y2 = label[i2];
        
        double F2;
        if(I0[i2])
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
        
        final boolean I0_contains_i2 = I0[i2];
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
    
    private int examineExampleR(int i2)
    {
        //y2 = target[i2]
        double y2 = label[i2];
        
        double F2;
        if(I0[i2])
            F2 = fcache[i2];
        else
        {
            fcache[i2] = F2 = y2-decisionFunctionR(i2);
            //update (b_low, i_low) or (b_up, i_up) using (F2, i2)
            if(I1[i2])
            {
                if(F2+eps < b_up)
                {
                    b_up = F2+epsilon;
                    i_up = i2;
                }
                else if(F2-epsilon > b_low)
                {
                    b_low = F2-epsilon;
                    i_low = i2;
                }
            }
            else if( I2[i2] && (F2+epsilon > b_low)  )
            {
                b_low = F2+epsilon;
                i_low = i2;
            }
            else if( I3[i2] && (F2-epsilon < b_up) )
            {
                b_up = F2-epsilon;
                i_up = i2;
            }
        }
        
        //check optimality using current b_low and b_up and, if violated, find 
        //an index i1 to do joint optimization ith i2
        boolean optimal = true;
        int i1 = -1;//giberish init value will not get used, but makes compiler smile
        
        //5 cases to check
        final double F2mEps = F2-epsilon;
        final double F2pEps = F2+epsilon;
        final double tol2 = 2*tolerance;
        if (I0_a[i2])//case 1
        {
            if (b_low - F2mEps > tol2)
            {
                optimal = false;
                i1 = i_low;
                if (F2mEps - b_up > b_low - F2mEps)
                    i1 = i_up;
            }
            else if (F2mEps - b_up > tol2)
            {
                optimal = false;
                i1 = i_up;
                if (b_low - F2mEps > F2mEps - b_up)
                    i1 = i_low;
            } 
        }
        else if (I0_b[i2])//case 2
        {
            if (b_low - F2pEps > tol2)
            {
                optimal = false;
                i1 = i_low; 
                if (F2pEps - b_up > b_low - F2pEps)
                    i1 = i_up;
            }
            else if (F2pEps - b_up > tol2)
            {
                optimal = false;
                i1 = i_up;
                if (b_low - F2pEps > F2pEps - b_up)
                    i1 = i_low;
            }
        }
        else if (I1[i2])//case 3
        {
            if (b_low - F2pEps > tol2)
            {
                optimal = false;
                i1 = i_low;
                if (F2pEps - b_up > b_low - F2pEps)
                    i1 = i_up;
            }
            else if (F2mEps - b_up > tol2)
            {
                optimal = false;
                i1 = i_up; 
                if (b_low - F2mEps > F2mEps - b_up)
                    i1 = i_low;
            }
        } 
        else if (I2[i2])//case 4
        {
            if (F2pEps - b_up > tol2)
            {
                optimal = false;
                i1 = i_up;
            }
        }
        else if (I3[i2])//case 5
        {
            if (b_low - F2mEps > tol2)
            {
                optimal = false;
                i1 = i_low;
            }
        }

        if(optimal)
            return 0;
        if(takeStepR(i1, i2))
            return 1;
        else
            return 0;
    }
    
    /**
     * Returns the local decision function for classification training purposes 
     * without the bias term
     * @param v the index of the point to select
     * @return the decision function output sans bias
     */
    protected double decisionFunction(int v)
    {
        double sum = 0;
        for(int i = 0; i < vecs.size(); i++)
            if(alphas[i] > 0)
                sum += alphas[i] * label[i] * kEval(v, i);

        return sum;
    }
    
    /**
     * Returns the local decision function for regression training purposes 
     * without the bias term
     * @param v the index of the point to select
     * @return the decision function output sans bias
     */
    protected double decisionFunctionR(int v)
    {
        double sum = 0;
        for (int i = 0; i < vecs.size(); i++)
            if (alphas[i] != alpha_s[i])//multipler would be zero
                sum += (alphas[i] - alpha_s[i]) * kEval(v, i);

        return sum;
    }

    @Override
    public PlattSMO clone()
    {
        PlattSMO copy = new PlattSMO(this.getKernel().clone());
        
        copy.C = this.C;
        if(this.alphas != null)
            copy.alphas = Arrays.copyOf(this.alphas, this.alphas.length);
        if(this.alpha_s != null)
            copy.alpha_s = Arrays.copyOf(this.alpha_s, this.alpha_s.length);
        if(this.weights != null)
            copy.weights = this.weights.clone();
        copy.b = this.b;
        copy.eps = this.eps;
        copy.epsilon = this.epsilon;
        copy.maxIterations = this.maxIterations;
        if(this.label != null)
            copy.label = Arrays.copyOf(this.label, this.label.length);
        copy.tolerance = this.tolerance;
        if(this.vecs != null)
            copy.vecs = new ArrayList<Vec>(this.vecs);
        copy.setCacheMode(this.getCacheMode());
        copy.setCacheValue(this.getCacheValue());
        
        return copy;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }

    /**
     * Sets the complexity parameter of SVM. The larger the C value the harder 
     * the margin SVM will attempt to find. Lower values of C allow for more 
     * misclassification errors. 
     * @param C the soft margin parameter
     */
    @WarmParameter(prefLowToHigh = true)
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
    public double regress(DataPoint data)
    {
        return kEvalSum(data.getNumericalValues())+b;
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        train(dataSet);
    }

    /**
     * Sets the epsilon for the epsilon insensitive loss when performing
     * regression. This variable has no impact during classification problems. 
     * For regression problems, any predicated value that is within the epsilon
     * of the target will be treated as "correct". Increasing epsilon usually 
     * decreases the number of support vectors, but may reduce the accuracy of
     * the model
     * 
     * @param epsilon the positive value for the acceptable error when doing 
     * regression
     */
    public void setEpsilon(double epsilon)
    {
        if(Double.isNaN(epsilon) || Double.isInfinite(epsilon) || epsilon <= 0)
            throw new IllegalArgumentException("epsilon must be in (0, infty), not " + epsilon);
        this.epsilon = epsilon;
    }

    /**
     * Returns the epsilon insensitive loss value
     * @return the epsilon insensitive loss value
     */
    public double getEpsilon()
    {
        return epsilon;
    }
    
    @Override
    public void train(RegressionDataSet dataSet, Regressor warmSolution, ExecutorService threadPool)
    {
        train(dataSet, warmSolution);
    }
    
    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, (Regressor)null);
    }

    @Override
    public void train(RegressionDataSet dataSet, Regressor warmSolution)
    {
        final int N = dataSet.getSampleSize();
        vecs = new ArrayList<Vec>(N);
        label = new double[N];
        fcache = new double[N];
        b = 0;
        weights = new DenseVector(N);
        boolean allWeightsAreOne = true;
        for(int i = 0; i < N; i++)
        {
            DataPoint dataPoint = dataSet.getDataPoint(i);
            vecs.add(dataPoint.getNumericalValues());
            fcache[i] = label[i] = dataSet.getTargetValue(i);
            weights.set(i, dataPoint.getWeight());
            if(dataPoint.getWeight() != 1)
                allWeightsAreOne = false;
        }
        if(allWeightsAreOne)//if everything == 1, don't waste the memory storying it
            weights = new ConstantVector(1.0, N);
        
        setCacheMode(getCacheMode());//Initiates the cahce
        
        I0 = new boolean[N];
        I0_a = new boolean[N];
        I0_b = new boolean[N];
        I1 = new boolean[N];
        I2 = new boolean[N];
        I3 = new boolean[N];
        
        
        //initialize alphas array to all zero
        alphas = new double[N];//zero is default value
        alpha_s = new double[N];
        
        
        i_up = i_low = 0;//value chosen completly at random, I promise (any input will be fine)
        Arrays.fill(I1, true);
        
        b_up  = b_low = dataSet.getTargetValue(i_up);
        b_up  += eps;
        b_low -= eps;
        
        boolean examinAll = true;
        
        //no errors set, all zero so far..
        
        if(warmSolution != null)
        {
            /*
             * warm for regression is kinda hard, so we use it to set the initial focus on a few data poitns. We set the alpha values to be non zero for the points that are errors and let everything else past the "margin" (ie: in the cone of the espilon) be zero. Then the first few passes of SMO will optimize this intial set, and then when examineAll becomes true larger corrections can be made. 
             */
            examinAll = false;
            b_low = -Double.MAX_VALUE;
            b_up = Double.MAX_VALUE;
            
            for (int i = 0; i < N; i++)
            {
                double err = label[i]-warmSolution.regress(dataSet.getDataPoint(i));
                if(Math.abs(err) < epsilon)
                    err = 0;
                else 
                    err -= Math.signum(err)*epsilon;
//                err = signum(err)*min(abs(err), 1);
                
                double C_i = C*weights.get(i);
                alphas[i] = fuzzyClamp(err, C_i, 1e-6);
                alpha_s[i] = fuzzyClamp(-err, C_i, 1e-6);
            }
            
            
            for (int i = 0; i < N; i++)
            {
                //fix F cache and set assignment
                final double C_i = C*weights.get(i);
                final double F_i = fcache[i] = label[i]-decisionFunctionR(i);
                updateSetR(i, C_i);
                //fix up the bounds on bias term, see eq (3) in "Improvements to the SMO algorithm for SVM regression."
                if(I0[i] || I1[i] || I3[i])
                    b_up = min(b_up, F_i);
                if(I0[i] || I1[i] || I2[i])
                    b_low = max(b_low, F_i);
            }
            
            b_up-=epsilon;
            b_low+=epsilon;
            
        }

        int numChanged = 0;
        
        int examinAllCount = 0;
        int iter = 0;
        while( (examinAll || numChanged > 0) && iter < maxIterations )
        {
            iter++;
            numChanged = 0;
            if (examinAll)
            {
                //loop I over all training examples
                for (int i = 0; i < N; i++)
                    numChanged += examineExampleR(i);
                examinAllCount++;
            }
            else
            {
                if(modificationOne)
                {
                    for(int i = 0; i < I0.length; i++)
                    {
                        if(!I0[i])
                            continue;
                        numChanged += examineExampleR(i);

                        if (b_up > b_low - 2*tolerance)
                        {
                            numChanged = 0;
                            break;
                        }
                    }
                }
                else//modification 2
                {
                    boolean inner_loop_success = true;
                    do
                    {
                        if(inner_loop_success == takeStepR(i_up, i_low))
                            numChanged++;
                    }
                    while(inner_loop_success && b_up < b_low-2*tolerance);
                    numChanged = 0;
                }
            }

            if (examinAll)
                examinAll = false;
            else if (numChanged == 0)
                examinAll = true;
        }
        
        b = (b_up+b_low)/2;

        //SVMs are usualy sparse, we dont need to keep all the original vectors!
        int supportVectorCount = 0;
        for(int i = 0; i < N; i++)
            if(alphas[i] != 0 || alpha_s[i] != 0)//Its a support vector
            {
                ListUtils.swap(vecs, supportVectorCount, i);
                alphas[supportVectorCount++] = alphas[i]-alpha_s[i];
            }

        vecs = new ArrayList<Vec>(vecs.subList(0, supportVectorCount));
        alphas = Arrays.copyOfRange(alphas, 0, supportVectorCount);
        label = null;
        
        fcache = null;
        I0 = I0_a = I0_b = I1 = I2 = I3 = I4 = null;
        
        setCacheMode(null);
        setAlphas(alphas);
    }

    @Override
    public boolean warmFromSameDataOnly()
    {
        return false;
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
        return new LogUniform(1e-1, 100);
    }
}
