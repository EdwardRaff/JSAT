
package jsat.classifiers.svm;

import static java.lang.Math.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.*;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.Vec;
import jsat.parameters.*;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.IntSetFixedSize;
import jsat.utils.ListUtils;

/**
 * An implementation of SVMs using Platt's Sequential Minimum Optimization (SMO) 
 * forboth Classification and Regression. 
 * <br><br>
 * See:<br>
 * <ul>
 * <li>Platt, J. C. (1998). <i>Sequential Minimal Optimization: A Fast Algorithm
 * for Training Support Vector Machines</i>. Advances in kernel methods 
 * (pp. 185 – 208). Retrieved from <a href="http://www.bradblock.com/Sequential_Minimal_Optimization_A_Fast_Algorithm_for_Training_Support_Vector_Machine.pdf">here</a></li>
 * <li>Keerthi, S. S., Shevade, S. K., Bhattacharyya, C., & Murthy, K. R. K. 
 * (2001). <i>Improvements to Platt’s SMO Algorithm for SVM Classifier Design
 * </i>. Neural Computation, 13(3), 637–649. doi:10.1162/089976601300014493</li>
 * <li>Smola, A. J., & Schölkopf, B. (2004). <i>A tutorial on support vector 
 * regression</i>. Statistics and Computing, 14(3), 199–222. 
 * doi:10.1023/B:STCO.0000035301.49549.88</li>
 * <li>Shevade, S. K., Keerthi, S. S., Bhattacharyya, C., & Murthy, K. K. (1999)
 * . <i>Improvements to the SMO algorithm for SVM regression</i>. Control D
 * ivision, Dept. of Mechanical Engineering CD-99–16. Control Division, Dept. of
 * Mechanical Engineering. doi:10.1109/72.870050</li>
 * <li>Shevade, S. K., Keerthi, S. S., Bhattacharyya, C., & Murthy, K. K. (2000)
 * . <i>Improvements to the SMO algorithm for SVM regression</i>. IEEE 
 * transactions on neural networks / a publication of the IEEE Neural Networks 
 * Council, 11(5), 1188–93. doi:10.1109/72.870050</li>
 * </ul>
 * 
 * @author Edward Raff
 */
public class PlatSMO extends SupportVectorLearner implements BinaryScoreClassifier, Regressor, Parameterized 
{
    /**
     * Bias
     */
    protected double b = 0, b_low, b_up;
    private double C = 1;
    private double tolerance = 1e-4;
    private double eps = 1e-3;
    private double epsilon = 1e-3;

    private int maxIterations = 10000;
    private boolean modificationOne = false;
    
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
     * i : 0 < a_i < C
     * <br>
     * For regression this contains both of I0_a and I0_b
     */
    private Set<Integer> I0;
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
     * Creates a new SVM object that uses no cache mode. 
     * 
     * @param kf the kernel trick to use
     */
    public PlatSMO(KernelTrick kf)
    {
        super(kf, SupportVectorLearner.CacheMode.NONE);
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(vecs == null)
            throw new UntrainedModelException("Classifier has yet to be trained");
        
        double sum = 0;
        CategoricalResults cr = new CategoricalResults(2);
        
        sum = kEvalSum(data.getNumericalValues());

        if(sum > b)
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

        final int N = dataSet.getSampleSize();
        vecs = new ArrayList<Vec>(N);
        label = new double[N];
        b = 0;
        for(int i = 0; i < N; i++)
        {
            DataPoint dataPoint = dataSet.getDataPoint(i);
            vecs.add(dataPoint.getNumericalValues());
            if(dataSet.getDataPointCategory(i) == 0)
                label[i] = -1;
            else
                label[i] = 1;
        }
        
        setCacheMode(getCacheMode());//Initiates the cahce
        
        I0 = new IntSetFixedSize(N);
        I1 = new boolean[N];
        I2 = new boolean[N];
        I3 = new boolean[N];
        I4 = new boolean[N];
        
        
        //initialize alphas array to all zero
        alphas = new double[N];//zero is default value
        fcache = new double[N];
        
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
                for (int i = 0; i < N; i++)
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
            alphas[i] *= label[i];
        
        int supportVectorCount = 0;
        for(int i = 0; i < N; i++)
            if(alphas[i] > 0 || alphas[i] < 0)//Its a support vector
            {
                ListUtils.swap(vecs, supportVectorCount, i);
                alphas[supportVectorCount++] = alphas[i];
            }

        vecs = new ArrayList<Vec>(vecs.subList(0, supportVectorCount));
        alphas = Arrays.copyOfRange(alphas, 0, supportVectorCount);
        label = null;
        
        fcache = null;
        I0 = null;
        I1 = I2 = I3 = I4 = null;
        
        setCacheMode(null);
        setAlphas(alphas);
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
    
    private double fuzzyClamp(double val, double max)
    {
        final double fuz = max*1e-8;
        if(val > max-fuz)
            return max;
        if(val < fuz)
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
        if(I0_a[i] || I0_b[i])
            I0.add(i);
        else
            I0.remove(i);
        I1[i] = a_i == 0 && as_i == 0;
        I2[i] = a_i == 0 && as_i == C;
        I3[i] = a_i == C && as_i == 0;
    }

    /**
     * Updates the index sets 
     * @param i1 the index to update for
     * @param a1 the alphas value for the index
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
        double alpha1 = alphas[i1], alpha2 = alphas[i2];
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
        double k12 = kEval(i2, i1);
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

        if(a2 < 1e-8)
            a2 = 0;
        else if (a2 > C - 1e-8)
            a2 = C;

        if(abs(a2 - alpha2) < eps*(a2+alpha2+eps))
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
        
        if(abs(newF1C-fcache[i1]) < 1e-15 && abs(newF2C-fcache[i2]) < 1e-15)
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
                L = max(0, gamma-C);
                H = min(C, gamma);
                if(L < H)
                {
                    double a2 = max(L, min(alpha2 - deltaPhi/eta, H));
                    a2 = fuzzyClamp(a2, C);
                    double a1 = alpha1 - (a2 - alpha2);
                    a1 = fuzzyClamp(a1, C);
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
                H = min(C, -gamma+C);
                if(L < H)
                {
                    double a2 = max(L, min(alpha2_S + (deltaPhi-2*epsilon)/eta, H));
                    a2 = fuzzyClamp(a2, C);
                    double a1 = alpha1 + (a2 - alpha2_S);
                    a1 = fuzzyClamp(a1, C);
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
                H = min(C, C+gamma);
                if(L < H)
                {
                    double a2 = max(L, min(alpha2 - (deltaPhi+2*epsilon)/eta, H));
                    a2 = fuzzyClamp(a2, C);
                    double a1 = alpha1_S + (a2 - alpha2);
                    a1 = fuzzyClamp(a1, C);
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
                L = max(0, -gamma-C);
                H = min(C, -gamma);
                if(L < H)
                {
                    double a2 = max(L, min(alpha2_S + deltaPhi/eta, H));
                    a2 = fuzzyClamp(a2, C);
                    double a1 = alpha1_S - (a2 - alpha2_S);
                    a1 = fuzzyClamp(a1, C);
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
        for(int i : I0)
            if(i != i1 && i != i2)
                fcache[i] -= ceof1 * kEval(i1, i) + ceof2 * kEval(i2, i);
        fcache[i1] -= ceof1 * k11 + ceof2 * k12;
        fcache[i2] -= ceof1 * k12 + ceof2 * k22;
        updateSetR(i1, C);//add weight data here later, thats why we pass C 
        updateSetR(i2, C);
        
        //Update threshold to reflect change in Lagrange multipliers Update
        b_low = Double.NEGATIVE_INFINITY;
        b_up = Double.POSITIVE_INFINITY;
        i_low = -1;
        i_up = -1;
        
        for (int i : I0)
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
    
    private int examineExampleR(int i2)
    {
        //y2 = target[i2]
        double y2 = label[i2];
        
        double F2;
        if(I0.contains(i2))
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
    public PlatSMO clone()
    {
        PlatSMO copy = new PlatSMO(this.getKernel().clone());
        
        copy.C = this.C;
        if(this.alphas != null)
            copy.alphas = Arrays.copyOf(this.alphas, this.alphas.length);
        if(this.alpha_s != null)
            copy.alpha_s = Arrays.copyOf(this.alpha_s, this.alpha_s.length);
        copy.b = this.b;
        copy.eps = this.eps;
        copy.epsilon = this.epsilon;
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
    public void train(RegressionDataSet dataSet)
    {
        final int N = dataSet.getSampleSize();
        vecs = new ArrayList<Vec>(N);
        label = new double[N];
        fcache = new double[N];
        b = 0;
        for(int i = 0; i < N; i++)
        {
            DataPoint dataPoint = dataSet.getDataPoint(i);
            vecs.add(dataPoint.getNumericalValues());
            fcache[i] = label[i] = dataSet.getTargetValue(i);
        }
        
        setCacheMode(getCacheMode());//Initiates the cahce
        
        I0 = new IntSetFixedSize(N);
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
        
        //no errors set, all zero so far..

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
                for (int i = 0; i < N; i++)
                    numChanged += examineExampleR(i);
                examinAllCount++;
            }
            else
            {
                if(modificationOne)
                {
                    for (int i : I0)
                    {
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
        I0 = null;
        I0_a = I0_b = I1 = I2 = I3 = I4 = null;
        
        setCacheMode(null);
        setAlphas(alphas);
    }

}
