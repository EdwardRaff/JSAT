package jsat.classifiers.linear;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.*;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import static java.lang.Math.*;
import java.util.*;
import jsat.DataSet;
import jsat.SimpleWeightVectorModel;
import jsat.SingleWeightVectorModel;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.Uniform;
import jsat.exceptions.FailedToFitException;
import jsat.linear.*;
import jsat.lossfunctions.LogisticLoss;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.WarmParameter;
import jsat.parameters.Parameterized;

/**
 * NewGLMNET is a batch method for solving Elastic Net regularized Logistic
 * Regression problems of the form <br>
 * 0.5 * (1-&alpha;) ||w||<sub>2</sub> + &alpha; * ||w||<sub>1</sub> + C * 
 * <big>&sum;</big><sup>N</sup><sub>i=1</sub> &#8467; (w<sup>T</sup> x<sub>i</sub> + b, y<sub>i</sub>). 
 * <br>
 * <br>
 * For &alpha; = 1, this becomes pure Lasso / L<sub>1</sub> regularized Logistic
 * Regression. For &alpha; = 0, this becomes pure Ridge/ L<sub>2</sub> regularized
 * Logistic Regression, however better solvers such as 
 * {@link LogisticRegressionDCD} are faster if using &alpha; = 0. <br>
 * The default behavior is to use &alpha;=1, and includes the bias term. 
 * Including the bias term can take longer to train, but can also increase
 * sparsity for some problems. <br>
 * <br>
 * This algorithm can be warm started from any classifier implementing the
 * {@link SingleWeightVectorModel} interface.
 * <br>
 * <br>
 * See:
 * <ul>
 * <li>Yuan, G., Ho, C.-H.,&amp;Lin, C. (2012). <i>An improved GLMNET for 
 * L1-regularized logistic regression</i>. Journal of Machine Learning Research,
 * 13, 1999–2030. doi:10.1145/2020408.2020421</li>
 * <li>King, R., Morgan, B. J. T., Gimenez, O., Brooks, S. P., Crc, H.,&amp;Raton,
 * B. (2010). <i>Regularization Paths for Generalized Linear Models via 
 * Coordinate Descent</i>. Journal of Statistical Software, 36(1), 1–22.</li>
 * <li>Zou, H.,&amp;Hastie, T. (2005). <i>Regularization and variable selection 
 * via the elastic net</i>. Journal of the Royal Statistical Society, Series B, 
 * 67(2), 301–320. doi:10.1111/j.1467-9868.2005.00503.x</li>
 * </ul>
 * 
 * @author Edward Raff
 */
public class NewGLMNET implements WarmClassifier, Parameterized, SingleWeightVectorModel
{

    private static final long serialVersionUID = 4133368677783573518L;
    //TODO make these other fields configurable as well
    private static final double DEFAULT_BETA = 0.5;
    private static final double DEFAULT_V = 1e-12;
    private static final double DEFAULT_GAMMA = 0;
    private static final double DEFAULT_SIGMA = 0.01;
    /**
     * The default tolerance for training is {@value #DEFAULT_EPS}. 
     */
    public static final double DEFAULT_EPS = 1e-2;
    /**
     * The default number of outer iterations of the training algorithm is 
     * {@value #DEFAULT_MAX_OUTER_ITER} . 
     */
    public static final int DEFAULT_MAX_OUTER_ITER = 100;
    
    /**
     * Weight vector
     */
    private Vec w;
    /**
     * Bias term
     */
    private double b;
    private double beta = DEFAULT_BETA;
    private double v = DEFAULT_V;
    private double gamma = DEFAULT_GAMMA;
    private double sigma = DEFAULT_SIGMA;
    private double C;
    private double alpha;
    private int maxOuterIters = DEFAULT_MAX_OUTER_ITER;
    private double e_out = DEFAULT_EPS;
    private boolean useBias = true;
    /**
     * The maximum allowed line-search steps
     */
    private int maxLineSearchSteps = 20;

    /**
     * Creates a new L<sub>1</sub> regularized Logistic Regression solver with 
     * C = 1. 
     */
    public NewGLMNET()
    {
        this(1);
    }
    
    /**
     * Creates a new L<sub>1</sub> regularized Logistic Regression solver
     * @param C the regularization term
     */
    public NewGLMNET(double C)
    {
        this(C, 1);
    }
    
    /**
     * Creates a new Elastic Net regularized Logistic Regression solver
     * @param C the regularization term
     * @param alpha the fraction of weight (in [0, 1]) to apply to L<sub>1</sub>
     * regularization instead of L<sub>2</sub> regularization. 
     */
    public NewGLMNET(double C, double alpha)
    {
        setC(C);
        setAlpha(alpha);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    protected NewGLMNET(NewGLMNET toCopy)
    {
        if(toCopy.w !=null)
            this.w = toCopy.w.clone();
        this.b = toCopy.b;
        this.beta = toCopy.beta;
        this.v = toCopy.v;
        this.gamma = toCopy.gamma;
        this.sigma = toCopy.sigma;
        this.C = toCopy.C;
        this.e_out = toCopy.e_out;
        this.maxOuterIters = toCopy.maxOuterIters;
        this.alpha = toCopy.alpha;
        this.useBias = toCopy.useBias;
    }
        
    /**
     * Sets the regularization term, where smaller values indicate a larger 
     * regularization penalty. 
     * 
     * @param C the positive regularization term
     */
    @WarmParameter(prefLowToHigh = true)
    public void setC(double C)
    {
        if(C <= 0 || Double.isInfinite(C) || Double.isNaN(C))
            throw new IllegalArgumentException("Regularization term C must be a positive value, not " + C);
        this.C = C;
    }

    /**
     * 
     * @return the regularization term
     */
    public double getC()
    {
        return C;
    }

    /**
     * Using &alpha; = 1 corresponds to pure L<sub>1</sub> regularization, and 
     * &alpha; = 0 corresponds to pure L<sub>2</sub> regularization. Any value 
     * in-between is then an Elastic Net regularization.
     * 
     * @param alpha the value in [0, 1] for determining the regularization 
     * penalty's interpolation between pure L<sub>2</sub> and L<sub>1</sub>
     * regularization. 
     */
    public void setAlpha(double alpha)
    {
        if(alpha < 0 || alpha > 1 || Double.isNaN(alpha))
            throw new IllegalArgumentException("alpha must be in [0, 1], not " + alpha);
        this.alpha = alpha;
    }

    /***
     * 
     * @return the fraction of weight (in [0, 1]) to apply to L<sub>1</sub>
     * regularization instead of L<sub>2</sub> regularization. 
     */
    public double getAlpha()
    {
        return alpha;
    }

    /**
     * Sets the maximum number of training iterations for the algorithm, 
     * specifically the outer loop as mentioned in the original paper. 
     * {@value #DEFAULT_MAX_OUTER_ITER} is the default value used, and may need 
     * to be increased for more difficult problems. 
     * 
     * @param maxOuterIters the maximum number of outer iterations
     */
    public void setMaxIters(int maxOuterIters)
    {
        if(maxOuterIters < 1)
            throw new IllegalArgumentException("Number of training iterations must be positive, not " + maxOuterIters);
        this.maxOuterIters = maxOuterIters;
    }

    /**
     * 
     * @return the maximum number of training iterations
     */
    public int getMaxIters()
    {
        return maxOuterIters;
    }

    /**
     * Sets the tolerance parameter for convergence. Smaller values will be more
     * exact, but larger values will converge faster. The default value is 
     * fairly exact at {@value #DEFAULT_EPS}, increasing it by an order of 
     * magnitude can often be done without hurting accuracy. 
     * 
     * @param e_out the tolerance parameter. 
     */
    public void setTolerance(double e_out)
    {
        if(e_out <= 0 || Double.isNaN(e_out))
            throw new IllegalArgumentException("convergence tolerance paramter must be positive, not " + e_out);
        this.e_out = e_out;
    }
    
    /**
     * 
     * @return the convergence tolerance parameter
     */
    public double getTolerance()
    {
        return e_out;
    }

    /**
     * Controls whether or not an un-regularized bias term is added to the 
     * model. Using a bias term can increase runtime, especially in sparse data 
     * sets, as each data point will have work done for the implicit bias term. 
     * However the bias term is usually needed for small dimension problems, and 
     * can improve the sparsity of the solution for higher dimensional problems. 
     * 
     * @param useBias {@code true} if an un-regularized bias term should be used
     * or {@code false} to not use any bias term. 
     */
    public void setUseBias(boolean useBias)
    {
        this.useBias = useBias;
    }

    /**
     * 
     * @return {@code true} if an un-regularized bias term will be used
     * or {@code false} to not use any bias term. 
     */
    public boolean isUseBias()
    {
        return useBias;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        return LogisticLoss.classify(w.dot(data.getNumericalValues())+b);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }
    
    @Override
    public void trainC(ClassificationDataSet dataSet, Classifier warmSolution, ExecutorService threadPool)
    {
        trainC(dataSet, warmSolution);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, Classifier warmSolution)
    {
        if(warmSolution instanceof SimpleWeightVectorModel)
        {
            SimpleWeightVectorModel swv = (SimpleWeightVectorModel) warmSolution;
            train(dataSet, swv.getRawWeight(0), swv.getBias(0), true);
        }
        else 
            throw new FailedToFitException("Warm solution is not of a");
    }
    
    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        train(dataSet, null, 0, false);
    }
    
    private void train(ClassificationDataSet dataSet, Vec w_init, double b_init, boolean useInit)
    {
        /*
         * The original NewGLMNET paper describes the algorithm as minimizing 
         * f(w) = ||w||_1 + L(w), where L(w) is the logistic loss summed over 
         * all the variables. To make adapation to elastic net easier, we define
         * f(w) = alpha ||w||_1 + L(w), where L(w) = (1-alpha) ||w||_2 + loss sum. 
         * This way we keep all the framework for L_1 regularization and 
         * shrinking, and just update the appropriate terms where necessary. 
         */
        
        //paper uses n= #features so we will follow their lead
        final int n = dataSet.getNumNumericalVars();
        //l = # data points
        final int l = dataSet.getSampleSize();
        
        if(useInit)
        {
            w = new DenseVector(w_init);
            b = useBias ? b_init : 0;
        }
        else
        {
            w = new DenseVector(n);
            b = 0;
        }
        List<Vec> X = dataSet.getDataVectors();
        
        double first_M_bar = 0;
        double e_in = 1.0;//set later when first_M_bar is set
        
        double[] w_dot_x = new double[l];
        double[] exp_w_dot_x = new double[l];
        double[] exp_w_dot_x_plus_dx = new double[l];
        /**
         * Used in the linear search step at the end
         */
        double[] d_dot_x = new double[l];
        /**
         * Contains the value 1/(1+e^(w^T x)). This is used in computing D and the partial derivatives. 
         */
        double[] D_part = new double[l];
        double[] D = new double[l];
        
        /**
         * Stores the value H<sup>k</sup><sub>j,j</sub> computer at the start of each iteration
         */
        double[] H = new double[n];
        /**
         * Stores the value H<sup>k</sup><sub>j,j</sub> computer at the start of
         * each iteration for the bias term
         */
        double H_bias = 0;
        /**
         * Stores the value &nambla; L<sub>j</sub>
         */
        double[] delta_L = new double[n];
        /**
         * The gradient value for the bias term
         */
        double delta_L_bias = 0;
        float[] y = new float[l];
        double w_norm_1;
        double w_norm_2;
        if(useInit)
        {
            for(int i = 0; i < l; i++)
            {
                y[i] = dataSet.getDataPointCategory(i)*2-1;
                w_dot_x[i] = w.dot(X.get(i))+b;
                final double tmp = exp_w_dot_x_plus_dx[i] = exp_w_dot_x[i] = exp(w_dot_x[i]);
                final double D_part_i = D_part[i]= 1/(1+tmp);
                D[i] = tmp*D_part_i*D_part_i;
            }
            w_norm_1 = w.pNorm(1);
            w_norm_2 = w.pNorm(2);
        }
        else//w = 0
        {
            for(int i = 0; i < l; i++)
            {
                y[i] = dataSet.getDataPointCategory(i)*2-1;
                w_dot_x[i] = 0.0;
                exp_w_dot_x_plus_dx[i] = exp_w_dot_x[i] = 1.0;
                D_part[i]= 0.5;
                D[i] = 0.25;
            }
            w_norm_1 = w.pNorm(1);
            w_norm_2 = w.pNorm(2);
        }
        
        List<Vec> columnsOfX = new ArrayList<Vec>(Arrays.asList(dataSet.getNumericColumns()));
        /**
         * sum of all x_j values in the negative class. Used for ∇_j L in trick
         * from LIBLINEAR eq(44)
         */
        double[] col_neg_class_sum = new double[n];
        for(int j = 0; j < n; j++)
        {
            Vec vec = columnsOfX.get(j);
            for(IndexValue iv : vec)
                if(y[iv.getIndex()] == -1)
                    col_neg_class_sum[j] += iv.getValue();
        }
        
        /**
         * Sum of all x_j values in the negative class for the bias term. 
         */
        double col_neg_class_sum_bias = 0;
        if(useBias)
        {
            for(int i = 0; i < l; i++)
                if(y[i] == -1)
                    col_neg_class_sum_bias++;
        }
                
        /**
         * weight for L_1 reg is alpha, so this will be the L_2 weight (1-alpha)
         */
        final double l2w = (1-alpha);
        
//        {
//            double objVal = 0;
//            for(int i = 0; i < l; i++)
//                objVal += C*log(1+exp(-y[i]*w_dot_x[i]));
//            objVal += alpha*w_norm_1 + l2w*w_norm_2;
//            System.out.println("Start Obj Val: " + objVal);
//        }
        
        //algo 3
        
        //Let M^out ← ∞
        double M_out = Double.POSITIVE_INFINITY;
        
        Vec d = new DenseVector(n);
        double d_bias = 0;
        boolean prevLineSearchFail = false;
        for(int k = 0; k < maxOuterIters; k++)//For k = 1, 2, 3, . . .
        {
            //algo 3, Step 1.
            IntList J = new IntList(n);
            ListUtils.addRange(J, 0, n, 1);
            double M = 0;
            double M_bar = 0;
            //algo 3, Step 2. 
            Iterator<Integer> j_iter = J.iterator();
            while(j_iter.hasNext())
            {
                int j = j_iter.next();
                double w_j = w.get(j);
                
                //2.1. Calculate H^k_{jj}, ∇_j L(w^k) and ∇^S_j f(w^k)
                double delta_j_L = 0;
                double deltaSqrd_L = 0;
                
                for(IndexValue x_i : columnsOfX.get(j))
                {
                    int i = x_i.getIndex();
                    double val = x_i.getValue();
                    
                    delta_j_L += -val*D_part[i];
                    //eq(44) from LIBLINEAR paper , re-factored to avoid a division by using D_part
                    deltaSqrd_L += val*val*D[i];
                }
                delta_L[j] = delta_j_L = l2w*w_j + C*(delta_j_L + col_neg_class_sum[j]);
                //H^k from eq (19)
                /*
                 * regular is C X^T D X, L2 just adds + I , but we are alreayd 
                 * doing + eps * I to make sure the gradient is there. So just 
                 * do the max of v and lambda_2
                 */
                H[j] = C*deltaSqrd_L + max(v, l2w);

                double deltaS_j_fw;
                if(w_j > 0)
                    deltaS_j_fw = delta_j_L+alpha;
                else if(w_j < 0)
                    deltaS_j_fw = delta_j_L-alpha;
                else//w_j = 0
                    deltaS_j_fw = signum(delta_j_L)*max(abs(delta_j_L)-alpha, 0);
                //done with step 2, we have all the info
                
                //2.2. If w^k_j = 0 and |∇_j L(w^k)| < 1−M^out/l   // outer-level shrinking
                //then J ←J\{j}.
                //else M ←max(M, |∇^S_j f(w^k)|) and M_bar ← M_bar +|∇^S_j f(w^k)|
                
                if(w_j == 0 && abs(delta_j_L) < alpha-M_out/l)
                    j_iter.remove();
                else
                {
                    M = max(M, abs(deltaS_j_fw));
                    M_bar += abs(deltaS_j_fw);
                }
            }
            
            if(useBias)
            {
                //2.1. Calculate H^k_{jj}, ∇_j L(w^k) and ∇^S_j f(w^k)
                double delta_j_L = 0;
                double deltaSqrd_L = 0;
                
                for(int i = 0; i < l ; i++)//all have an implicit bias term
                {
                    delta_j_L += -D_part[i];
                    //eq(44) from LIBLINEAR paper , re-factored to avoid a division by using D_part
                    deltaSqrd_L += D[i];
                }
                delta_L_bias = delta_j_L = C*(delta_j_L + col_neg_class_sum_bias);
                //H^k from eq (19) , but dont need v * I since its the bias term
                H_bias = C*deltaSqrd_L + v;

                double deltaS_j_fw = delta_L_bias;
                M = max(M, abs(deltaS_j_fw));
                M_bar += abs(deltaS_j_fw);
            }
            
            if (k == 0)//first run
                if (useInit)//we have some value of W already, 
                    e_in = first_M_bar = getM_Bar_for_w0(n, l, columnsOfX, col_neg_class_sum, col_neg_class_sum_bias);
                else//normal algo
                    e_in = first_M_bar = M_bar;
            //algo 3, Step 3. 3. If M_bar ≤ eps_out ,  return w^k 
            
            if(M_bar <= e_out*first_M_bar)
                break;
            //algo 3, Step 4. Let M_out ←M
            M_out = M;
            
            //algo 3, Step 5. Run algo 4
            //START: Algorithm 4 Inner iterations of NewGLMNET with shrinking
            double M_in = Double.POSITIVE_INFINITY;
            IntList T = new IntList(J);

            d.zeroOut();
            d_bias = 0;
            
            /**
             * Sometimes we see the |z| be very small over and over, so we stop 
             * if we see it too many times in a row (which means we really 
             * aren't making much progress)
             */
            int smallZInARow = 0;
            
            for(int p = 0; p < 1000; p++)// inner iterations
            {
                //step 1.
                double m = 0, m_bar = 0;
                /**
                 * Used to check if we aren't really making any progress
                 */
                double max_abs_z = 0;
                Collections.shuffle(T);
                Iterator<Integer> T_iter = T.iterator();
                final double dynRange = n*5.0/T.size();//used for dynamic clip, see below
                while(T_iter.hasNext())//step 2.
                {
                    final int j = T_iter.next();
                    final double w_j = w.get(j);
                    final double d_j = d.get(j);
                    //from eq(16)
                    //∇_j q^bar_k(d) = ∇_j L(w^k) + (∇^2 L(w^k) d)_j
                    //∇^2_jj q^bar_k(d)=∇^2_{jj} L(w^k)
                    
                    double delta_qBar_j = 0;
                    //first compute the (∇^2 L(w^k) d)_j portion
                    //see after algo 2 before eq (17)
                    for(IndexValue iv : columnsOfX.get(j))
                    {
                        int i = iv.getIndex();
                        delta_qBar_j += iv.getValue()*D[i]*d_dot_x[i];
                    }
                    delta_qBar_j *= C;
                    
                    //now add the part we know from before
                    delta_qBar_j += delta_L[j];
                    /*
                     * For L_2, use (A+B)C = AC + BC to modify ((lambda_2 * I + ∇_2 L(w))d)j
                     * so we need to add lambda_2 * I d^{p, j}_j to the final 
                     * value. I * x = x, and we are taking the value of the j'th
                     * coordinate, so we just have to add lambda_2 d_j
                     */
                    delta_qBar_j += l2w*d_j;
                    
                    double deltaS_q_k_j;
                    if(w_j + d_j > 0)
                        deltaS_q_k_j = delta_qBar_j + alpha;
                    else if(w_j + d_j < 0)
                        deltaS_q_k_j = delta_qBar_j - alpha;
                    else //w_j + d_j == 0
                        deltaS_q_k_j = signum(delta_qBar_j)*max(abs(delta_qBar_j)-alpha, 0);
                    
                    double deltaSqrd_q_jj = H[j];
                    
                    if(w_j + d_j == 0 && abs(delta_qBar_j) < alpha - M_in/l)
                    {
                        T_iter.remove();//inner-level shrinking
                    }
                    else
                    {
                        m = max(m, abs(deltaS_q_k_j));
                        m_bar += abs(deltaS_q_k_j);
                        double z;
                        //find z by eq (9), our w_j is actuall w_j+d_j
                        
                        if(delta_qBar_j+alpha <= deltaSqrd_q_jj*(w_j+d_j))
                            z = -(delta_qBar_j+alpha)/deltaSqrd_q_jj;
                        else if(delta_qBar_j-alpha >= deltaSqrd_q_jj*(w_j+d_j))
                            z = -(delta_qBar_j-alpha)/deltaSqrd_q_jj;
                        else
                            z = -(w_j+d_j);
                        
                        if(abs(z) < 1e-11)
                            continue;
                        
                        /*
                         * When everyone is active, clip the updates to a 
                         * smaller range - as we are going to have a lot of 
                         * changes going on and this might make steps far larger
                         * than it should.  When there are fewer active 
                         * dimensions, allow for more change
                         */
                        z = min(max(z,-dynRange),dynRange);
                        
                        max_abs_z = max(max_abs_z, abs(z));
                        
                        d.increment(j, z);
                        
                        //book keeping, see eq(17)
                        for(IndexValue iv : columnsOfX.get(j))
                            d_dot_x[iv.getIndex()] += z*iv.getValue();
                    }
                }
                
                if(useBias)
                {
                    //from eq(16)
                    //∇_j q^bar_k(d) = ∇_j L(w^k) + (∇^2 L(w^k) d)_j
                    //∇^2_jj q^bar_k(d)=∇^2_{jj} L(w^k)
                    
                    double delta_qBar_j = 0;
                    //first compute the (∇^2 L(w^k) d)_j portion
                    //see after algo 2 before eq (17)
                    for(int i = 0; i < l; i++)
                        delta_qBar_j += 1*D[i]*d_dot_x[i];//compiler will take out 1*, left just to remind us its the bias term
                    delta_qBar_j *= C;
                    
                    //now add the part we know from before
                    delta_qBar_j += delta_L_bias;
                    
                    double deltaS_q_k_j = delta_qBar_j;
                    
                    double deltaSqrd_q_jj = H_bias;
                    
                    m = max(m, abs(deltaS_q_k_j));
                    m_bar += abs(deltaS_q_k_j);
                    
                    double z = -delta_qBar_j/(deltaSqrd_q_jj);

                    if (abs(z) > 1e-11)
                    {
                        z = min(max(z, -dynRange), dynRange);
                        
                        max_abs_z = max(max_abs_z, abs(z));

                        d_bias += z;

                        //book keeping, see eq(17)
                        for(int i = 0; i < l ; i++)
                            d_dot_x[i] += z;
                    }
                }
                
                boolean breakInnerLoopAnyway = false;
                
                if(max_abs_z == 0)
                    breakInnerLoopAnyway = true;
                else if (max_abs_z <= 1e-6)
                {
                    if(smallZInARow++ >= 3)//give it a few chances
                        breakInnerLoopAnyway = true;
                }
                else if(max_abs_z <= 1e-3)
                {
                    if(smallZInARow++ >= 30)//give it a lot chances
                        breakInnerLoopAnyway = true;
                }
                else
                    smallZInARow = 0;//reset, we are making progress!
                
                //step 3. 
                if(m_bar <= e_in || breakInnerLoopAnyway)
                {
                    
                    if(T.size() == J.size())
                    {
                        /*
                         * If at one outer iteration, the condition (26) holds
                         * after only one cycle of n CD steps, then we reduce 
                         * e_in by 1/4. 
                         * That is, the program automatically adjusts e_in if it
                         * finds that too few CD steps are conducted for 
                         * minimizing qk(d)
                         */
                        if(p == 0)
                            e_in /= 4;
                        break;
                    }
                    else
                    {
                        T.clear();
                        T.addAll(J);
                        M_in = Double.POSITIVE_INFINITY;
                    }
                }
                else
                    M_in = m;
                
            }
            //END: Algorithm 4 Inner iterations of NewGLMNET with shrinking
            
            //algo 3, Step 6. Compute λ = max{1,β,β^2, . . . } such that λd satisfies (20)
            //Use the form of eq(45) from Aug2014 LIBLINEAR paper
            
            //get ||w+d||_1 and ∇L^T d in one loop together
            double wPd_norm_1 = w_norm_1;
            double wPd_norm_2 = w_norm_2;
            double delta_L_dot_d = 0;
            
            for(IndexValue iv: d)
            {
                final int j = iv.getIndex();
                final double w_j = w.get(j);
                final double d_j = iv.getValue();
                wPd_norm_1 -= abs(w_j);
                wPd_norm_1 += abs(w_j+d_j);
                wPd_norm_2 -= w_j*w_j;
                wPd_norm_2 += (w_j+d_j)*(w_j+d_j);
                delta_L_dot_d += d_j*delta_L[j];
            }
            
            delta_L_dot_d += d_bias*delta_L_bias;
            
            final double breakCondition = sigma*(delta_L_dot_d + 
                    alpha*(wPd_norm_1-w_norm_1) + 
                    l2w*(wPd_norm_2-w_norm_2)  );
            
            double lambda = 1;
            int t = 0;
            double wPlambda_d_norm_1 = wPd_norm_1;
            double wPlambda_d_norm_2 = wPd_norm_2;
            while(t < maxLineSearchSteps)//we may want to adjust this as beta changes
            {
                //"For line search, we use the following form of the sufficient decrease condition" eq(45) from LIBLINEAR paper Aug 2014
                double newTerm = 0;
                for(int i = 0; i < l; i++)
                {
                    double exp_lamda_d_dot_x = exp(lambda*d_dot_x[i]);
                    exp_w_dot_x_plus_dx[i] = exp_w_dot_x[i]*exp_lamda_d_dot_x;
                    newTerm += log((exp_w_dot_x_plus_dx[i]+1)/(exp_w_dot_x_plus_dx[i]+exp_lamda_d_dot_x  ));
                    if(y[i] == -1)
                        newTerm += lambda*d_dot_x[i];
                }
                
                newTerm = l2w*(wPlambda_d_norm_2 - w_norm_2) +//l2 reg
                        alpha*(wPlambda_d_norm_1 - w_norm_1) + //l1 reg
                        C*newTerm;//loss
                if(newTerm <= lambda * breakCondition)
                    break;
                //else
                lambda = pow(beta, ++t);
                //update norm 
                wPlambda_d_norm_1 = w_norm_1;
                wPlambda_d_norm_2 = w_norm_2;
                for(IndexValue iv: d)
                {
                    final double w_j = w.get(iv.getIndex());
                    final double lambda_d_j = lambda*iv.getValue();
                    wPlambda_d_norm_1 -= abs(w_j);
                    wPlambda_d_norm_1 += abs(w_j+lambda_d_j);
                    wPlambda_d_norm_2 -= w_j*w_j;
                    wPlambda_d_norm_2 += (w_j+lambda_d_j)*(w_j+lambda_d_j);
                }
            }

            //if line search fails twice in a row, just quit
            if(t == maxLineSearchSteps)//this shouldn't happen unless we are having serious trouble improving our results
                if (prevLineSearchFail)
                    break;//jsut finish. 
                else
                    prevLineSearchFail = true;
            else
                prevLineSearchFail = false;

            //algo 3, Step 7. 7. w^{k+1} = w^k +λ d.
            w.mutableAdd(lambda, d);
            b += lambda * d_bias;
            w_norm_1 = wPlambda_d_norm_1;
            w_norm_2 = wPlambda_d_norm_2;
            //and more book keeping
            //val from last line search is new w
            System.arraycopy(exp_w_dot_x_plus_dx, 0, exp_w_dot_x, 0, l);
            //(w+lambda d)^T  x = w^T x + d^T x
            for(int i = 0; i < l; i++)
            {
                w_dot_x[i] += lambda*d_dot_x[i];
                final double D_part_i = D_part[i]= 1/(1+exp_w_dot_x[i]);
                D[i] = exp_w_dot_x[i]*D_part_i*D_part_i;
            }
            Arrays.fill(d_dot_x, 0.0);//new d = 0, always
            
//            double objVal = 0;
//            for(int i = 0; i < l; i++)
//                objVal += C*log(1+exp(-y[i]*w_dot_x[i]));
//            objVal += alpha*w_norm_1 + l2w*w_norm_2;
//            System.out.println("Iter "+ k + " has New Obj Val: " + objVal);
            
        }
    }

    /**
     * When we perform a warm start, we want to train to the same point that we
     * would have if we had not done a warm start. But our stopping point is
     * based on the initial relative error. To get around that, this method
     * computes what the error would have been for the zero weight vector
     * @param n
     * @param l
     * @param columnsOfX
     * @param col_neg_class_sum
     * @param col_neg_class_sum_bias
     * @return the error for M_bar that would have been computed if we were using the zero weight vector
     */
    private double getM_Bar_for_w0(int n, int l, List<Vec> columnsOfX, double[] col_neg_class_sum, double col_neg_class_sum_bias)
    {
        /**
         * if w=0, then D_part[i] = 0.5 for all i
         */
        final double D_part_i = 0.5;
        
        
        //algo 3, Step 1.
        double M_bar = 0;
        //algo 3, Step 2. 
        for(int j = 0; j < n; j++)
        {
            final double w_j = 0;

            //2.1. Calculate H^k_{jj}, ∇_j L(w^k) and ∇^S_j f(w^k)
            double delta_j_L = -columnsOfX.get(j).sum()*0.5;

            delta_j_L = /* (l2w * w_j) not needed b/c w_j=0*/ + C * (delta_j_L + col_neg_class_sum[j]);

            double deltaS_j_fw;
            //only the w_j = 0 case applies, b/c that is what this method is for!             
            //w_j = 0
            deltaS_j_fw = signum(delta_j_L) * max(abs(delta_j_L) - alpha, 0);
            //done with step 2, we have all the info
            M_bar += abs(deltaS_j_fw);

        }

        if (useBias)
        {
            //2.1. Calculate H^k_{jj}, ∇_j L(w^k) and ∇^S_j f(w^k)
            double delta_j_L = 0;

            for (int i = 0; i < l; i++)//all have an implicit bias term
                delta_j_L += -D_part_i;
            delta_j_L = C * (delta_j_L + col_neg_class_sum_bias);
            
            double deltaS_j_fw = delta_j_L;
            
            M_bar += abs(deltaS_j_fw);
        }
        return M_bar;
    }
    
    
    
    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public NewGLMNET clone()
    {
        return new NewGLMNET(this);
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
    public Vec getRawWeight()
    {
        return w;
    }

    @Override
    public double getBias()
    {
        return b;
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
        if(index < 1)
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
    public boolean warmFromSameDataOnly()
    {
        return false;
    }
    
    /**
     * Guess the distribution to use for the trade off term term
     * {@link #setAlpha(double) (double) &alpha;} in Elastic Net regularization.
     *
     * @param d the data set to get the guess for
     * @return the guess for the &alpha; parameter 
     */
    public static Distribution guessAlpha(DataSet d)
    {
        //Would do [0, .75], but if you are doing to be so close to full L2 reg you should really be using a different solver
        return new Uniform(0.25, 0.75);
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
        double maxLambda = LinearTools.maxLambdaLogisticL1((ClassificationDataSet) d);
        double minC = 1/(2*maxLambda*d.getSampleSize());
        return new LogUniform(minC*10, minC*1000);
    }
}
