package jsat.distributions.kernels;

import static java.lang.Math.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import jsat.classifiers.linear.kernelized.Projectron;
import jsat.linear.*;
import jsat.math.FastMath;
import jsat.math.Function;
import jsat.math.FunctionBase;
import jsat.math.optimization.GoldenSearch;
import jsat.regression.KernelRLS;
import jsat.utils.DoubleList;
import jsat.utils.ListUtils;
import jsat.utils.random.RandomUtil;

/**
 * The Kernel Point represents a kernelized weight vector by a linear 
 * combination of vectors transformed through a 
 * {@link KernelTrick kernel fuctiion}. This implementation allows the selection
 * of multiple different budget maintenance strategies <br>
 * <br>
 * See {@link KernelRLS} and {@link Projectron} for methods and papers based on
 * the same ideas used to create this class. <br>
 * Credit goes to Davis King of the <a href="http://dlib.net/ml.html">dlib 
 * library</a> for the idea of this type of class. <br>
 * <br>
 * Changing the 
 * {@link #setBudgetStrategy(jsat.distributions.kernels.KernelPoint.BudgetStrategy) 
 * budget maintinance method} or other parameters should be done <i>before</i>
 * adding any data points to the KernelPoint. <br>
 * If a maximum budget is specified, it may always be increased - but may not be
 * decreased. 
 * 
 * @author Edward Raff
 */
public class KernelPoint
{
    protected KernelTrick k;
    private double errorTolerance;
    
    protected List<Vec> vecs;
    protected List<Double> kernelAccel;
    protected Matrix K;
    protected Matrix InvK;
    protected Matrix KExpanded;
    protected Matrix InvKExpanded;
    protected DoubleList alpha;
    protected BudgetStrategy budgetStrategy = BudgetStrategy.PROJECTION;
    protected int maxBudget = Integer.MAX_VALUE;
    
    /**
     * These enums control the method used to reduce the size of the support
     * vector set in the kernel point. 
     */
    public enum BudgetStrategy
    {
        /**
         * The budget is maintained by projecting the incoming vector onto 
         * the set of current vectors. If the error in the projection is less 
         * than {@link #setErrorTolerance(double) } the projection is used, and
         * the input is added to the support vector set if the error was too 
         * large. <br>
         * Once the maximum budget size is reached, the projection is used 
         * regardless of the error of the projection. <br>
         * <br>
         * The time complexity of each update is <i>O(B<sup>2</sup>)</i> and 
         * uses <i>O(B<sup>2</sup>)</i> memory. 
         * 
         */
        PROJECTION,
        /**
         * The budget is maintained by merging two support vectors to minimize 
         * the error in the squared norm. The merged support vector is not a 
         * member of the training set. <b>This method is only valid for the 
         * {@link RBFKernel} </b>. Using any other kernel may cause invalid 
         * results<br>
         * <br>
         * See:<br>
         * <ul>
         * <li>Wang, Z., Crammer, K.,&amp;Vucetic, S. (2012). <i>Breaking the 
         * Curse of Kernelization : Budgeted Stochastic Gradient Descent for 
         * Large-Scale SVM Training</i>. The Journal of Machine Learning 
         * Research, 13(1), 3103–3131.</li>
         * <li>Wang, Z., Crammer, K.,&amp;Vucetic, S. (2010). <i>Multi-class 
         * pegasos on a budget</i>. In 27th International Conference on Machine
         * Learning (pp. 1143–1150). Retrieved from 
         * <a href="http://www.ist.temple.edu/~vucetic/documents/wang10icml.pdf">
         * here</a></li>
         * </ul>
         * <br>
         * The time complexity of each update is <i>O(B)</i> and 
         * uses <i>O(B)</i> memory. 
         */
        MERGE_RBF,
        /**
         * The budget is maintained by refusing to add new data points once the
         * budget is reached. <br>
         * <br>
         * The time complexity of each update is <i>O(B)</i> and 
         * uses <i>O(B)</i> memory. 
         */
        STOP,
        /**
         * The budget is maintained by randomly dropping a previous support 
         * vector. <br>
         * <br>
         * The time complexity of each update is <i>O(B)</i> and 
         * uses <i>O(B)</i> memory. 
         */
        RANDOM,
    }
    
    //Internal structure
    private double sqrdNorm = 0;
    private boolean normGood = true;

    /**
     * Creates a new Kernel Point, which is a point in the kernel space 
     * represented by an accumulation of vectors and uses the 
     * {@link BudgetStrategy#PROJECTION} strategy with an unbounded maximum 
     * budget
     * 
     * @param k the kernel to use
     * @param errorTolerance the maximum error in [0, 1] allowed for projecting 
     * a vector instead of adding it to the basis set
     */
    public KernelPoint(KernelTrick k, double errorTolerance)
    {
        this.k = k;
        setErrorTolerance(errorTolerance);
        setBudgetStrategy(BudgetStrategy.PROJECTION);
        setMaxBudget(Integer.MAX_VALUE);
        if(k.supportsAcceleration())
            kernelAccel = new DoubleList(16);
        alpha = new DoubleList(16);
        vecs = new ArrayList<Vec>(16);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public KernelPoint(KernelPoint toCopy)
    {
        this.k = toCopy.k.clone();
        this.errorTolerance = toCopy.errorTolerance;
        if(toCopy.vecs != null)
        {
            this.vecs = new ArrayList<Vec>(toCopy.vecs.size());
            for(Vec v : toCopy.vecs)
                this.vecs.add(v.clone());
            if(toCopy.kernelAccel != null)
                this.kernelAccel = new DoubleList(toCopy.kernelAccel);
            
            this.alpha = new DoubleList(toCopy.alpha);
        }
        
        if(toCopy.KExpanded != null)
        {
            this.KExpanded = toCopy.KExpanded.clone();
            this.InvKExpanded = toCopy.InvKExpanded.clone();

            this.K = new SubMatrix(KExpanded, 0, 0, toCopy.K.rows(), toCopy.K.cols());
            this.InvK = new SubMatrix(InvKExpanded, 0, 0, toCopy.InvK.rows(), toCopy.InvK.rows());
        }
        
        this.maxBudget = toCopy.maxBudget;
        this.sqrdNorm = toCopy.sqrdNorm;
        this.normGood = toCopy.normGood;
    }

    /**
     * Sets the maximum budget for support vectors to allow. Setting to 
     * {@link Integer#MAX_VALUE} is essentially an unbounded number of support 
     * vectors. Increasing the budget after adding the first vector is always 
     * allowed, but it may not be possible to reduce the number of current 
     * support vectors is above the desired budget. 
     * 
     * @param maxBudget the maximum number of allowed support vectors
     */
    public void setMaxBudget(int maxBudget)
    {
        if(maxBudget < 1)
            throw new IllegalArgumentException("Budget must be positive, not " + maxBudget);
        this.maxBudget = maxBudget;
    }

    /**
     * Returns the current maximum budget for support vectors
     * @return the maximum budget for support vectors
     */
    public int getMaxBudget()
    {
        return maxBudget;
    }

    /**
     * Sets the method used for maintaining the budget of support vectors. This
     * method must be called <i>before</i> any vectors are added to the 
     * KernelPoint. <br>
     * <br>
     * The budget maintenance strategy used controls the time complexity and 
     * memory use of the model. 
     * @param budgetStrategy the budget maintenance strategy
     */
    public void setBudgetStrategy(BudgetStrategy budgetStrategy)
    {
        if(getBasisSize() > 0)
            throw new RuntimeException("KerenlPoint already started, budget may not be changed");
        this.budgetStrategy = budgetStrategy;
    }

    /**
     * Returns the budget method used 
     * @return the budget method used 
     */
    public BudgetStrategy getBudgetStrategy()
    {
        return budgetStrategy;
    }

    /**
     * Sets the error tolerance used for projection maintenance strategies such 
     * as {@link BudgetStrategy#PROJECTION}
     * @param errorTolerance the error tolerance in [0, 1]
     */
    public void setErrorTolerance(double errorTolerance)
    {
        if(Double.isNaN(errorTolerance) || errorTolerance < 0 || errorTolerance > 1)
            throw new IllegalArgumentException("Error tolerance must be in [0, 1], not " + errorTolerance);
        this.errorTolerance = errorTolerance;
    }
    
    /**
     * Returns the error tolerance that is used depending on the 
     * {@link BudgetStrategy} in use
     * @return the error tolerance value
     */
    public double getErrorTolerance()
    {
        return errorTolerance;
    }
    
    /**
     * Returns the squared values of the 2 norm of the point this object 
     * represents
     * 
     * @return the squared value of the 2 norm
     */
    public double getSqrdNorm()
    {
        if(!normGood)
        {
            sqrdNorm = 0;
            for(int i = 0; i < alpha.size(); i++)
            {
                if(K != null)//we already know all the values of K
                {
                    sqrdNorm += alpha.get(i)*alpha.get(i)*K.get(i, i);
                    for(int j = i+1; j < alpha.size(); j++)
                        sqrdNorm += 2*alpha.get(i)*alpha.get(j)*K.get(i, j);
                }
                else//nope, compute as needed
                {
                    sqrdNorm += alpha.get(i)*alpha.get(i)*k.eval(i, i, vecs, kernelAccel);
                    for(int j = i+1; j < alpha.size(); j++)
                        sqrdNorm += 2*alpha.get(i)*alpha.get(j)*k.eval(i, j, vecs, kernelAccel);
                }
            }
            normGood = true;
        }
        return sqrdNorm;
    }
    
    /**
     * Computes the dot product between the kernel point this object represents 
     * and the given input vector in the kernel space. 
     * 
     * @param x the input vector to work with
     * @return the dot product in the kernel space between this point and {@code x}
     */
    public double dot(Vec x)
    {
        return dot(x, k.getQueryInfo(x));
    }
    
    /**
     * Computes the dot product between the kernel point this object represents 
     * and the given input vector in the kernel space
     * 
     * @param x the input vector to work with
     * @param qi the query information for the vector, or {@code null} only if 
     * the kernel in use does not support acceleration. 
     * @return the dot product in the kernel space between this point and {@code x}
     */
    public double dot(Vec x, List<Double> qi)
    {
        if(getBasisSize() == 0)
            return 0;
        return k.evalSum(vecs, kernelAccel, alpha.getBackingArray(), x, qi, 0, alpha.size());
    }
    
    /**
     * Returns the dot product between this point and another in the kernel 
     * space
     * @param x the point to take the dot product with
     * @return the dot product in the kernel space between this point and {@code x}
     */
    public double dot(KernelPoint x)
    {
        if(getBasisSize() == 0 || x.getBasisSize() == 0) 
            return 0;
        int shift = this.alpha.size();
        List<Vec> mergedVecs = ListUtils.mergedView(this.vecs, x.vecs);
        List<Double> mergedCache;
        if(this.kernelAccel == null || x.kernelAccel == null)
            mergedCache = null;
        else
            mergedCache = ListUtils.mergedView(this.kernelAccel, x.kernelAccel);
        
        double dot = 0;
        for(int i = 0; i < this.alpha.size(); i++)
            for(int j = 0; j < x.alpha.size(); j++)
            {
                dot += this.alpha.get(i)*x.alpha.get(j)*k.eval(i, j+shift, mergedVecs, mergedCache);
            }
        return dot;
    }
    
    /**
     * Computes the Euclidean distance between this kernel point and the given 
     * input in the kernel space
     * @param x the input vector to work with
     * @return the Euclidean distance between this point and {@code x} in the 
     * kernel space
     */
    public double dist(Vec x)
    {
        return dist(x, k.getQueryInfo(x));
    }
    
    /**
     * Computes the Euclidean distance between this kernel point and the given 
     * input in the kernel space
     * @param x the input vector to work with
     * @param qi the query information for the vector, or {@code null} only if 
     * the kernel in use does not support acceleration. 
     * @return the Euclidean distance between this point and {@code x} in the 
     * kernel space
     */
    public double dist(Vec x, List<Double> qi)
    {
        double k_xx = k.eval(0, 0, Arrays.asList(x), qi);
        return Math.sqrt(k_xx+getSqrdNorm()-2*dot(x, qi));
    }
    
    /**
     * Computes the Euclidean distance between this kernel point and the given
     * kernel point in the kernel space
     * @param x the input point to work with
     * @return the Euclidean distance between this point and {@code x} in the 
     * kernel space
     */
    public double dist(KernelPoint x)
    {
        if(this == x)//dist to self is 0
            return 0;
        double d = this.getSqrdNorm() + x.getSqrdNorm() - 2 * dot(x);
        return Math.sqrt(Math.max(0, d));//Avoid rare cases wehre 2*dot might be slightly larger
    }
    
    /**
     * Alters this point to be multiplied by the given value
     * @param c the value to multiply by
     */
    public void mutableMultiply(double c)
    {
        if(Double.isNaN(c) || Double.isInfinite(c))
            throw new IllegalArgumentException("multiplier must be a real value, not " + c);
        if(getBasisSize() == 0)
            return;
        sqrdNorm *= c*c;
        alpha.getVecView().mutableMultiply(c);
    }
    
    /**
     * Alters this point to contain the given input vector as well
     * @param x_t the vector to add
     */
    public void mutableAdd(Vec x_t)
    {
        mutableAdd(1.0, x_t);
    }
    
    /**
     * Alters this point to contain the given input vector as well
     * @param c the multiplicative constant to apply with the vector
     * @param x_t the vector to add
     */
    public void mutableAdd(double c, Vec x_t)
    {
        mutableAdd(c, x_t, k.getQueryInfo(x_t));
    }
    
    /**
     * Alters this point to contain the given input vector as well
     * @param c the multiplicative constant to apply with the vector
     * @param x_t the vector to add
     * @param qi the query information for the vector, or {@code null} only if 
     * the kernel in use does not support acceleration. 
     */
    public void mutableAdd(double c, Vec x_t, final List<Double> qi)
    {
        if(c == 0)
            return;
        normGood = false;
        double y_t = c;
        final double k_tt = k.eval(0, 0, Arrays.asList(x_t), qi);
        
        if(budgetStrategy == BudgetStrategy.PROJECTION)
        {
            if(K == null)//first point to be added
            {
                KExpanded = new DenseMatrix(16, 16);
                K = new SubMatrix(KExpanded, 0, 0, 1, 1);
                K.set(0, 0, k_tt);
                InvKExpanded = new DenseMatrix(16, 16);
                InvK = new SubMatrix(InvKExpanded, 0, 0, 1, 1);
                InvK.set(0, 0, 1/k_tt);
                alpha.add(y_t);
                vecs.add(x_t);
                if(kernelAccel != null)
                    kernelAccel.addAll(qi);
                return;
            }

            //Normal case
            DenseVector kxt = new DenseVector(K.rows());

            for (int i = 0; i < kxt.length(); i++)
                kxt.set(i, k.eval(i, x_t, qi, vecs, kernelAccel));

            //ALD test
            final Vec alphas_t = InvK.multiply(kxt);
            final double delta_t = k_tt-alphas_t.dot(kxt);
            final int size = K.rows();

            if(delta_t > errorTolerance && size < maxBudget)//add to the dictionary
            {
                vecs.add(x_t);
                if(kernelAccel != null)
                    kernelAccel.addAll(qi);

                if(size == KExpanded.rows())//we need to grow first
                {
                    KExpanded.changeSize(size*2, size*2);
                    InvKExpanded.changeSize(size*2, size*2);
                }

                Matrix.OuterProductUpdate(InvK, alphas_t, alphas_t, 1/delta_t);
                K = new SubMatrix(KExpanded, 0, 0, size+1, size+1);
                InvK = new SubMatrix(InvKExpanded, 0, 0, size+1, size+1);

                //update bottom row and side columns
                for(int i = 0; i < size; i++)
                {
                    K.set(size, i, kxt.get(i));
                    K.set(i, size, kxt.get(i));

                    InvK.set(size, i, -alphas_t.get(i)/delta_t);
                    InvK.set(i, size, -alphas_t.get(i)/delta_t);
                }

                //update bottom right corner
                K.set(size, size, k_tt);
                InvK.set(size, size, 1/delta_t);
                alpha.add(y_t);

            }
            else//project onto dictionary
            {
                Vec alphaVec = alpha.getVecView();
                alphaVec.mutableAdd(y_t, alphas_t);
                normGood = false;
            }
        }
        else if(budgetStrategy == BudgetStrategy.MERGE_RBF)
        {
            normGood = false;
            addPoint(x_t, qi, y_t);
            
            if(vecs.size() > maxBudget)
            {
                /*
                 * we use the same approximation method as in projection 
                 * (Section 4.2) by fixing m as theSV with the smallest value
                 * of || α_m ||^2
                 */
                int m = 0;
                double alpha_m = abs(alpha.get(m));
                for(int i = 1; i < alpha.size(); i++)
                    if(abs(alpha.getD(i)) < abs(alpha_m))
                    {
                        alpha_m = alpha.getD(i);
                        m = i;
                    }
                
                
                double minLoss = Double.POSITIVE_INFINITY;
                int n = -1;
                double n_h = 0;
                double n_alpha_z = 0;
                double tol = 1e-3;
                while (n == -1)
                {
                    for (int i = 0; i < alpha.size(); i++)
                    {
                        if (i == m)
                            continue;
                        double a_m = alpha_m, a_n = alpha.getD(i);
                        double normalize = a_m+a_n;
                        if (abs(normalize) < tol)//avoid alphas that nearly cancle out
                            continue;
                        final double k_mn = k.eval(i, m, vecs, kernelAccel);
                        
                        double h = getH(k_mn, a_m/normalize, a_n/normalize);
                        
                        /*
                         * we can get k(m, z) without forming z when using RBF
                         * 
                         * exp(-(m-z)^2) = exp(-(m- (h m+(1-h) n))^2 ) = 
                         * exp(-(x-y)^2(h-1)^2) = exp((x-y)^2)^(h-1)^2
                         * 
                         * and since: 0 < h < 1 (h-1)^2 = (1-h)^2
                         */
                        double k_mz = pow(k_mn, (1 - h) * (1 - h));
                        double k_nz = pow(k_mn, h * h);
                        
                        //TODO should we fall back to forming z if we use a non RBF kernel?


                        /*
                         * Determin the best by the smallest change in norm, 2x2 
                         * matrix for the original alphs and alpha_z on its own
                         */
                        double alpha_z = a_m * k_mz + a_n * k_nz;

                        double loss = a_m * a_m + a_n * a_n
                                + 2 * k_mn * a_m * a_n
                                - alpha_z*alpha_z;

                        if (loss < minLoss)
                        {
                            minLoss = loss;
                            n = i;
                            n_h = h;
                            n_alpha_z = alpha_z;
                        }
                    }
                    tol /= 10;
                }

                Vec n_z = vecs.get(m).multiply(n_h);
                n_z.mutableAdd(1-n_h, vecs.get(n));
                final List<Double> nz_qi = k.getQueryInfo(n_z);
                
                finalMergeStep(m, n, n_z, nz_qi, n_alpha_z, true);
            }
        }
        else if(budgetStrategy == BudgetStrategy.STOP)
        {
            normGood = false;
            if(getBasisSize() < maxBudget)
                addPoint(x_t, qi, y_t);
        }
        else if(budgetStrategy == BudgetStrategy.RANDOM)
        {
            normGood = false;
            if(getBasisSize() >= maxBudget)
            {
                Random rand = RandomUtil.getRandom();//TODO should probably move this out
                int toRemove = rand.nextInt(vecs.size());
                removeIndex(toRemove);
            }
            
            addPoint(x_t, qi, y_t);
        }
        else
            throw new RuntimeException("BUG: report me!");
            
        
    }
    
    /**
     * Adds a point to the set
     * @param x_t the value to add
     * @param qi the query information for the value
     * @param y_t the constant value to add
     */
    private void addPoint(Vec x_t, final List<Double> qi, double y_t)
    {
        vecs.add(x_t);
        if (kernelAccel != null)
            kernelAccel.addAll(qi);
        alpha.add(y_t);
    }
    
    /**
     * Performs the last merging step removing the old vecs and adding the new
     * merged one
     * @param m the first of the original index to remove
     * @param n the second of the original index to remove
     * @param n_z the merged vec to replace them with
     * @param nz_qi the query info for the new vec
     * @param n_alpha_z the alpha value for the new merged vec
     */
    protected void finalMergeStep(int m, int n, Vec n_z, final List<Double> nz_qi, double n_alpha_z, boolean alterVecs)
    {
        int smallIndx = min(m, n);
        int largeIndx = max(m, n);
        
        alpha.remove(largeIndx);
        alpha.remove(smallIndx);

        if(alterVecs)
        {
            vecs.remove(largeIndx);
            vecs.remove(smallIndx);

            kernelAccel.remove(largeIndx);
            kernelAccel.remove(smallIndx);


            vecs.add(n_z);
            //XXX the following check was redundant
//            if (kernelAccel != null)
                kernelAccel.addAll(nz_qi);
        }
        alpha.add(n_alpha_z);
    }

    /**
     * Gets the minimum of H in [0, 1] the for RBF merging<br>
     * a<sub>m</sub>k<sub>mn</sub><sup>(1-h)^2</sup> + a<sub>n</sub>k<sub>mn</sub><sup>h^2</sup>
     * <br>. 
     * THIS METHOD IS A BOTTLE NECK, so it has some optimization hacks<br>
     * Only one of the coefficients can be negative. 
     * @param k_mn the shared kernel value on both halves of the equation
     * @param a_m the first coefficient
     * @param a_n the second coefficient
     * @return the value of h that maximizes the response
     */
    protected static double getH(final double k_mn, final double a_m, final double a_n)
    {
        if(a_m == a_n)
            return 0.5;
        
        final Function f = new FunctionBase()
        {
            /**
			 * 
			 */
			private static final long serialVersionUID = -6891301465754898634L;

			@Override
            public double f(Vec x)
            {
                final double h = x.get(0);
                //negative to maximize isntead of minimize
                /*
                 * We aren't solving to a super high degree of accuracy anyway, 
                 * so use an approximate pow. Its impact is only noticible for 
                 * very small budget sizes
                 */
                return -(a_m * FastMath.pow(k_mn, (1 - h) * (1 - h)) + a_n * FastMath.pow(k_mn, h * h));
            }
        };
        
        /*
         * Only a few iterations of golden search are done. Often the exact min 
         * is very nearly 0 or 1, and that dosn't seem to really help with the 
         * merging. I've gotten better generalization so far by allowing only a
         * small number of fininte steps. 
         */
        /*
         * if one is pos and the other is negative, the minimum value is going 
         * to be near 0 or 1
         */
        if(Math.signum(a_m) != Math.signum(a_n))
            if(a_m < 0)//we give a 
                return GoldenSearch.minimize(1e-3, 100, 0.0, 0.2, 0, f, 0.0);
            else if(a_n < 0)
                return GoldenSearch.minimize(1e-3, 100, 0.8, 1.0, 0, f, 0.0);
        
        
        if(a_m > a_n)
            return GoldenSearch.minimize(1e-3, 100, 0.5, 1.0, 0, f, 0.0);
        else
            return GoldenSearch.minimize(1e-3, 100, 0.0, 0.5, 0, f, 0.0);
            
    }
    
    /**
     * Removes the vec, alpha, and kernel cache associate with the given index
     * @param toRemove the index to remove
     */
    protected void removeIndex(int toRemove)
    {
        if(kernelAccel != null)
        {
            int num = this.kernelAccel.size()/vecs.size();
            for(int i = 0; i < num; i++)
                kernelAccel.remove(toRemove);
        }
        alpha.remove(toRemove);
        vecs.remove(toRemove);
    }
    
    /**
     * Returns the number of vectors serving as the basis set
     * @return the number of vectors serving as the basis set
     */
    public int getBasisSize()
    {
        if(vecs == null)
            return 0;
        return vecs.size();
    }
    
    /**
     * Returns the list of the raw vectors being used by the kernel points. 
     * Altering this vectors will alter the same vectors used by the KernelPoint
     * and will cause inconsistent results.<br>
     * <br>
     * The returned list may not be modified
     * 
     * @return a the list of all the vectors in use as a basis set by this KernelPoint
     */
    public List<Vec> getRawBasisVecs()
    {
        return Collections.unmodifiableList(vecs);
    }

    @Override
    public KernelPoint clone() 
    {
        return new KernelPoint(this);
    }
}
