package jsat.distributions.kernels;

import static java.lang.Math.abs;
import static java.lang.Math.pow;
import java.util.*;
import static jsat.distributions.kernels.KernelPoint.getH;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.utils.DoubleList;
import jsat.utils.random.RandomUtil;

/**
 * This class represents a list of {@link KernelPoint} objects. This is done to 
 * avoid excessive memory duplication that can occur when multiple KernelPoints
 * are in use at the same time. 
 * 
 * @author Edward Raff
 */
public class KernelPoints
{
    private KernelTrick k;
    private double errorTolerance;
    private KernelPoint.BudgetStrategy budgetStrategy = KernelPoint.BudgetStrategy.PROJECTION;
    private int maxBudget = Integer.MAX_VALUE;
    private List<KernelPoint> points;
    
    /**
     * Creates a new set of kernel points that uses one unified gram matrix for 
     * each KernelPoint 
     * @param k the kernel trick to use in which to represent a vector in the 
     * kernel space
     * @param points the initial number of kernel points to store in this set
     * @param errorTolerance the maximum error allowed for projecting a vector 
     * instead of adding it to the basis set
     */
    public KernelPoints(KernelTrick k, int points, double errorTolerance)
    {
        this(k, points, errorTolerance, true);
    }

    /**
     * Creates a new set of kernel points
     * @param k the kernel trick to use in which to represent a vector in the 
     * kernel space
     * @param points the initial number of kernel points to store in this set
     * @param errorTolerance the maximum error allowed for projecting a vector 
     * instead of adding it to the basis set
     * @param mergeGrams whether or not to merge the gram matrices of each 
     * KernelPoint. 
     */
    public KernelPoints(KernelTrick k, int points, double errorTolerance, boolean mergeGrams)
    {
        if(points < 1)
            throw new IllegalArgumentException("Number of points must be positive, not " + points);
        this.k = k;
        this.errorTolerance = errorTolerance;
        this.points = new ArrayList<KernelPoint>(points);
        this.points.add(new KernelPoint(k, errorTolerance));
        this.points.get(0).setMaxBudget(maxBudget);
        this.points.get(0).setBudgetStrategy(budgetStrategy);
        for(int i = 1; i < points; i++)
            addNewKernelPoint();
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public KernelPoints(KernelPoints toCopy)
    {
        this.k = toCopy.k.clone();
        this.errorTolerance = toCopy.errorTolerance;
        this.points = new ArrayList<KernelPoint>(toCopy.points.size());
        if(toCopy.points.get(0).getBasisSize() == 0)//special case, nothing has been added
        {
            for(int i = 0; i < toCopy.points.size(); i++)
                this.points.add(new KernelPoint(k, errorTolerance));
        }
        else
        {
            KernelPoint source = this.points.get(0).clone();
            for (int i = 1; i < toCopy.points.size(); i++)
            {
                KernelPoint toAdd = new KernelPoint(k, errorTolerance);
                standardMove(toAdd, source);
                toAdd.kernelAccel = source.kernelAccel;
                toAdd.vecs = source.vecs;
                toAdd.alpha = new DoubleList(toCopy.points.get(i).alpha);
            }
        }
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
    public void setBudgetStrategy(KernelPoint.BudgetStrategy budgetStrategy)
    {
        this.budgetStrategy = budgetStrategy;
        for(KernelPoint kp : points)
            kp.setBudgetStrategy(budgetStrategy);
    }

    /**
     * Returns the budget method used 
     * @return the budget method used 
     */
    public KernelPoint.BudgetStrategy getBudgetStrategy()
    {
        return budgetStrategy;
    }

    public KernelTrick getKernel()
    {
        return k;
    }
    
    /**
     * Sets the error tolerance used for projection maintenance strategies such 
     * as {@link KernelPoint.BudgetStrategy#PROJECTION}
     * @param errorTolerance the error tolerance in [0, 1]
     */
    public void setErrorTolerance(double errorTolerance)
    {
        if(Double.isNaN(errorTolerance) || errorTolerance < 0 || errorTolerance > 1)
            throw new IllegalArgumentException("Error tolerance must be in [0, 1], not " + errorTolerance);
        this.errorTolerance = errorTolerance;
        for(KernelPoint kp : points)
            kp.setErrorTolerance(errorTolerance);
    }
    
    /**
     * Returns the error tolerance that is used depending on the 
     * {@link KernelPoint.BudgetStrategy} in use
     * @return the error tolerance value
     */
    public double getErrorTolerance()
    {
        return errorTolerance;
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
        for(KernelPoint kp : points)
            kp.setMaxBudget(maxBudget);
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
     * Returns the squared 2 norm value of the {@code k}'th KernelPoint
     * @param k the KernelPoint to get the norm of
     * @return the squared 2 norm of the {@code k}'th KernelPoint
     */
    public double getSqrdNorm(int k)
    {
        return points.get(k).getSqrdNorm();
    }
    
    /**
     * Computes the dot product between the {@code k}'th KernelPoint and the 
     * given vector in the kernel space. 
     * @param k the index of the KernelPoint in this set to contribute to the 
     * dot product
     * @param x the vector to contribute to the dot product
     * @param qi the query information for the vector, or {@code null} only if 
     * the kernel in use does not support acceleration. 
     * @return the dot product between the {@code k}'th KernelPoint and the 
     * given vector
     */
    public double dot(int k, Vec x, List<Double> qi)
    {
        return points.get(k).dot(x, qi);
    }
    
    /**
     * Computes the dot product between each KernelPoint in this set and the 
     * given vector in the kernel space. The results are equivalent to an array
     * and setting each value using 
     * {@link #dot(int, jsat.linear.Vec, java.util.List)  } <br>
     * This method should be faster than computing the dot products individual 
     * since it avoids redundant kernel computations
     * 
     * @param x the vector to contribute to the dot product
     * @param qi the query information for the vector, or {@code null} only if 
     * the kernel in use does not support acceleration. 
     * @return an array where the <i>i'th</i> index contains the dot product of
     * the <i>i'th</i> KernelPoint and the given vector
     */
    public double[] dot(Vec x, List<Double> qi)
    {
        double[] dots = new double[points.size()];
        final List<Vec> vecs = points.get(0).vecs;
        final List<Double> cache = points.get(0).kernelAccel;
        for(int i = 0; i < vecs.size(); i++)
        {
            double k_ix = k.eval(i, x, qi, vecs, cache);
            for(int j = 0; j < points.size(); j++)
            {
                double alpha = points.get(j).alpha.getD(i);
                if(alpha != 0)
                    dots[j] += k_ix*alpha;
            }
        }
        return dots;
    }
    
    /**
     * Computes the dot product between the {@code k}'th KernelPoint and the 
     * given KernelPoint 
     * @param k the index of the KernelPoint in this set to contribute to the 
     * dot product
     * @param x the other KernelPoint to contribute to the dot product
     * @return the dot product between the {@code k}'th KernelPoint and the 
     * given KernelPoint
     */
    public double dot(int k, KernelPoint x)
    {
        return points.get(k).dot(x);
    }
    
    /**
     * Computes the dot product between the {@code k}'th KernelPoint and the 
     * {@code j}'th KernelPoint in the given set of points. 
     * @param k the index of the KernelPoint in this set to contribute to the 
     * dot product
     * @param X the other set of KernelPoints
     * @param j the index of the KernelPoint in the given set to contribute to 
     * the dot product
     * @return the dot product between the {@code k}'th KernelPoint and the 
     * {@code j}'th KernelPoint in the given set
     */
    public double dot(int k, KernelPoints X, int j)
    {
        return points.get(k).dot(X.points.get(j));
    }
    
    /**
     * Computes the Euclidean distance in the kernel space between the 
     * {@code k}'th KernelPoint and the given vector
     * @param k the index of the KernelPoint in this set to contribute to the 
     * dot product
     * @param x the point to get the Euclidean distance to
     * @param qi the query information for the vector, or {@code null} only if 
     * the kernel in use does not support acceleration. 
     * @return the Euclidean distance between the {@code k}'th KernelPoint and
     * {@code x} in the kernel space
     */
    public double dist(int k, Vec x, List<Double> qi)
    {
        return points.get(k).dist(x, qi);
    }
    
    /**
     * Computes the Euclidean distance in the kernel space between the 
     * {@code k}'th KernelPoint and the given KernelPoint
     * @param k the index of the KernelPoint in this set to contribute to the 
     * dot product
     * @param x the kernel point to get the Euclidean distance to
     * @return the Euclidean distance between the {@code k}'th KernelPoint and
     * {@code x} in the kernel space
     */
    public double dist(int k, KernelPoint x)
    {
        return points.get(k).dist(x);
    }
    
    /**
     * Computes the Euclidean distance in the kernel space between the 
     * {@code k}'th KernelPoint and the {@code j}'th KernelPoint in the given 
     * set
     * 
     * @param k the index of the KernelPoint in this set to contribute to the 
     * dot product
     * @param X the other set of kernel points to obtain the target KernelPoint
     * @param j the index of the KernelPoint in the given set to contribute to 
     * the dot product
     * @return the Euclidean distance between the {@code k}'th KernelPoint and
     * the {@code j}'th KernelPoint in the other set
     */
    public double dist(int k, KernelPoints X, int j)
    {
        return points.get(k).dist(X.points.get(j));
    }
    
    /**
     * Alters the {@code k}'th KernelPoint by multiplying it with a constant 
     * value
     * @param k the index of the KernelPoint to modify
     * @param c the constant to multiply the KernelPoint by
     */
    public void mutableMultiply(int k, double c)
    {
        points.get(k).mutableMultiply(c);
    }
    
    /**
     * Alters all the KernelPoint objects contained in this set by the same 
     * constant value
     * @param c the constant to multiply the KernelPoints by
     */
    public void mutableMultiply(double c)
    {
        for(KernelPoint kp : points)
            kp.mutableMultiply(c);
    }
    
    /**
     * Alters ones of the KernelPoint objects by adding / subtracting a vector 
     * from it
     * @param k the index of the KernelPoint to use
     * @param c the constant to multiply the vector being added by
     * @param x_t the vector to add to the kernel point
     * @param qi the query information for the vector, or {@code null} only if 
     * the kernel in use does not support acceleration. 
     */
    public void mutableAdd(int k, double c, Vec x_t, final List<Double> qi)
    {
        
    }
    
    /**
     * Alters some of the KernelPoints by adding / subtracting a vector from it
     * @param x_t the vector to add to the kernel point
     * @param cs the array with the constant multiplies. Each non zero in 
     * {@code cs} is a constant to update one of the vectors by. The vector 
     * updated is the one corresponding to the index of the non zero value
     * @param qi the query information for the vector, or {@code null} only if 
     * the kernel in use does not support acceleration. 
     */
    public void mutableAdd(Vec x_t, Vec cs, final List<Double> qi)
    {
        int origSize = getBasisSize();
        if(cs.nnz() == 0)
            return;
        
        if(budgetStrategy == KernelPoint.BudgetStrategy.PROJECTION)
        {
            for(IndexValue iv : cs)
            {
                int k = iv.getIndex(); 
                KernelPoint kp_k = points.get(k);
                double c = iv.getValue();
                if(kp_k.getBasisSize() == 0)//Special case, init people
                {
                    kp_k.mutableAdd(c, x_t, qi);
                    //That initializes the structure, now we need to make people point to the same ones
                    for(int i = 0; i < points.size(); i++)
                    {
                        if(i == k)
                            continue;
                        KernelPoint kp_i = points.get(i);
                        standardMove(kp_i, kp_k);

                        //Only done one time since structures are mutable
                        kp_i.kernelAccel = kp_k.kernelAccel;
                        kp_i.vecs = kp_k.vecs;
                        //and then everyone gets their own private alphas added too
                        kp_i.alpha = new DoubleList(16);
                        kp_i.alpha.add(0.0);
                    }
                }
                else//standard case
                {
                    kp_k.mutableAdd(c, x_t, qi);
                    if(origSize != kp_k.getBasisSize())//update kernels & add alpha
                    {
                        for(int i = 0; i < points.size(); i++)
                            if(i != k)
                            {
                                KernelPoint kp_i = points.get(i);
                                standardMove(kp_i, kp_k);
                                kp_i.alpha.add(0.0);
                            }
                    }
                }
                
                origSize = getBasisSize();//may have changed, but only once
            }
        }
        else if (budgetStrategy == KernelPoint.BudgetStrategy.MERGE_RBF)
        {
            Iterator<IndexValue> cIter = cs.getNonZeroIterator();
            if (getBasisSize() < maxBudget)
            {
                IndexValue firstIndx = cIter.next();
                KernelPoint kp_k = points.get(firstIndx.getIndex());
                kp_k.mutableAdd(firstIndx.getValue(), x_t, qi);
                //fill in the non zeros
                while (cIter.hasNext())
                {
                    IndexValue iv = cIter.next();
                    points.get(iv.getIndex()).alpha.add(iv.getValue());
                }
                addMissingZeros();
            }
            else//we are going to exceed the budget
            {
                KernelPoint kp_k = points.get(0);

                //inser the new vector before merging
                kp_k.vecs.add(x_t);
                if (kp_k.kernelAccel != null)
                    kp_k.kernelAccel.addAll(qi);
                for (IndexValue iv : cs)
                    points.get(iv.getIndex()).alpha.add(iv.getValue());
                addMissingZeros();

                //now go through and merge
                /*
                 * we use the same approximation method as in projection 
                 * (Section 4.2) by fixing m as theSV with the smallest value
                 * of || α_m ||^2
                 */
                int m = 0;
                double alpha_m = 0;
                for (KernelPoint kp : points)
                    alpha_m += pow(kp.alpha.getD(m), 2);
                for (int i = 1; i < kp_k.alpha.size(); i++)
                {
                    double tmp = 0;
                    for (KernelPoint kp : points)
                        tmp += pow(kp.alpha.getD(i), 2);
                    if (tmp < alpha_m)
                    {
                        alpha_m = tmp;
                        m = i;
                    }
                }


                double minLoss = Double.POSITIVE_INFINITY;
                int n = -1;
                double n_h = 0;
                double tol = 1e-3;
                double n_k_mz = 0;
                double n_k_nz = 0;
                while (n == -1)
                {
                    for (int i = 0; i < kp_k.alpha.size(); i++)
                    {
                        if (i == m)
                            continue;
                        double a_m = 0, a_n = 0;
                        for (KernelPoint kp : points)
                        {
                            double a1 = kp.alpha.getD(m);
                            double a2 = kp.alpha.getD(i);
                            double normalize = a1 + a2;
                            if (normalize < 1e-7)
                                continue;
                            a_m += a1 / normalize;
                            a_n += a2 / normalize;
                        }
                        if (abs(a_m + a_n) < tol)//avoid alphas that nearly cancle out
                            break;
                        double k_mn = this.k.eval(i, m, kp_k.vecs, kp_k.kernelAccel);

                        double h = getH(k_mn, a_m, a_n);

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

                        double loss = 0;
                        /*
                         * Determin the best by the smallest change in norm, 2x2 
                         * matrix for the original alphs and alpha_z on its own
                         */
                        for (KernelPoint kp : points)
                        {
                            double aml = kp.alpha.getD(m);
                            double anl = kp.alpha.getD(i);
                            double alpha_z = aml * k_mz + anl * k_nz;

                            loss += aml * aml + anl * anl
                                    + 2 * k_mn * aml * anl
                                    - alpha_z * alpha_z;
                        }

                        if (loss < minLoss)
                        {
                            minLoss = loss;
                            n = i;
                            n_h = h;
                            n_k_mz = k_mz;
                            n_k_nz = k_nz;
                        }
                    }
                    tol /= 10;
                }

                Vec n_z = kp_k.vecs.get(m).multiply(n_h);
                n_z.mutableAdd(1 - n_h, kp_k.vecs.get(n));
                final List<Double> nz_qi = this.k.getQueryInfo(n_z);
                for (int z = 0; z < points.size(); z++)
                {
                    KernelPoint kp = points.get(z);
                    double aml = kp.alpha.getD(m);
                    double anl = kp.alpha.getD(n);
                    double alpha_z = aml * n_k_mz + anl * n_k_nz;
                    kp.finalMergeStep(m, n, n_z, nz_qi, alpha_z, z == 0);
                }

            }
        }
        else if (budgetStrategy == KernelPoint.BudgetStrategy.STOP)
        {
            if(getBasisSize() < maxBudget)
            {
                this.points.get(0).vecs.add(x_t);
                if(this.points.get(0).kernelAccel != null)
                    this.points.get(0).kernelAccel.addAll(qi);
                for(IndexValue iv : cs)
                    this.points.get(iv.getIndex()).alpha.add(iv.getValue());
                addMissingZeros();
            }
        }
        else if(budgetStrategy == KernelPoint.BudgetStrategy.RANDOM)
        {
            if(getBasisSize() >= maxBudget)
            {
                int toRemove = RandomUtil.getRandom().nextInt(getBasisSize());
                if (getBasisSize() == maxBudget)
                    this.points.get(0).removeIndex(toRemove);//now remove alpha from others
                for (int i = 1; i < this.points.size(); i++)
                    this.points.get(i).removeIndex(toRemove);
            }
            //now add the point 
            this.points.get(0).vecs.add(x_t);
            if (this.points.get(0).kernelAccel != null)
                this.points.get(0).kernelAccel.addAll(qi);
            for (IndexValue iv : cs)
                this.points.get(iv.getIndex()).alpha.add(iv.getValue());
            addMissingZeros();
        }
        else
            throw new RuntimeException("BUG: Report Me!");
    }
    
    /**
     * Adds a new Kernel Point to the internal list this object represents. The
     * new Kernel Point will be equivalent to creating a new KernelPoint 
     * directly.
     */
    public void addNewKernelPoint()
    {
        KernelPoint source = points.get(0);
        KernelPoint toAdd = new KernelPoint(k, errorTolerance);
        toAdd.setMaxBudget(maxBudget);
        toAdd.setBudgetStrategy(budgetStrategy);
        
        standardMove(toAdd, source);
        toAdd.kernelAccel = source.kernelAccel;
        toAdd.vecs = source.vecs;
        toAdd.alpha = new DoubleList(source.alpha.size());
        for (int i = 0; i < source.alpha.size(); i++)
            toAdd.alpha.add(0.0);
        points.add(toAdd);
    }

    /**
     * Updates the gram matrix storage of the destination to point at the exact 
     * same objects as the ones from the source. 
     * @param destination the destination object
     * @param source the source object 
     */
    private void standardMove(KernelPoint destination, KernelPoint source)
    {
        destination.InvK = source.InvK;
        destination.InvKExpanded = source.InvKExpanded;
        destination.K = source.K;
        destination.KExpanded = source.KExpanded;
    }
    
    /**
     * Returns the number of basis vectors in use.
     * If a vector has been added to more than one 
     * Kernel Point it may get double counted (or more), so the value returned 
     * may not be reasonable in that case. 
     * @return the number of basis vectors in use
     */
    public int getBasisSize()
    {
        return this.points.get(0).getBasisSize();
    }
    
    /**
     * Returns a list of the raw vectors being used by the kernel points. 
     * Altering this vectors will alter the same vectors used by these objects
     * and will cause inconsistent results. 
     * 
     * @return the list of raw basis vectors used by the Kernel points
     */
    public List<Vec> getRawBasisVecs()
    {
        List<Vec> vecs = new ArrayList<Vec>(getBasisSize());
        vecs.addAll(this.points.get(0).vecs);
        return vecs;
    }
    
    /**
     * Returns the number of KernelPoints stored in this set
     * @return the number of KernelPoints stored in this set
     */
    public int size()
    {
        return points.size();
    }

    @Override
    public KernelPoints clone()
    {
        return new KernelPoints(this);
    }

    /**
     * Adds zeros to all alpha vecs that are not of the same length as the 
     * vec list
     */
    private void addMissingZeros()
    {
        //go back and add 0s for the onces we missed
        for (int i = 0; i < points.size(); i++)
            while(points.get(i).alpha.size() < this.points.get(0).vecs.size())
                points.get(i).alpha.add(0.0);
    }
    
}
