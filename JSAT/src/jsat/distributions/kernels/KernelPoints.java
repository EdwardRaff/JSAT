package jsat.distributions.kernels;

import java.util.ArrayList;
import java.util.List;
import jsat.linear.Vec;
import jsat.utils.DoubleList;

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
    private boolean mergeGrams;
    
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
     * @see #setMergeGrams(boolean) 
     */
    public KernelPoints(KernelTrick k, int points, double errorTolerance, boolean mergeGrams)
    {
        if(points < 1)
            throw new IllegalArgumentException("Number of points must be positive, not " + points);
        this.k = k;
        setMergeGrams(mergeGrams);
        this.errorTolerance = errorTolerance;
        this.points = new ArrayList<KernelPoint>(points);
        for(int i = 0; i < points; i++)
            this.points.add(new KernelPoint(k, errorTolerance));
        
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public KernelPoints(KernelPoints toCopy)
    {
        this.k = toCopy.k.clone();
        this.errorTolerance = toCopy.errorTolerance;
        this.mergeGrams = toCopy.mergeGrams;
        this.points = new ArrayList<KernelPoint>(toCopy.points.size());
        if(toCopy.points.get(0).getBasisSize() == 0)//special case, nothing has been added
        {
            for(int i = 0; i < toCopy.points.size(); i++)
                this.points.add(new KernelPoint(k, errorTolerance));
        }
        else
        {
            if(this.mergeGrams)
            {
                KernelPoint source = this.points.get(0).clone();
                for(int i = 1; i < toCopy.points.size(); i++)
                {
                    KernelPoint toAdd = new KernelPoint(k, errorTolerance);
                    standardMove(toAdd, source);
                    toAdd.kernelAccel = source.kernelAccel;
                    toAdd.vecs = source.vecs;
                    toAdd.alpha = new DoubleList(toCopy.points.get(i).alpha);
                }
            }
            else
            {
                for(KernelPoint kp : toCopy.points)
                    this.points.add(kp.clone());
            }
        }
    }

    public KernelTrick getKernel()
    {
        return k;
    }

    public double getErrorTolerance()
    {
        return errorTolerance;
    }

    /**
     * Controls whether or not multiple KernelPoints will share a single gram 
     * matrix or will each maintain their own individual matrices. If most 
     * KernelPoint objects will receive the same vectors, then sharing a single 
     * gram matrix should use the least memory.  If each KernelPoint will be 
     * altered with different vectors, than maintaining independent matrices 
     * should have the least memory use. <br>
     * Note: If this method is called, it must occur <b><i>before</i></b> any 
     * alterations are done to any of the KernelPoint objects it would contain. 
     * Altering this value after data points have been added / altered will 
     * result in incorrect results.
     * 
     * @param mergeGrams {@code true} to make all KernelPoints share one gram 
     * matrix, or {@code false} to make them use different ones.  
     */
    public void setMergeGrams(boolean mergeGrams)
    {
        this.mergeGrams = mergeGrams;
    }

    /**
     * Returns {@code true} if one gram matrix will be shared by all kernel 
     * points, {@code false} if they will each maintain their own kernel matrix. 
     * @return {@code true} if one gram matrix will be shared by all kernel 
     * points, {@code false} if they will each maintain their own kernel matrix. 
     */
    public boolean isMergeGrams()
    {
        return mergeGrams;
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
     * Alters ones of the KernelPoint objects by adding / subjecting a vector 
     * from it
     * @param k the index of the KernelPoint to use
     * @param c the constant to multiply the vector being added by
     * @param x_t the vector to add to the kernel point
     * @param qi the query information for the vector, or {@code null} only if 
     * the kernel in use does not support acceleration. 
     */
    public void mutableAdd(int k, double c, Vec x_t, final List<Double> qi)
    {
        KernelPoint kp_k = points.get(k);
        
        if(kp_k.getBasisSize() == 0)//Special case, init people
        {
            kp_k.mutableAdd(c, x_t, qi);
            //That initializes the structure, now we need to make people point to the same ones
            for(int i = 0; i < points.size() && mergeGrams; i++)
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
            int origSize = kp_k.getBasisSize();
            kp_k.mutableAdd(c, x_t, qi);
            if(origSize != kp_k.getBasisSize() && mergeGrams)//update kernels & add alpha
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
    }
    
    /**
     * Adds a new Kernel Point to the internal list this object represents. The
     * new Kernel Point will be equivalent to creating a new KernelPoint 
     * directly. <br>
     * If{@link #isMergeGrams() } is {@code true}, the new point will share the
     * currently existing kernel matrix. 
     */
    public void addNewKernelPoint()
    {
        if(mergeGrams)
        {
            KernelPoint source = points.get(0);
            KernelPoint toAdd = new KernelPoint(k, errorTolerance);
            standardMove(toAdd, source);
            toAdd.kernelAccel = source.kernelAccel;
            toAdd.vecs = source.vecs;
            toAdd.alpha = new DoubleList(source.alpha.size());
            for(int i = 0; i < source.alpha.size(); i++)
                toAdd.alpha.add(0.0);
            points.add(toAdd);
        }
        else
        {
            points.add(new KernelPoint(k, errorTolerance));
        }
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
     * Returns the number of basis vectors in use. If {@link #isMergeGrams() }
     * is {@code true}, this value is the exact number of basis vectors in use. 
     * If {@code false}, this returns the sum of the number of basis vectors 
     * used by each KernelPoint. If a vector has been added to more than one 
     * Kernel Point it may get double counted (or more), so the value returned 
     * may not be reasonable in that case. 
     * @return the number of basis vectors in use
     */
    public int getBasisSize()
    {
        if(mergeGrams)
            return this.points.get(0).getBasisSize();
        int size = 0;
        for(KernelPoint kp : this.points)
            size += kp.getBasisSize();
        return size;
    }
    
    /**
     * Returns a list of the raw vectors being used by the kernel points. 
     * Altering this vectors will alter the same vectors used by these objects
     * and will cause inconsistent results. <br>
     * <br>
     * If {@link #isMergeGrams() } is {@code true}, the returned list contains 
     * no duplicate vectors and will be the correct number of items. If 
     * {@code false}, any vectors that were added to more than one KernelPoint 
     * may be returned multiple times in this list. 
     * 
     * @return the list of raw basis vectors used by the Kernel points
     */
    public List<Vec> getRawBasisVecs()
    {
        List<Vec> vecs = new ArrayList<Vec>(getBasisSize());
        if(mergeGrams)
            vecs.addAll(this.points.get(0).vecs);
        else
            for(KernelPoint kp : this.points)
                vecs.addAll(kp.vecs);
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
    
}
