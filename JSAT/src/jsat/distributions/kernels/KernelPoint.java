package jsat.distributions.kernels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import jsat.classifiers.linear.kernelized.Projectron;
import jsat.linear.*;
import jsat.regression.KernelRLS;
import jsat.utils.DoubleList;
import jsat.utils.ListUtils;

/**
 * The Kernel Point represents a kernelized weight vector by a linear 
 * combination of vectors transformed through a 
 * {@link KernelTrick kernel fuctiion}. This implementation maintains a sparse 
 * set of vectors by projecting any new vector onto the current set of vectors 
 * if the error from projecting the new point is less than a certain value. <br>
 * The KernelPoint can be used to efficiently kernelize any algorithm that 
 * relies only on additions of vectors, multiplications, and dot products. <br>
 * <br>
 * See {@link KernelRLS} and {@link Projectron} for methods and papers based on
 * the same ideas used to create this class. <br>
 * Credit goes to Davis King of the <a href="http://dlib.net/ml.html">dlib 
 * library</a> for the idea. 
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
    
    //Internal structure
    private double sqrdNorm = 0;
    private boolean normGood = true;

    /**
     * Creates a new Kernel Point, which is a point in the kernel space 
     * represented by an accumulation of vectors  
     * 
     * @param k the kernel to use
     * @param errorTolerance the maximum error allowed for projecting a vector 
     * instead of adding it to the basis set
     */
    public KernelPoint(KernelTrick k, double errorTolerance)
    {
        this.k = k;
        this.errorTolerance = errorTolerance;
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
            
            this.KExpanded = toCopy.KExpanded.clone();
            this.InvKExpanded = toCopy.InvKExpanded.clone();
            
            this.K = new SubMatrix(KExpanded, 0, 0, toCopy.K.rows(), toCopy.K.cols());
            this.InvK = new SubMatrix(InvKExpanded, 0, 0, toCopy.InvK.rows(), toCopy.InvK.rows());
            this.alpha = new DoubleList(toCopy.alpha);
        }
        
        this.sqrdNorm = toCopy.sqrdNorm;
        this.normGood = toCopy.normGood;
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
                sqrdNorm += alpha.get(i)*alpha.get(i)*K.get(i, i);
                for(int j = i+1; j < alpha.size(); j++)
                    sqrdNorm += 2*alpha.get(i)*alpha.get(j)*K.get(i, j);
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
        normGood = false;
        double y_t = c;
        final double k_tt = k.eval(0, 0, Arrays.asList(x_t), qi);
        
        if(K == null)//first point to be added
        {
            KExpanded = new DenseMatrix(16, 16);
            K = new SubMatrix(KExpanded, 0, 0, 1, 1);
            K.set(0, 0, k_tt);
            InvKExpanded = new DenseMatrix(16, 16);
            InvK = new SubMatrix(InvKExpanded, 0, 0, 1, 1);
            InvK.set(0, 0, 1/k_tt);
            alpha = new DoubleList(16);
            alpha.add(y_t);
            vecs = new ArrayList<Vec>(16);
            vecs.add(x_t);
            if(k.supportsAcceleration())
                kernelAccel = new DoubleList(16);
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

        if(delta_t > errorTolerance)//add to the dictionary
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
     * Returns a list of the raw vectors being used by the kernel points. 
     * Altering this vectors will alter the same vectors used by the KernelPoint
     * and will cause inconsistent results.
     * 
     * @return a list of all the vectors in use as a basis set by this KernelPoint
     */
    public List<Vec> getRawBasisVecs()
    {
        List<Vec> retList = new ArrayList<Vec>(getBasisSize());
        retList.addAll(vecs);
        return retList;
    }

    @Override
    public KernelPoint clone() 
    {
        return new KernelPoint(this);
    }
}
