package jsat.distributions.kernels;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.DoubleList;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * This class provides a generalization of the {@link RBFKernel} to arbitrary 
 * {@link DistanceMetric distance metrics}, and is of the form 
 * <i>exp(-d(x, y)<sup>2</sup>/(2 {@link #setSigma(double) &sigma;}<sup>2</sup>
 * ))</i>. So long as the distance metric is valid, the resulting kernel trick
 * will be a valid kernel. <br>
 * <br>
 * If the {@link EuclideanDistance} is used, then this becomes equivalent to the
 * {@link RBFKernel}. <br> 
 * <br>
 * Note, that since the {@link KernelTrick} has no concept of training - the 
 * distance metric can not require training either. A pre-trained metric can 
 * be admissible thought. 
 * 
 * @author Edward Raff
 */
public class GeneralRBFKernel extends DistanceMetricBasedKernel
{

    private static final long serialVersionUID = 1368225926995372017L;
    private double sigma;
    private double sigmaSqrd2Inv;

    /**
     * Creates a new Generic RBF Kernel
     * @param d the distance metric to use
     * @param sigma the standard deviation to use
     */
    public GeneralRBFKernel(DistanceMetric d, double sigma)
    {
        super(d);
        setSigma(sigma);
    }
    
    /**
     * Sets the kernel width parameter, which must be a positive value. Larger 
     * values indicate a larger width
     * 
     * @param sigma the sigma value
     */
    public void setSigma(double sigma)
    {
        if(sigma <= 0 || Double.isNaN(sigma) || Double.isInfinite(sigma))
            throw new IllegalArgumentException("Sigma must be a positive constant, not " + sigma);
        this.sigma = sigma;
        this.sigmaSqrd2Inv = 0.5/(sigma*sigma);
    }

    /**
     * 
     * @return the width parameter to use for the kernel 
     */
    public double getSigma()
    {
        return sigma;
    }
    
    @Override
    public KernelTrick clone()
    {
        return new GeneralRBFKernel(d.clone(), sigma);
    }

    @Override
    public double eval(Vec a, Vec b)
    {
        double dist = d.dist(a, b);
        return Math.exp(-dist*dist * sigmaSqrd2Inv);
    }

    @Override
    public double eval(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        double dist = d.dist(a, b, qi, vecs, cache);
        return Math.exp(-dist*dist * sigmaSqrd2Inv);
        
    }

    @Override
    public double eval(int a, int b, List<? extends Vec> vecs, List<Double> cache)
    {
        double dist = d.dist(a, b, vecs, cache);
        return Math.exp(-dist*dist * sigmaSqrd2Inv);
    }
    
    /**
     * Guess the distribution to use for the kernel width term
     * {@link #setSigma(double) &sigma;} in the General RBF kernel.
     *
     * @param d the data set to get the guess for
     * @return the guess for the &sigma; parameter in the General RBF Kernel 
     */
    public Distribution guessSigma(DataSet d)
    {
        return guessSigma(d, this.d);
    }
    
    /**
     * Guess the distribution to use for the kernel width term
     * {@link #setSigma(double) &sigma;} in the General RBF kernel.
     *
     * @param d the data set to get the guess for
     * @param dist the distance metric to assume is being used in the kernel
     * @return the guess for the &sigma; parameter in the General RBF Kernel 
     */
    public static Distribution guessSigma(DataSet d, DistanceMetric dist)
    {
        //we will use a simple strategy of estimating the mean sigma to test based on the pair wise distances of random points

        //to avoid n^2 work for this, we will use a sqrt(n) sized sample as n increases so that we only do O(n) work
        List<Vec> allVecs = d.getDataVectors();

        int toSample = d.getSampleSize();
        if (toSample > 5000)
            toSample = 5000 + (int) Math.floor(Math.sqrt(d.getSampleSize() - 5000));

        DoubleList vals = new DoubleList(toSample*toSample);

        if (d instanceof ClassificationDataSet && ((ClassificationDataSet) d).getPredicting().getNumOfCategories() == 2)
        {
            ClassificationDataSet cdata = (ClassificationDataSet) d;
            List<Vec> class0 = new ArrayList<Vec>(toSample / 2);
            List<Vec> class1 = new ArrayList<Vec>(toSample / 2);
            IntList randOrder = new IntList(d.getSampleSize());
            ListUtils.addRange(randOrder, 0, d.getSampleSize(), 1);
            Collections.shuffle(randOrder);
            //collet a random sample of data
            for (int i = 0; i < randOrder.size(); i++)
            {
                int indx = randOrder.getI(i);
                if (cdata.getDataPointCategory(indx) == 0 && class0.size() < toSample / 2)
                    class0.add(cdata.getDataPoint(indx).getNumericalValues());
                else if (cdata.getDataPointCategory(indx) == 1 && class0.size() < toSample / 2)
                    class1.add(cdata.getDataPoint(indx).getNumericalValues());
            }

            int j_start = class0.size();
            class0.addAll(class1);
            List<Double> cache = dist.getAccelerationCache(class0);
            for (int i = 0; i < j_start; i++)
                for (int j = j_start; j < class0.size(); j++)
                    vals.add(dist.dist(i, j, allVecs, cache));
        }
        else
        {
            Collections.shuffle(allVecs);
            if (d.getSampleSize() > 5000)
                allVecs = allVecs.subList(0, toSample);

            List<Double> cache = dist.getAccelerationCache(allVecs);
            for (int i = 0; i < allVecs.size(); i++)
                for (int j = i + 1; j < allVecs.size(); j++)
                    vals.add(dist.dist(i, j, allVecs, cache));
        }
        
        Collections.sort(vals);
        double median = vals.get(vals.size()/2);
        return new LogUniform(Math.exp(Math.log(median)-4), Math.exp(Math.log(median)+4));
    }

    @Override
    public boolean normalized()
    {
        return true;
    }
}
