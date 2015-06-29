
package jsat.distributions.kernels;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.distributions.Distribution;
import jsat.distributions.Exponential;
import jsat.distributions.LogUniform;
import jsat.distributions.Uniform;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.math.OnLineStatistics;
import jsat.text.GreekLetters;
import jsat.utils.DoubleList;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * Provides a kernel for the Radial Basis Function, which is of the form
 * <br>
 * k(x, y) = exp(-||x-y||<sup>2</sup>/(2*&sigma;<sup>2</sup>))
 * 
 * @author Edward Raff
 */
public class RBFKernel extends BaseL2Kernel
{

    private static final long serialVersionUID = -6733691081172950067L;
    private double sigma;
    private double sigmaSqrd2Inv;

    /**
     * Creates a new RBF kernel with &sigma; = 1
     */
    public RBFKernel()
    {
        this(1.0);
    }

    /**
     * Creates a new RBF kernel
     * @param sigma the sigma parameter
     */
    public RBFKernel(double sigma)
    {
        setSigma(sigma);
    }

    @Override
    public double eval(Vec a, Vec b)
    {
        if(a == b)//Same refrence means dist of 0, exp(0) = 1
            return 1;
        return Math.exp(-Math.pow(a.pNormDist(2, b),2) * sigmaSqrd2Inv);
    }

    @Override
    public double eval(int a, int b, List<? extends Vec> trainingSet, List<Double> cache)
    {
        if(a == b)
            return 1; 
        return Math.exp(-getSqrdNorm(a, b, trainingSet, cache)* sigmaSqrd2Inv);
    }
    
    @Override
    public double eval(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        return Math.exp(-getSqrdNorm(a, b, qi, vecs, cache)* sigmaSqrd2Inv);
    }

    /**
     * Sets the sigma parameter, which must be a positive value
     * @param sigma the sigma value
     */
    public void setSigma(double sigma)
    {
        if(sigma <= 0)
            throw new IllegalArgumentException("Sigma must be a positive constant, not " + sigma);
        this.sigma = sigma;
        this.sigmaSqrd2Inv = 0.5/(sigma*sigma);
    }

    public double getSigma()
    {
        return sigma;
    }

    @Override
    public String toString()
    {
        return "RBF Kernel( " + GreekLetters.sigma +" = " + sigma +")";
    }

    @Override
    public RBFKernel clone()
    {
        return new RBFKernel(sigma);
    }
    
    /**
     * Another common (equivalent) form of the RBF kernel is k(x, y) = 
     * exp(-&gamma;||x-y||<sup>2</sup>). This method converts the &sigma; value 
     * used by this class to the equivalent &gamma; value. 
     * @param sigma the value of &sigma;
     * @return the equivalent &gamma; value. 
     */
    public static double sigmaToGamma(double sigma)
    {
        if(sigma <= 0 || Double.isNaN(sigma) || Double.isInfinite(sigma))
            throw new IllegalArgumentException("sigma must be positive, not " + sigma);
        return 1/(2*sigma*sigma);
    }
    
    /**
     * Another common (equivalent) form of the RBF kernel is k(x, y) = 
     * exp(-&gamma;||x-y||<sup>2</sup>). This method converts the &gamma; value 
     * equivalent &sigma; value used by this class. 
     * @param gamma the value of &gamma;
     * @return the equivalent &sigma; value
     */
    public static double gammToSigma(double gamma)
    {
        if(gamma <= 0 || Double.isNaN(gamma) || Double.isInfinite(gamma))
            throw new IllegalArgumentException("gamma must be positive, not " + gamma);
        return 1/Math.sqrt(2*gamma);
    }
    
    /**
     * Guess the distribution to use for the kernel width term
     * {@link #setSigma(double) &sigma;} in the RBF kernel.
     *
     * @param d the data set to get the guess for
     * @return the guess for the &sigma; parameter in the RBF Kernel 
     */
    public static Distribution guessSigma(DataSet d)
    {
        //we will use a simple strategy of estimating the mean sigma to test based on the pair wise distances of random points

        //to avoid n^2 work for this, we will use a sqrt(n) sized sample as n increases so that we only do O(n) work
        List<Vec> allVecs = d.getDataVectors();

        int toSample = d.getSampleSize();
        if (toSample > 5000)
            toSample = 5000 + (int) Math.floor(Math.sqrt(d.getSampleSize() - 5000));

        DoubleList vals = new DoubleList(toSample*toSample);
        EuclideanDistance dist = new EuclideanDistance();

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
}
