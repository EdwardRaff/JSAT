
package jsat.distributions.multivariate;

import jsat.math.Function;
import jsat.math.optimization.NelderMead;
import java.util.ArrayList;
import jsat.distributions.Gamma;
import jsat.linear.DenseVector;
import java.util.Random;
import java.util.List;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;

/**
 * The Symmetric Dirichlet Distribution is a special case of the {@link Dirichlet} distribution, and occurs when all alphas have the same value. 
 * 
 * @author Edward Raff
 */
public class SymmetricDirichlet extends MultivariateDistributionSkeleton
{

	private static final long serialVersionUID = -1206894014440494142L;
	private double alpha;
    private int dim;
    
    /**
     * Creates a new Symmetric Dirichlet distribution. 
     * 
     * @param alpha the positive alpha value for the distribution
     * @param dim the dimension of the distribution. 
     * @throws ArithmeticException if a non positive alpha or dimension value is given
     */
    public SymmetricDirichlet(double alpha, int dim)
    {
        setAlpha(alpha);
        setDimension(dim);
    }

    /**
     * Sets the dimension size of the distribution
     * @param dim the new dimension size
     */
    public void setDimension(int dim)
    {
        if(dim <= 0)
            throw new ArithmeticException("A positive number of dimensions must be given");
        this.dim = dim;
    }

    /**
     * Returns the dimension size of the current distribution 
     * @return the number of dimensions in this distribution 
     */
    public int getDimension()
    {
        return dim;
    }
    
    /**
     * Sets the alpha value used for the distribution 
     * @param alpha the positive value for the distribution
     * @throws ArithmeticException if the value given is not a positive value
     */
    public void setAlpha(double alpha) throws ArithmeticException
    {
        if(alpha <= 0 || Double.isNaN(alpha) || Double.isInfinite(alpha))
            throw new ArithmeticException("Symmetric Dirichlet Distribution parameters must be positive, " + alpha + " is invalid");
        this.alpha = alpha;
    }

    /**
     * Returns the alpha value used by this distribution 
     * @return the alpha value used by this distribution 
     */
    public double getAlpha()
    {
        return alpha;
    }
    
    @Override
    public SymmetricDirichlet clone()
    {
        return new SymmetricDirichlet(alpha, dim);
    }

    @Override
    public double logPdf(Vec x)
    {
        if(x.length() != dim)
            throw new ArithmeticException( dim + " variable distribution can not awnser a " + x.length() + " dimension variable");
        double logVal = 0;
        int K = x.length();
        for(int i = 0; i < K; i++)
            logVal += log(x.get(i))*(alpha-1);
        
        logVal = logVal + lnGamma(alpha*K) - lnGamma(alpha)*K;
        if(Double.isInfinite(logVal) || Double.isNaN(logVal) || abs(x.sum() - 1.0) > 1e-14)
            return -Double.MAX_VALUE;
        return logVal;
    }

    public double pdf(Vec x)
    {
        return exp(logPdf(x));
    }

    public <V extends Vec> boolean setUsingData(final List<V> dataSet)
    {
        Function logLike = new Function() 
        {

            /**
			 * 
			 */
			private static final long serialVersionUID = -3591420776536183583L;

			public double f(double... x)
            {
                return f(DenseVector.toDenseVec(x));
            }

            public double f(Vec x)
            {
                double a = x.get(0);
                double constantTerm = lnGamma(a*dim);
                constantTerm -= lnGamma(a)*dim;
                
                double sum = 0.0;
                for(int i = 0; i < dataSet.size(); i++)
                {
                    Vec s = dataSet.get(i);
                    for(int j = 0; j < s.length(); j++)
                        sum += log(s.get(j))*(a-1.0);
                }
                
                return -(sum+constantTerm*dataSet.size());
            }
        };
        NelderMead optimize = new NelderMead();
        Vec guess = new DenseVector(1);
        List<Vec> guesses = new ArrayList<Vec>();
        guesses.add(guess.add(1.0));
        guesses.add(guess.add(0.1));
        guesses.add(guess.add(10.0));
        this.alpha = optimize.optimize(1e-10, 100, logLike, guesses).get(0);
        return true;
    }

    public boolean setUsingDataList(final List<DataPoint> dataPoint)
    {
        Function logLike = new Function() 
        {

            /**
			 * 
			 */
			private static final long serialVersionUID = -1145407955317879017L;

			public double f(double... x)
            {
                return f(DenseVector.toDenseVec(x));
            }

            public double f(Vec x)
            {
                double a = x.get(0);
                double constantTerm = lnGamma(a*dim);
                constantTerm -= lnGamma(a)*dim;
                double weightSum = 0.0;
                
                double sum = 0.0;
                for(int i = 0; i < dataPoint.size(); i++)
                {
                    DataPoint dp = dataPoint.get(i);
                    weightSum += dp.getWeight();
                    Vec s = dp.getNumericalValues();
                    for(int j = 0; j < s.length(); j++)
                        sum += log(s.get(j))*(a-1.0)*dp.getWeight();
                }
                
                return -(sum+constantTerm*weightSum);
            }
        };
        NelderMead optimize = new NelderMead();
        Vec guess = new DenseVector(1);
        List<Vec> guesses = new ArrayList<Vec>();
        guesses.add(guess.add(1.0));
        guesses.add(guess.add(0.1));
        guesses.add(guess.add(10.0));
        this.alpha = optimize.optimize(1e-10, 100, logLike, guesses).get(0);
        return true;
    }
    
    public List<Vec> sample(int count, Random rand)
    {
        List<Vec> samples = new ArrayList<Vec>(count);
        
        double[] gammaSamples = new Gamma(alpha, 1.0).sample(count*dim, rand);
        int samplePos = 0;
        for(int i = 0; i < count; i++)
        {
            Vec sample = new DenseVector(dim);
            for(int j = 0; j < dim; j++)
                sample.set(j, gammaSamples[samplePos++]);
            sample.mutableDivide(sample.sum());
            samples.add(sample);
        }
        
        return samples;
    }
    
}
