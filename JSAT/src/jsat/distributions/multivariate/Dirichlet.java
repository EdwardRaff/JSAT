
package jsat.distributions.multivariate;

import java.util.ArrayList;
import java.util.Random;
import java.util.List;
import jsat.classifiers.DataPoint;
import jsat.distributions.Gamma;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.optimization.NelderMead;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;

/**
 * An implementation of the Dirichlet distribution. The Dirichlet distribution takes a vector of 
 * positive alphas as its argument, which also specifies the dimension of the distribution. The 
 * Dirichlet distribution has a non zero {@link #pdf(jsat.linear.Vec) PDF} only when the input 
 * vector sums to 1.0, and contains no negative or zero values.
 * 
 * @author Edward Raff
 */
public class Dirichlet extends MultivariateDistributionSkeleton
{

	private static final long serialVersionUID = 6229508050763067569L;
	private Vec alphas;

    /**
     * Creates a new Dirichlet distribution. 
     * 
     * @param alphas the positive alpha values for the distribution. The length of the vector indicates the dimension
     * @throws ArithmeticException if any of the alpha values are not positive
     */
    public Dirichlet(final Vec alphas)
    {
        setAlphas(alphas);
    }

    /**
     * Sets the alphas of the distribution. A copy is made, so altering the input does not effect the distribution.
     * @param alphas the parameter values
     * @throws ArithmeticException if any of the alphas are not positive numbers 
     */
    public void setAlphas(final Vec alphas) throws ArithmeticException
    {
        double tmp;
        for(int i = 0; i < alphas.length(); i++) {
          if ((tmp = alphas.get(i)) <= 0 || Double.isNaN(tmp) || Double.isInfinite(tmp)) {
            throw new ArithmeticException("Dirichlet Distribution parameters must be positive, " + tmp + " is invalid");
          }
        }
        this.alphas = alphas.clone();
    }

    /**
     * Returns the backing vector that contains the alphas specifying the current distribution. Mutable operations should not be applied. 
     * @return the alphas that make the current distribution.
     */
    public Vec getAlphas()
    {
        return alphas;
    }
    
    @Override
    public Dirichlet clone()
    {
        return new Dirichlet(alphas);
    }

    @Override
    public double logPdf(final Vec x)
    {
         if(x.length() != alphas.length()) {
           throw new ArithmeticException( alphas.length() + " variable distribution can not awnser a " + x.length() + " dimension variable");
         }
        double logVal = 0;
        double tmp;
        double sum = 0.0;
        for(int i = 0; i < alphas.length(); i++)
        {
            tmp = x.get(i);
            if(tmp <= 0) {//All values must be positive to be possible
              return -Double.MAX_VALUE;
            }
            sum += tmp;
            logVal += log(x.get(i))*(alphas.get(i)-1.0);
        }
        
        if(abs(sum - 1.0) > 1e-14) {//Some wiglle room, but should sum to one
          return -Double.MAX_VALUE;
         }
        
        /**
         * Normalizing constant is defined by 
         * 
         *              N
         *            =====
         *             | |
         *             | |  Gamma/a \
         *             | |       \ i/
         *             | |
         *            i = 1
         * B(alpha) = ---------------
         *                 /  N     \
         *                 |=====   |
         *                 |\       |
         *            Gamma| >    a |
         *                 |/      i|
         *                 |=====   |
         *                 \i = 1   /
         */
        double logNormalizer = 0.0;
        
        for(int i = 0; i < alphas.length(); i++) {
          logNormalizer += lnGamma(alphas.get(i));
         }
        logNormalizer -= lnGamma(alphas.sum());
        
        return logVal - logNormalizer;
    }

  @Override
    public double pdf(final Vec x)
    {
        return exp(logPdf(x));
    }
    
  @Override
    public <V extends Vec> boolean setUsingData(final List<V> dataSet)
    {
        final Function logLike = new Function() 
        {

			private static final long serialVersionUID = -2341982303993570445L;

      @Override
			public double f(final double... x)
            {
                return f(DenseVector.toDenseVec(x));
            }

      @Override
            public double f(final Vec x)
            {
                double constantTerm = lnGamma(x.sum());
                for(int i = 0; i < x.length(); i++) {
                  constantTerm -= lnGamma(x.get(i));
                }
                
                double sum = 0.0;
                for(int i = 0; i < dataSet.size(); i++)
                {
                    final Vec s = dataSet.get(i);
                    for(int j = 0; j < x.length(); j++) {
                      sum += log(s.get(j))*(x.get(j)-1.0);
                    }
                }
                
                return -(sum+constantTerm*dataSet.size());
            }
        };
        final NelderMead optimize = new NelderMead();
        final Vec guess = new DenseVector(dataSet.get(0).length());
        final List<Vec> guesses = new ArrayList<Vec>();
        guesses.add(guess.add(1.0));
        guesses.add(guess.add(0.1));
        guesses.add(guess.add(10.0));
        this.alphas = optimize.optimize(1e-10, 100, logLike, guesses);

        return true;
    }

  @Override
    public boolean setUsingDataList(final List<DataPoint> dataPoint)
    {
        final Function logLike = new Function() 
        {

			private static final long serialVersionUID = 1597787004137999603L;

      @Override
			public double f(final double... x)
            {
                return f(DenseVector.toDenseVec(x));
            }

      @Override
            public double f(final Vec x)
            {
                double constantTerm = lnGamma(x.sum());
                for(int i = 0; i < x.length(); i++) {
                  constantTerm -= lnGamma(x.get(i));
                }
                double weightSum = 0.0;
                
                double sum = 0.0;
                for(int i = 0; i < dataPoint.size(); i++)
                {
                    
                    final DataPoint dp = dataPoint.get(i);
                    final Vec s = dp.getNumericalValues();
                    weightSum += dp.getWeight();
                    for(int j = 0; j < x.length(); j++) {
                      sum += log(s.get(j))*(x.get(j)-1.0)*dp.getWeight();
                    }
                }
                
                return -(sum+constantTerm*weightSum);
            }
        };
        final NelderMead optimize = new NelderMead();
        final Vec guess = new DenseVector(dataPoint.get(0).numNumericalValues());
        final List<Vec> guesses = new ArrayList<Vec>();
        guesses.add(guess.add(1.0));
        guesses.add(guess.add(0.1));
        guesses.add(guess.add(10.0));
        this.alphas = optimize.optimize(1e-10, 100, logLike, guesses);

        return true;
    }
    
  @Override
    public List<Vec> sample(final int count, final Random rand)
    {
        final List<Vec> samples = new ArrayList<Vec>(count);
        
        final double[][] gammaSamples = new double[alphas.length()][];
        for(int i = 0; i < gammaSamples.length; i++)
        {
            final Gamma gamma = new Gamma(alphas.get(i), 1.0);
            gammaSamples[i] = gamma.sample(count, rand);
        }
        
        for(int i = 0; i < count; i++)
        {
            final Vec sample = new DenseVector(alphas.length());
            for(int j = 0; j < alphas.length(); j++) {
              sample.set(j, gammaSamples[j][i]);
            }
            sample.mutableDivide(sample.sum());
            samples.add(sample);
        }
        
        return samples;
    }
}
