package jsat;

import java.util.Random;
import jsat.distributions.Normal;
import jsat.linear.DenseVector;

/**
 * Helper class to avoid issues with sampling from the normal distribution when 
 * testing since the normal can have extreme values
 * @author Edward Raff
 */
public class NormalClampedSample extends Normal
{

	private static final long serialVersionUID = 3970933766374506189L;
	double min, max;

    public NormalClampedSample(final double mean, final double stndDev)
    {
        this(mean, stndDev, mean-3*stndDev, mean+3*stndDev);
    }

    public NormalClampedSample(final double mean, final double stndDev, final double min, final double max)
    {
        super(mean, stndDev);
        this.min = Math.min(min, max);
        this.max = Math.max(min, max);
    }

    @Override
    public double invCdf(final double d)
    {
        return Math.max(min, Math.min(max, super.invCdf(d)));
    }
    
    @Override
    public double[] sample(final int numSamples, final Random rand)
    {
        final double[] ret =  super.sample(numSamples, rand); 
        for(int i = 0; i < ret.length; i++) {
          ret[i] = Math.max(min, Math.min(max, ret[i]));
        }
        return ret;
    }

    @Override
    public DenseVector sampleVec(final int numSamples, final Random rand)
    {
        final DenseVector ret =  super.sampleVec(numSamples, rand); 
        for(int i = 0; i < ret.length(); i++) {
          ret.set(i, Math.max(min, Math.min(max, ret.get(i))));
        }
        return ret;
    }
    
}
