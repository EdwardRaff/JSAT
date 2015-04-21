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

    public NormalClampedSample(double mean, double stndDev)
    {
        this(mean, stndDev, mean-3*stndDev, mean+3*stndDev);
    }

    public NormalClampedSample(double mean, double stndDev, double min, double max)
    {
        super(mean, stndDev);
        this.min = Math.min(min, max);
        this.max = Math.max(min, max);
    }

    @Override
    public double invCdf(double d)
    {
        return Math.max(min, Math.min(max, super.invCdf(d)));
    }
    
    @Override
    public double[] sample(int numSamples, Random rand)
    {
        double[] ret =  super.sample(numSamples, rand); 
        for(int i = 0; i < ret.length; i++)
            ret[i] = Math.max(min, Math.min(max, ret[i]));
        return ret;
    }

    @Override
    public DenseVector sampleVec(int numSamples, Random rand)
    {
        DenseVector ret =  super.sampleVec(numSamples, rand); 
        for(int i = 0; i < ret.length(); i++)
            ret.set(i, Math.max(min, Math.min(max, ret.get(i))));
        return ret;
    }
    
}
