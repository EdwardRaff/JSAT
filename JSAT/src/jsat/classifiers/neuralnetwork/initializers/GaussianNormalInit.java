package jsat.classifiers.neuralnetwork.initializers;

import java.util.Random;
import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * This object initializes the values of weights by sampling from the zero mean 
 * Gaussian
 * @author Edward Raff
 */
public class GaussianNormalInit implements WeightInitializer, BiastInitializer
{

	private static final long serialVersionUID = -882418891606717433L;
	private double stndDev;

    /**
     * Creates a new GuassianNormalInit object for initializing weights
     * @param stndDev the standard deviation of the distribution to sample from
     */
    public GaussianNormalInit(double stndDev)
    {
        this.stndDev = stndDev;
    }

    /**
     * Sets the standard deviation of the distribution that will be sampled from
     * @param stndDev the standard deviation to use
     */
    public void setStndDev(double stndDev)
    {
        this.stndDev = stndDev;
    }

    /**
     * 
     * @return the standard deviation of the Gaussian that is sampled from
     */
    public double getStndDev()
    {
        return stndDev;
    }

    @Override
    public void init(Matrix w, Random rand)
    {
        for(int i = 0; i < w.rows(); i++)
            for(int j = 0; j < w.cols(); j++)
                w.set(i, j, rand.nextGaussian()*stndDev);
        
    }

    @Override
    public void init(Vec b, int fanIn, Random rand)
    {
        for(int i = 0; i < b.length(); i++)
            b.set(i, rand.nextGaussian()*stndDev);
    }

    @Override
    public GaussianNormalInit clone()
    {
        return new GaussianNormalInit(stndDev);
    }
}
