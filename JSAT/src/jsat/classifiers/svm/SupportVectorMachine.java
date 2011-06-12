
package jsat.classifiers.svm;

import jsat.classifiers.Classifier;
import jsat.distributions.kernels.KernelTrick;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public abstract class SupportVectorMachine implements Classifier
{
    private KernelTrick kernel;
    protected Vec[] vecs;
    private CacheMode cacheMode;
    
    private double[][] fullCache;
    
    public enum CacheMode {NONE, FULL};

    public SupportVectorMachine(KernelTrick kernel, CacheMode cacheMode)
    {
        this.cacheMode = cacheMode;
        this.kernel = kernel;
    }
    
    public void setKernel(KernelTrick kernel)
    {
        this.kernel = kernel;
    }

    public CacheMode getCacheMode()
    {
        return cacheMode;
    }

    public void setCacheMode(CacheMode cacheMode)
    {
        this.cacheMode = cacheMode;
        
        if(cacheMode == CacheMode.FULL && vecs != null)
        {
            fullCache = new double[vecs.length][];
            for(int i = 0; i < vecs.length; i++)
                fullCache[i] = new double[vecs.length-i];
            
            for(int i = 0; i < vecs.length; i++)
                for(int j = i; j < vecs.length; j++)
                    fullCache[i][j-i] = kEval(vecs[i], vecs[j]);
        }
        else if(cacheMode == CacheMode.NONE)
            fullCache = null;
    }

    
    
    public KernelTrick getKernel()
    {
        return kernel;
    }
    
    protected double kEval(Vec a, Vec b)
    {
        return kernel.eval(a, b);
    }
    
    protected double kEval(int a, int b)
    {
        if(cacheMode == CacheMode.FULL)
        {
            if(a > b)
            {
                int tmp = a;
                a = b;
                b = tmp;
            }
            
            return fullCache[a][b-a];
        }
        return kernel.eval(vecs[a], vecs[b]);
    }
}
