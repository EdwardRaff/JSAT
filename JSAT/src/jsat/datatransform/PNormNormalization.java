package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;

/**
 * PNormNormalization transformation performs normalizations of a vector x by 
 * one its p-norms where p is in (0, Infinity)
 * 
 * @author Edward Raff
 */
public class PNormNormalization implements InPlaceTransform
{

    private static final long serialVersionUID = 2934569881395909607L;
    private double p;

    /**
     * Creates a new object that normalizes based on the 2-norm
     */
    public PNormNormalization()
    {
        this(2.0);
    }
    
    /**
     * Creates a new p norm
     * @param p the norm to use
     */
    public PNormNormalization(double p)
    {
        if(p <= 0 || Double.isNaN(p))
            throw new IllegalArgumentException("p must be greater than zero, not " + p);
        this.p = p;
    }

    @Override
    public void fit(DataSet data)
    {
        //no-op, nothing needs to be done
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        DataPoint dpNew = dp.clone();
        
        mutableTransform(dpNew);
        return dpNew;
    }
        
    @Override
    public void mutableTransform(DataPoint dp)
    {
        Vec vec = dp.getNumericalValues();
        double norm = vec.pNorm(p);
        if(norm != 0)
            vec.mutableDivide(norm);
    }
     
    @Override
    public boolean mutatesNominal()
    {
        return false;
    }

    @Override
    public PNormNormalization clone()
    {
        return new PNormNormalization(p);
    }
}
