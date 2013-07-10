package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.IndexValue;
import jsat.linear.Vec;

/**
 * PNormNormalization transformation performs normalizations of a vector x by 
 * one its p-norms where p is in (0, Infinity)
 * 
 * @author Edward Raff
 */
public class PNormNormalization implements InPlaceTransform
{
    private double p;

    /**
     * Creates a new p norm
     * @param p the norm to use
     */
    public PNormNormalization(double p)
    {
        this.p = p;
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
    public DataTransform clone()
    {
        return new PNormNormalization(p);
    }
    
    static public class PNormNormalizationFactory implements DataTransformFactory
    {
        private double p;

        /**
         * Creates a new p norm factory
         * @param p the norm to use
         */
        public PNormNormalizationFactory(double p)
        {
            this.p = p;
        }

        @Override
        public DataTransform getTransform(DataSet dataset)
        {
            return new PNormNormalization(p);
        }
        
    }
}
