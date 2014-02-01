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
        if(p <= 0 || Double.isNaN(p))
            throw new IllegalArgumentException("p must be greater than zero, not " + p);
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
    public PNormNormalization clone()
    {
        return new PNormNormalization(p);
    }
    
    /**
     * Factor for producing {@link PNormNormalization} transforms
     */
    static public class PNormNormalizationFactory extends DataTransformFactoryParm
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
        
        /**
         * Copy constructor 
         * @param toCopy the object to copy
         */
        public PNormNormalizationFactory(PNormNormalizationFactory toCopy)
        {
            this.p = p;
        }

        /**
         * Sets the norm that the vector should be normalized by. 
         * @param p the norm to use in (0, Infinity]
         */
        public void setPNorm(double p)
        {
            if(p <= 0 || Double.isNaN(p))
                throw new IllegalArgumentException("p must be greater than zero, not " + p);
            this.p = p;
        }

        /**
         * Returns the p-norm that the vectors will be normalized by
         * @return the p-norm that the vectors will be normalized by
         */
        public double getPNorm()
        {
            return p;
        }

        @Override
        public DataTransform getTransform(DataSet dataset)
        {
            return new PNormNormalization(p);
        }

        @Override
        public PNormNormalizationFactory clone()
        {
            return new PNormNormalizationFactory(this);
        }
        
    }
}
