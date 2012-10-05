package jsat.datatransform;

import jsat.DataSet;
import jsat.linear.*;

/**
 * An extension of {@link WhitenedPCA}, is the Whitened Zero Component Analysis.
 * Whitened ZCA can not project to a lower dimension, as it rotates the output 
 * in the original dimension.
 * 
 * @author Edward Raff
 */
public class WhitenedZCA extends WhitenedPCA
{
    /**
     * Creates a new Whitened ZCA.
     * 
     * @param dataSet the data set to whiten
     * @param regularization the amount of regularization to add, avoids 
     * numerical instability
     */
    public WhitenedZCA(DataSet dataSet, double regularization)
    {
        super(dataSet, regularization);
    }
    
    /**
     * Creates a new Whitened ZCA. The regularization parameter will be
     * chosen as the log of the condition of the covariance. 
     * 
     * @param dataSet the data set to whiten
     */
    public WhitenedZCA(DataSet dataSet)
    {
        super(dataSet);
    }
    

    @Override
    protected void setUpTransform(SingularValueDecomposition svd)
    {
        double[] s = svd.getSingularValues();
        Vec diag = new DenseVector(s.length);

        for(int i = 0; i < s.length; i++)
            diag.set(i, 1.0/Math.sqrt(s[i]+regularization));
        
        Matrix U = svd.getU();
        
        transform = U.multiply(Matrix.diag(diag)).multiply(U.transpose());
    }
    
    public class WhitenedZCATransformFactory implements DataTransformFactory
    {
        private Double reg;

        /**
         * Creates a new WhitenedZCA factory 
         * @param reg the regularization to use
         */
        public WhitenedZCATransformFactory(double reg)
        {
            this.reg = reg;
        }

        /**
         * Creates a new WhitenedZCA
         */
        public WhitenedZCATransformFactory()
        {
            reg = null;
        }
        
        @Override
        public DataTransform getTransform(DataSet dataset)
        {
            if(reg == null)
                return new WhitenedZCA(dataset);
            return new WhitenedZCA(dataset, reg);
        }
        
    }
}
