package jsat.datatransform;

import java.util.Random;
import jsat.classifiers.DataPoint;
import jsat.distributions.Normal;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * The Johnson-Lindenstrauss (JL) Transform is a type of random projection down 
 * to a lower dimensional space. The goal is, with a high probability, to keep 
 * the {@link EuclideanDistance  Euclidean distances} between points 
 * approximately the same in the original and projected space. <br>
 * The JL lemma, with a high probability, bounds the error of a distance 
 * computation between two points <i>u</i> and <i>v</i> in the lower dimensional
 * space by (1 &plusmn; &epsilon;) d(<i>u</i>, <i>v</i>)<sup>2</sup>, where d is
 * the Euclidean distance. It works best for very high dimension problems, 1000 
 * or more.
 * <br>
 * For more information see: <br>
 * Achlioptas, D. (2003). <i>Database-friendly random projections: 
 * Johnson-Lindenstrauss with binary coins</i>. Journal of Computer and System 
 * Sciences, 66(4), 671–687. doi:10.1016/S0022-0000(03)00025-4
 * 
 * @author Edward Raff
 */
public class JLTransform implements DataTransform 
{
    //TODO for BINARY, only store the RNG and Seed
    //TODO for SPARSE, avoid unecessary computations for 0 values
    /**
     * Determines which distribution to construct the transform matrix from
     */
    public enum TransformMode
    {
        /**
         * The transform matrix values come from the gaussian distribution and
         * is dense
         */
        GAUSS, 
        /**
         * The transform matrix values are binary and faster to generate.
         */
        BINARY, 
        /**
         * The transform matrix values are sparse. NOTE: this sparsity 
         * is not currently taken advantage of
         */
        SPARSE
    }
    
    private TransformMode mode;
    
    private Matrix R;

    /**
     * Copy constructor
     * @param transform the transform to copy 
     */
    protected JLTransform(JLTransform transform)
    {
        this.mode = transform.mode;
        this.R = transform.R.clone();
    }

    /**
     * Creates a new JL Transform
     * @param k the target dimension size
     * @param d the size of dimension in the original problem space
     * @param mode how to construct the transform
     * @param rand the source of randomness
     */
    public JLTransform(int k, int d, TransformMode mode, Random rand)
    {
        this.mode = mode;
        
        
        R = new DenseMatrix(k, d);
        
        if(mode == TransformMode.GAUSS)
        {
            double cnst = Math.sqrt(k);
            Normal norm = new Normal(0.0, 1.0);
            for(int i = 0; i < R.rows(); i++)
                for(int j = 0; j < R.cols(); j++)
                    R.set(i, j, norm.invCdf(rand.nextDouble())/cnst);
        }
        else if(mode == TransformMode.BINARY)
        {
            double cnst = Math.sqrt(k);
            for(int i = 0; i < R.rows(); i++)
                for(int j = 0; j < R.cols(); j++)
                    R.set(i, j, (rand.nextBoolean() ? -1 : 1) / cnst);
        }
        else if(mode == TransformMode.SPARSE)
        {
            double cnst = Math.sqrt(3)/Math.sqrt(k);
            for(int i = 0; i < R.rows(); i++)
                for(int j = 0; j < R.cols(); j++)
                {
                    double p = rand.nextDouble();
                    if(p <= 2.0/3.0)//0 with prob 2/3
                        continue;
                    //1 with prob 1/6, -1 with prob 1/6
                    R.set(i, j, Math.signum(p-5.0/6.0)*cnst);
                }
        }
        
    }

    @Override
    public DataPoint transform(DataPoint dp)
    {
        Vec newVec = dp.getNumericalValues();
        newVec = R.multiply(newVec);
        
        DataPoint newDP = new DataPoint(newVec, dp.getCategoricalValues(), 
                dp.getCategoricalData(), dp.getWeight());
        
        return newDP;
    }

    @Override
    public DataTransform clone()
    {
        return new JLTransform(this);
    }
    
}
