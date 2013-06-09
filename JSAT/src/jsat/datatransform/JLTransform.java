package jsat.datatransform;

import java.util.Random;
import jsat.classifiers.DataPoint;
import jsat.distributions.Normal;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;
import jsat.linear.RandomMatrix;
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
    //TODO for SPARSE, avoid unecessary computations for 0 values
    /**
     * Determines which distribution to construct the transform matrix from
     */
    public enum TransformMode
    {
        /**
         * The transform matrix values come from the gaussian distribution and
         * is dense <br><br>
         * This transform is expensive to use when not using an in memory matrix
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
    public JLTransform(final int k, final int d, final TransformMode mode, Random rand)
    {
        this(k, d, mode, rand, true);
    }
    
    /**
     * Creates a new JL Transform
     * @param k the target dimension size
     * @param d the size of dimension in the original problem space
     * @param mode how to construct the transform
     * @param rand the source of randomness
     * @param inMemory if {@code false}, the matrix will be stored in O(1) 
     * memory at the cost of execution time. 
     */
    public JLTransform(final int k, final int d, final TransformMode mode, Random rand, boolean inMemory)
    {
        this.mode = mode;
        
        Matrix oldR = R = new RandomMatrixJL(k, d, rand.nextLong(), mode);

        if(inMemory)
        {
            R = new DenseMatrix(k, d);
            R.mutableAdd(oldR);
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
    
    private static class RandomMatrixJL extends RandomMatrix
    {
        private double cnst; 
        private TransformMode mode;
        
        public RandomMatrixJL(int rows, int cols, long XORSeed, TransformMode mode)
        {
            super(rows, cols, XORSeed);
            this.mode = mode;
            int k = rows;
            if (mode == TransformMode.GAUSS || mode == TransformMode.BINARY)
                cnst = 1.0 / Math.sqrt(k);
            else if (mode == TransformMode.SPARSE)
                cnst = Math.sqrt(3) / Math.sqrt(k);
        }
        
        @Override
        protected double getVal(Random rand)
        {
            if (mode == TransformMode.GAUSS)
            {
                return rand.nextGaussian()*cnst;
            }
            else if (mode == TransformMode.BINARY)
            {
                return (rand.nextBoolean() ? -cnst : cnst);
            }
            else if (mode == TransformMode.SPARSE)
            {
                int val = rand.nextInt(6);
                //1 with prob 1/6, -1 with prob 1/6
                if(val == 0)
                    return -cnst;
                else if(val == 1)
                    return cnst;
                else //0 with prob 2/3
                    return 0;
            }
            else
                throw new RuntimeException("BUG: Please report");
        }
        
    }
    
}
