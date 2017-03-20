package jsat.datatransform;

import java.util.Random;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;
import jsat.linear.RandomMatrix;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

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
public class JLTransform extends DataTransformBase
{

    private static final long serialVersionUID = -8621368067861343912L;

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
     * Creates a new JL Transform that uses a target dimension of 50 features.
     * This may not be optimal for any particular dataset.
     *
     * @param k the target dimension size
     */
    public JLTransform()
    {
        this(50);
    }
    
    /**
     * Creates a new JL Transform
     * @param k the target dimension size
     */
    public JLTransform(final int k)
    {
        this(k, TransformMode.SPARSE);
    }
    
    /**
     * Creates a new JL Transform
     * @param k the target dimension size
     * @param mode how to construct the transform
     * @param rand the source of randomness
     */
    public JLTransform(final int k, final TransformMode mode)
    {
        this(k, mode, true);
    }
    
    private int k;
    private boolean inMemory;
    
    /**
     * Creates a new JL Transform
     * @param k the target dimension size
     * @param mode how to construct the transform
     * @param inMemory if {@code false}, the matrix will be stored in O(1) 
     * memory at the cost of execution time. 
     */
    public JLTransform(final int k, final TransformMode mode, boolean inMemory)
    {
        this.mode = mode;
        this.k = k;
        this.inMemory = inMemory;
    }

    @Override
    public void fit(DataSet data)
    {
        final int d = data.getNumNumericalVars();
        Random rand = RandomUtil.getRandom();
        Matrix oldR = R = new RandomMatrixJL(k, d, rand.nextLong(), mode);

        if(inMemory)
        {
            R = new DenseMatrix(k, d);
            R.mutableAdd(oldR);
        }
    }

    /**
     * The JL transform uses a random matrix to project the data, and the mode
     * controls which method is used to construct this matrix.
     *
     * @param mode how to construct the transform
     */
    public void setMode(TransformMode mode)
    {
        this.mode = mode;
    }

    /**
     * 
     * @return how to construct the transform
     */
    public TransformMode getMode()
    {
        return mode;
    }

    /**
     * Sets whether or not the transform matrix is stored explicitly in memory
     * or not. Explicit storage is often faster, but can be prohibitive for
     * large datasets
     * @param inMemory {@code true} to explicitly store the transform matrix,
     * {@code false} to re-create it on the fly as needed
     */
    public void setInMemory(boolean inMemory)
    {
        this.inMemory = inMemory;
    }

    /**
     * 
     * @return {@code true} if this object will explicitly store the transform
     * matrix, {@code false} to re-create it on the fly as needed
     */
    public boolean isInMemory()
    {
        return inMemory;
    }

    /**
     * Sets the target dimension size to use for the output
     * @param k the dimension after apply the transform
     */
    public void setProjectedDimension(int k)
    {
        this.k = k;
    }

    /**
     * 
     * @return the dimension after apply the transform
     */
    public int getProjectedDimension()
    {
        return k;
    }
    
    
    
    public static Distribution guessProjectedDimension(DataSet d)
    {
        //huristic, could be improved by some theory app
        double max = 100;
        double min = 10;
        if(d.getNumNumericalVars() > 10000)
        {
            min = 100;
            max = 1000;
        }
        return new LogUniform(min, max);
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
        private static final long serialVersionUID = 2009377824896155918L;
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
