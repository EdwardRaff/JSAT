package jsat.datatransform;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Matrix;
import jsat.linear.RandomMatrix;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.IntList;
import jsat.utils.random.RandomUtil;

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

    private static final long serialVersionUID = -8621368067861343913L;

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
         * The transform matrix values are sparse, using the original "Data Base
         * Friendly" transform approach.
         */
        SPARSE,
        /**
         * The transform matrix is sparser, making it faster to apply. For most
         * all datasets should provide results of equal quality to
         * {@link #SPARSE} option while being faster.
         */
        SPARSE_SQRT,
        /**
         * The transform matrix is highly sparse, making it exceptionally fast
         * for larger datasets. However, accuracy may be reduced for some
         * problems.
         */
        SPARSE_LOG
    }
    
    /**
     * This is used to make the Sparse JL option run faster by avoiding FLOPS.
     * <br>
     * There will be one IntList for every feature in the feature set. Each
     * IntList value, abs(j), indicates which of the transformed indecies
     * feature i will contribute value to. The sign of sign(j) indicates if it
     * should be additive or subtractive.
     */
    private List<IntList> sparse_jl_map;
    private double sparse_jl_cnst;
    
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
        this.k = transform.k;
        if(transform.sparse_jl_map != null)
        {
            this.sparse_jl_map = new ArrayList<>(transform.sparse_jl_map.size());
            for(IntList a : transform.sparse_jl_map)
                this.sparse_jl_map.add(new IntList(a));
        }
        this.sparse_jl_cnst = transform.sparse_jl_cnst;
    }

    /**
     * Creates a new JL Transform that uses a target dimension of 50 features.
     * This may not be optimal for any particular dataset.
     *
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
        this(k, TransformMode.SPARSE_SQRT);
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
    
    /**
     * Target dimension size
     */
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

        if(mode == TransformMode.GAUSS)
        {
            if(inMemory)
            {
                R = new DenseMatrix(k, d);
                R.mutableAdd(oldR);
            }
        }
        else//Sparse case! Lets do this smarter
        {
            int s;
            switch(mode)
            {
                case SPARSE_SQRT:
                    s = (int) Math.round(Math.sqrt(d+1));
                    break;
                case SPARSE_LOG:
                    s = (int) Math.round(d/Math.log(d+1));
                    break;
                default://default case, use original SPARSE JL algo
                    s = 3;
            }
            
            sparse_jl_cnst = Math.sqrt(s);
            
            //Lets set up some random mats. 
            sparse_jl_map = new ArrayList<>(d);
            IntList all_embed_dims = IntList.range(0, k);
            int nnz = k/s;
            for(int j = 0; j < d; j++)
            {
                Collections.shuffle(all_embed_dims, rand);
                IntList x_j_map = new IntList(nnz);
                //First 1/(2 s) become the positives
                for(int i = 0; i < nnz; i++)
                    x_j_map.add(i);
                //Second 1/(2 s) become the negatives
                for(int i = nnz/2; i < nnz; i++)
                    x_j_map.add(-i);
                //Sort this after so that the later use of this iteration order is better behaved for CPU cache & prefetching
                Collections.sort(x_j_map, (Integer o1, Integer o2) -> Integer.compare(Math.abs(o1), Math.abs(o2)));
                
                sparse_jl_map.add(x_j_map);
            }
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
        Vec newVec;
        switch(mode)
        {
            case SPARSE:
            case SPARSE_SQRT:
            case SPARSE_LOG:
                //Sparse JL case, do adds and final mul
                newVec = new DenseVector(k);
                
                for(IndexValue iv : dp.getNumericalValues())
                {
                    double x_i = iv.getValue();
                    int i = iv.getIndex();
                    
                    for(int j : sparse_jl_map.get(i))
                    {
                        if(j >= 0)
                            newVec.increment(j, x_i);
                        else
                            newVec.increment(-j, -x_i);
                    }
                    newVec.mutableMultiply(sparse_jl_cnst);
                }
                
                break;
            default://default case, do the explicity mat-mul
                newVec = dp.getNumericalValues();
                newVec = R.multiply(newVec);
        }
        

        DataPoint newDP = new DataPoint(newVec, dp.getCategoricalValues(), 
                dp.getCategoricalData());
        
        return newDP;
    }

    @Override
    public JLTransform clone()
    {
        return new JLTransform(this);
    }
    
    private static class RandomMatrixJL extends RandomMatrix
    {
        private static final long serialVersionUID = 2009377824896155918L;
        public double cnst;
        private TransformMode mode;

        public RandomMatrixJL(RandomMatrixJL toCopy) 
        {
            super(toCopy);
            this.cnst = toCopy.cnst;
            this.mode = toCopy.mode;
        }
        
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

        @Override
        public RandomMatrixJL clone() 
        {
            return new RandomMatrixJL(this);
        }
        
    }
    
}
