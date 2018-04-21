package jsat.linear.vectorcollection.lsh;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import jsat.linear.*;
import jsat.linear.distancemetrics.CosineDistance;
import jsat.linear.distancemetrics.CosineDistanceNormalized;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.utils.BoundedSortedList;
import jsat.utils.IndexTable;
import jsat.utils.ProbailityMatch;
import jsat.utils.random.RandomUtil;

/**
 * An implementation of Locality Sensitive Hashing for the 
 * {@link CosineDistance} using random projections. This forms a vector 
 * collection that performs a linear search of the data set, but does so in a 
 * more efficient manner by comparing hamming distances in a byte array. 
 * However, the reported distances are only approximations - and may not be 
 * correct. For this reason the results are also approximate.
 * <br><br>
 * See:<br>
 * <ul>
 * <li>Charikar, M. S. (2002). <i>Similarity estimation techniques from rounding
 * algorithms</i>. Proceedings of the thiry-fourth annual ACM symposium on 
 * Theory of computing - STOC  ’02 (pp. 380–388). New York, New York, USA: 
 * ACM Press. doi:10.1145/509907.509965</li>
 * <li>Durme, B. Van,&amp;Lall, A. (2010). <i>Online Generation of Locality 
 * Sensitive Hash Signatures</i>. Proceedings of the ACL 2010 Conference Short 
 * Papers (pp. 231–235). Stroudsburg, PA, USA.</li>
 * </ul>
 * 
 * @author Edward Raff
 */
public class RandomProjectionLSH<V extends Vec> implements VectorCollection<V>
{

    private static final long serialVersionUID = -2042964665052386855L;
    private static final int NO_POOL = -1;
    private Matrix randProjMatrix;
    private int[] projections;
    private int slotsPerEntry;
    private List<V> vecs;
    
    /*
     * Implemtation note: store an integer for the bits, but if we use a small 
     * 64 bit encoding, we waste anotehr 32-64 bits on object overhead, so store
     * it all in one big array isntead of individual ones. Haming distance is 
     * the number of bit differences, and sinx XOR results in a 1 bit only if 
     * the bits are not the same, we can use the bit count of the XORed value to
     * count the hamming distance. 
     */
    
    private ThreadLocal<Vec> tempVecs;
    
    /**
     * Creates a new Random Projection LSH object that uses a full matrix of 
     * normally distributed values. 
     * 
     * @param vecs the list of vectors to form a collection for
     * @param ints the number of integers to use for the encoding
     * @param inMemory {@code true} to construct the full matrix in memory, or 
     * {@code false} to construct the needed values on demand. This reduces 
     * memory use at increased CPU usage. 
     */
    public RandomProjectionLSH(List<V> vecs, int ints, boolean inMemory)
    {
        randProjMatrix = new NormalMatrix(ints*Integer.SIZE, vecs.get(0).length(), NO_POOL);
        if(inMemory)
        {
            Matrix dense = new DenseMatrix(randProjMatrix.rows(), randProjMatrix.cols());
            dense.mutableAdd(randProjMatrix);
            randProjMatrix = dense;
        }
        build(true, vecs, new CosineDistance());
    }

    /**
     * Creates a new Random Projection LSH object that uses a pool of normally
     * distributed values to approximate a full matrix with considerably less
     * memory storage. 
     * 
     * @param vecs the list of vectors to form a collection for
     * @param ints the number of integers to use for the encoding
     * @param poolSize the number of normally distributed random variables to 
     * store. Matrix values will be pulled on demand from an index in the pool
     * of values. 
     */
    public RandomProjectionLSH(List<V> vecs, int ints, int poolSize)
    {
        randProjMatrix = new NormalMatrix(ints*Integer.SIZE, vecs.get(0).length(), poolSize);
        build(true, vecs, new CosineDistance());
    }
    
    /**
     * Copy Constructor
     * @param toCopy the object to copy
     */
    protected RandomProjectionLSH(RandomProjectionLSH<V> toCopy)
    {
        this.randProjMatrix = toCopy.randProjMatrix.clone();
        this.projections = Arrays.copyOf(toCopy.projections, toCopy.projections.length);
        this.slotsPerEntry = toCopy.slotsPerEntry;
        this.vecs = new ArrayList<>(toCopy.vecs);
        
        this.tempVecs = new ThreadLocal<Vec>()
        {

            @Override
            protected Vec initialValue()
            {
                return new DenseVector(randProjMatrix.rows());
            }
        };
    }
    
    @Override
    public List<Double> getAccelerationCache()
    {
        return null;
    }

    @Override
    public void build(boolean parallel, List<V> collection, DistanceMetric dm)
    {
        setDistanceMetric(dm);
        this.vecs = new ArrayList<>(collection);
        tempVecs = ThreadLocal.withInitial(()->new DenseVector(randProjMatrix.rows()));
                
        slotsPerEntry = randProjMatrix.rows()/Integer.SIZE;
        
        projections = new int[slotsPerEntry*vecs.size()];
        
        
        Vec projected = tempVecs.get();
        
        for(int slot = 0; slot < vecs.size(); slot++)
        {
            projected.zeroOut();
            projectVector(vecs.get(slot), slot*slotsPerEntry, projections, projected);
        }
        
    }

    @Override
    public void search(Vec query, double range, List<Integer> neighbors, List<Double> distances)
    {
        int minHammingDist = (int) cosineToHamming(CosineDistance.distanceToCosine(range));
        
        final int[] queryProj = new int[slotsPerEntry];
        Vec tmpSapce = tempVecs.get();
        tmpSapce.zeroOut();
        projectVector(query, 0, queryProj, tmpSapce);
                
        for(int slot = 0; slot < vecs.size(); slot++)
        {
            int hamming = 0;
            int pos = 0;
            while(pos < slotsPerEntry)
                hamming += Integer.bitCount(projections[slot*slotsPerEntry+pos]^queryProj[pos++]);
            
            if(hamming <= minHammingDist)
            {
                neighbors.add(slot);
                distances.add(CosineDistance.cosineToDistance(hammingToCosine(hamming)));
            }
        }
        
        IndexTable it = new IndexTable(distances);
        it.apply(neighbors);
        it.apply(distances);
    }

    @Override
    public void search(Vec query, int numNeighbors, List<Integer> neighbors, List<Double> distances)
    {
        BoundedSortedList<ProbailityMatch<Integer>> toRet = new BoundedSortedList<>(numNeighbors);
        
        final int[] queryProj = new int[slotsPerEntry];
        Vec tmpSapce = tempVecs.get();
        tmpSapce.zeroOut();
        projectVector(query, 0, queryProj, tmpSapce);
                
        for(int slot = 0; slot < vecs.size(); slot++)
        {
            int hamming = 0;
            int pos = 0;
            while(pos < slotsPerEntry)
                hamming += Integer.bitCount(projections[slot*slotsPerEntry+pos]^queryProj[pos++]);
            
            if(toRet.size() < numNeighbors || hamming < toRet.last().getProbability())
                toRet.add(new ProbailityMatch<>(hamming, slot));
        }
        
        //now conver the hamming values to distance values
        for(int i = 0; i < toRet.size(); i++)
        {
            neighbors.add(toRet.get(i).getMatch());
            distances.add(CosineDistance.cosineToDistance(hammingToCosine(toRet.get(i).getProbability())));
        }
    }
    
    /**
     * Returns the signature or encoding length in bits. 
     * @return the signature length in bits
     */
    public int getSignatureBitLength()
    {
        return randProjMatrix.rows()*Integer.SIZE;
    }

    /**
     * Projects a given vector into the array of integers.
     * 
     * @param vecs the vector to project
     * @param slot the index into the array to start placing the bit values
     * @param projected a vector full of zeros of the same length as 
     * {@link #getSignatureBitLength() } to use as a temp space. 
     */
    private void projectVector(Vec vec, int slot, int[] projLocation, Vec projected)
    {
        randProjMatrix.multiply(vec, 1.0, projected);
        int pos = 0;
        int bitsLeft = Integer.SIZE;
        int curVal = 0;
        
        while(pos < slotsPerEntry)
        {
            while(bitsLeft > 0)
            {
                curVal <<= 1;
                if(projected.get(pos*Integer.SIZE+(Integer.SIZE-bitsLeft)) >= 0)
                    curVal |= 1;
                bitsLeft--;
            }
            projLocation[slot+pos] = curVal;
            curVal = 0;
            bitsLeft = Integer.SIZE;
            pos++;
        }
    }

    @Override
    public int size()
    {
        return vecs.size();
    }

    @Override
    public V get(int indx)
    {
        return vecs.get(indx);
    }

    @Override
    public VectorCollection<V> clone()
    {
        return new RandomProjectionLSH<>(this);
    }

    /**
     * Matrix of random normal N(0, 1) values
     */
    private static final class NormalMatrix extends RandomMatrix
    {

        private static final long serialVersionUID = -5274754647385324984L;
        private final double[] pool;
        private final long seedMult;

        public NormalMatrix(int rows, int cols, int poolSize)
        {
            super(rows, cols);
            if(poolSize > 0)
            {
                pool = new double[poolSize];
                Random rand = RandomUtil.getRandom();
                for(int i = 0; i < pool.length; i++)
                    pool[i] = rand.nextGaussian();
            }
            else
                pool = null;
            seedMult = RandomUtil.getRandom().nextLong();
        }

        public NormalMatrix(NormalMatrix toCopy)
        {
            super(toCopy);
            if(toCopy.pool == null)
                this.pool = null;
            else
                this.pool = Arrays.copyOf(toCopy.pool, toCopy.pool.length);
            seedMult = toCopy.seedMult;
        }

        @Override
        public double get(int i, int j)
        {
            if(pool == null)
                return super.get(i, j); 
            else
            {
                long index = ((i+1)*(j+cols())*seedMult) & Integer.MAX_VALUE;
                return pool[(int)index % pool.length];
            }
        }

        @Override
        protected double getVal(Random rand)
        {
            if(pool == null)
                return rand.nextGaussian();
            else
                return pool[rand.nextInt(pool.length)];
        }

        @Override
        public Matrix clone()
        {
            return new NormalMatrix(this);
        }
    }
    
    private double hammingToCosine(double ham)
    {
        return Math.cos(ham*Math.PI/randProjMatrix.rows());
    }
    
    private double cosineToHamming(double cos)
    {
        return randProjMatrix.rows()*Math.acos(cos)/Math.PI;
    }

    @Override
    public void setDistanceMetric(DistanceMetric dm)
    {
        if(!(dm instanceof CosineDistance || dm instanceof CosineDistanceNormalized))
                throw new IllegalArgumentException("RandomProjectionLSH is only compatible with the Cosine Distance metric");
    }

    @Override
    public DistanceMetric getDistanceMetric()
    {
        return new CosineDistance();
    }
}
