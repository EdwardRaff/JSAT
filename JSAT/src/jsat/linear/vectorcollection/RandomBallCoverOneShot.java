package jsat.linear.vectorcollection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.BoundedSortedList;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.IndexTable;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.concurrent.ParallelUtils;

/**
 * An implementation of the on shot search for the Random Ball Cover algorithm. 
 * Unlike most algorithms, it attempts to satisfy queries in <i>O(sqrt(n))</i> 
 * time. It does this to be more efficient in its computation and easily 
 * parallelizable. Construction time is <i>O(n<sup>3/2</sup>)</i>. <br>
 * The one shot algorithm is an approximate nearest neighbor search, and returns
 * the correct nearest neighbor with a certain probability. If an incorrect 
 * neighbor is found, it's distance from the true nearest neighbor is bounded.  
 * <br>
 * The RBC algorithm was not originally developed for range queries. While the 
 * exact RBC version can perform efficient range queries, the one-shot version 
 * is more likely to produce different results, potentially missing a large 
 * portion of the vectors that should have been included.
 * <br><br>
 * See: Cayton, L. (2012). <i>Accelerating Nearest Neighbor Search on Manycore 
 * Systems</i>. 2012 IEEE 26th International Parallel and Distributed Processing
 * Symposium, 402–413. doi:10.1109/IPDPS.2012.45
 * 
 * @author Edward Raff
 */
public class RandomBallCoverOneShot<V extends Vec> implements VectorCollection<V>
{

    private static final long serialVersionUID = -2562499883847452797L;
    private DistanceMetric dm;
    private List<List<Integer>> ownedVecs;
    private List<Integer> R;
    private List<V> allVecs;
    private List<Double> distCache;
    
    /**
     * The number of points each representative will consider
     */
    private int s;
    
    /**
     * Distance from representative i to its farthest neighbor it owns
     */
    double[] repRadius;

    /**
     * Creates a new one-shot version of the Random Cover Ball. 
     * @param vecs the vectors to place into the RBC
     * @param dm the distance metric to use
     * @param s the number of points to be claimed by each representative. 
     * @param parallel {@code true} if construction should be done in parallel,
     * {@code false} for single threaded.
     */
    public RandomBallCoverOneShot(List<V> vecs, DistanceMetric dm, int s, boolean parallel)
    {
        this.s = s;
        build(parallel, vecs, dm);
    }
    
    /**
     * Creates a new one-shot version of the Random Cover Ball. 
     * @param vecs the vectors to place into the RBC
     * @param dm the distance metric to use
     * @param parallel {@code true} if construction should be done in parallel,
     * {@code false} for single threaded.
     */
    public RandomBallCoverOneShot(List<V> vecs, DistanceMetric dm, boolean parallel)
    {
        this(vecs, dm, (int)Math.sqrt(vecs.size()), parallel);
    }
    
    /**
     * Creates a new one-shot version of the Random Cover Ball. 
     * @param vecs the vectors to place into the RBC
     * @param dm the distance metric to use
     * @param s the number of points to be claimed by each representative. 
     */
    public RandomBallCoverOneShot(List<V> vecs, DistanceMetric dm, int s)
    {
        this(vecs, dm, s, false);
    }
    
    /**
     * Creates a new one-shot version of the Random Cover Ball. 
     * @param vecs the vectors to place into the RBC
     * @param dm the distance metric to use
     */
    public RandomBallCoverOneShot(List<V> vecs, DistanceMetric dm)
    {
        this(vecs, dm, (int)Math.sqrt(vecs.size()));
    }

    public RandomBallCoverOneShot()
    {
        this.dm = new EuclideanDistance();
        this.s = -1;
    }

    /**
     * Copy constructor
     * @param other the RandomBallCover to create a copy of
     */
    private RandomBallCoverOneShot(RandomBallCoverOneShot<V> other)
    {
        this.dm = other.dm.clone();
        this.ownedVecs = new ArrayList<>(other.ownedVecs.size());
        for(int i = 0; i < other.ownedVecs.size(); i++)
        {
            this.ownedVecs.add(new IntList(other.ownedVecs.get(i)));
        }
        this.R = new IntList(other.R);
        this.repRadius = Arrays.copyOf(other.repRadius, other.repRadius.length);
        this.s = other.s;
        if(other.distCache != null)
            this.distCache = new DoubleList(other.distCache);
        if(other.allVecs != null)
            this.allVecs = new ArrayList<>(other.allVecs);
    }

    @Override
    public void build(boolean parallel, List<V> collection, DistanceMetric dm)
    {
        this.allVecs = new ArrayList<>(collection);
        distCache = dm.getAccelerationCache(collection, parallel);
        IntList allIndices = new IntList(allVecs.size());
        ListUtils.addRange(allIndices, 0, allVecs.size(), 1);
        if(s < 0)
            s = (int) Math.sqrt(allVecs.size());
        
        setUp(allIndices, parallel);
    }

    private void setUp(List<Integer> allIndices, boolean parallel) 
    {
        int repCount = (int) Math.max(1, Math.sqrt(allIndices.size()));
        Collections.shuffle(allIndices);
        
        R = allIndices.subList(0, repCount);
        repRadius = new double[R.size()];
        final List<Integer> allRemainingVecs = allIndices.subList(repCount, allIndices.size());
        ownedVecs = new ArrayList<>(repCount);
        

        for (int i = 0; i < repCount; i++)
        {
            ownedVecs.add(new IntList(s));
        }

        ParallelUtils.run(parallel, R.size(), (i)->
        {
            final int Ri = R.get(i);
            final List<Integer> ROwned = ownedVecs.get(i);
            BoundedSortedList<IndexDistPair> nearest = new BoundedSortedList<>(s);
            for(int v : allRemainingVecs)
                nearest.add(new IndexDistPair(v, dm.dist(v, Ri, allVecs, distCache)));
            for(IndexDistPair pmv : nearest)
                ROwned.add(pmv.getIndex());
            
        });
        
    }
    
    @Override
    public List<Double> getAccelerationCache()
    {
        return distCache;
    }

    @Override
    public void search(Vec query, double range, List<Integer> neighbors, List<Double> distances)
    {
        neighbors.clear();
        distances.clear();

        List<Double> qi = dm.getQueryInfo(query);
        //Find the best representative r_q
        double tmp;
        double bestDist = Double.POSITIVE_INFINITY;
        int bestRep = 0;
        for (int i = 0; i < R.size(); i++)
        {
            if ((tmp = dm.dist(R.get(i), query, qi, allVecs, distCache)  ) < bestDist)
            {
                bestRep = i;
                bestDist = tmp;
            }
            
            if(tmp <= range)
            {
                neighbors.add(R.get(i));
                distances.add(tmp);
            }
        }
        
        for (int v : ownedVecs.get(bestRep))
            if((tmp = dm.dist(v, query, qi, allVecs, distCache) ) <= range)
            {
                neighbors.add(v);
                distances.add(tmp);
            }
        
        IndexTable it = new IndexTable(distances);
        it.apply(neighbors);
        it.apply(distances);
    }

    @Override
    public void search(Vec query, int numNeighbors, List<Integer> neighbors, List<Double> distances)
    {
        neighbors.clear();
        distances.clear();
        
        BoundedSortedList<IndexDistPair> knn =
                new BoundedSortedList<>(numNeighbors);

        List<Double> qi = dm.getQueryInfo(query);
        //Find the best representative r_q
        double tmp;
        double bestDist = Double.POSITIVE_INFINITY;
        int bestRep = 0;
        for (int i = 0; i < R.size(); i++)
            if ((tmp = dm.dist(R.get(i), query, qi, allVecs, distCache) ) < bestDist)
            {
                bestRep = i;
                bestDist = tmp;
            }
        knn.add(new IndexDistPair(R.get(bestRep), bestDist));

        for (int v : ownedVecs.get(bestRep))
            knn.add(new IndexDistPair(v, dm.dist(v, query, qi, allVecs, distCache)));

        for(IndexDistPair v : knn)
        {
            neighbors.add(v.getIndex());
            distances.add(v.getDist());
        }
    }

    @Override
    public int size()
    {
        return R.size()*s;
    }

    @Override
    public V get(int indx)
    {
        return allVecs.get(indx);
    }

    @Override
    public RandomBallCoverOneShot<V> clone()
    {
        return new RandomBallCoverOneShot<>(this);
    }

    @Override
    public void setDistanceMetric(DistanceMetric dm)
    {
        this.dm = dm;
    }

    @Override
    public DistanceMetric getDistanceMetric()
    {
        return dm;
    }
}
