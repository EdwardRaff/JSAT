package jsat.linear.vectorcollection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.VecPairedComparable;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.utils.BoundedSortedList;
import jsat.utils.FakeExecutor;
import jsat.utils.ProbailityMatch;

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
    private DistanceMetric dm;
    private List<List<V>> ownedVecs;
    private List<V> R;
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
     * @param execServ the source of threads for parallel construction
     */
    public RandomBallCoverOneShot(List<V> vecs, DistanceMetric dm, int s, ExecutorService execServ)
    {
        this.dm = dm;
        this.s = s;
        try
        {
            setUp(new ArrayList<V>(vecs), execServ);
        }
        catch (InterruptedException ex)
        {
            try
            {
                setUp(vecs, new FakeExecutor());
            }
            catch (InterruptedException ex1)
            {
                //Wont happen with a fake executor, nothing to through the interupted exception in that case
            }
        }
    }
    
    /**
     * Creates a new one-shot version of the Random Cover Ball. 
     * @param vecs the vectors to place into the RBC
     * @param dm the distance metric to use
     * @param execServ the source of threads for parallel construction
     */
    public RandomBallCoverOneShot(List<V> vecs, DistanceMetric dm, ExecutorService execServ)
    {
        this(vecs, dm, (int)Math.sqrt(vecs.size()), execServ);
    }
    
    /**
     * Creates a new one-shot version of the Random Cover Ball. 
     * @param vecs the vectors to place into the RBC
     * @param dm the distance metric to use
     * @param s the number of points to be claimed by each representative. 
     */
    public RandomBallCoverOneShot(List<V> vecs, DistanceMetric dm, int s)
    {
        this(vecs, dm, s, new FakeExecutor());
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

    /**
     * Copy constructor
     * @param other the RandomBallCover to create a copy of
     */
    private RandomBallCoverOneShot(RandomBallCoverOneShot<V> other)
    {
        this.dm = other.dm.clone();
        this.ownedVecs = new ArrayList<List<V>>(other.ownedVecs.size());
        for(int i = 0; i < other.ownedVecs.size(); i++)
        {
            this.ownedVecs.add(new ArrayList<V>(other.ownedVecs.get(i)));
        }
        this.R = new ArrayList<V>(other.R);
        this.repRadius = Arrays.copyOf(other.repRadius, other.repRadius.length);
        this.s = other.s;
    }

    private void setUp(List<V> allVecs, ExecutorService execServ) throws InterruptedException
    {
        int repCount = (int) Math.max(1, Math.sqrt(allVecs.size()));;
        Collections.shuffle(allVecs);
        
        R = allVecs.subList(0, repCount);
        repRadius = new double[R.size()];
        final List<V> allRemainingVecs = allVecs.subList(repCount, allVecs.size());
        ownedVecs = new ArrayList<List<V>>(repCount);
        

        for (int i = 0; i < repCount; i++)
        {
            ownedVecs.add(new ArrayList<V>(s));
        }

        final CountDownLatch latch = new CountDownLatch(R.size());
        for (int i = 0; i < R.size(); i++)
        {
            final V Ri = R.get(i);
            final List<V> ROwned = ownedVecs.get(i);
            execServ.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    BoundedSortedList<ProbailityMatch<V>> nearest =
                            new BoundedSortedList<ProbailityMatch<V>>(s, s);
                    for(V v : allRemainingVecs)
                        nearest.add(new ProbailityMatch<V>(dm.dist(v, Ri), v));
                    
                    for(ProbailityMatch<V> pmv : nearest)
                        ROwned.add(pmv.getMatch());

                    latch.countDown();
                }
            });
        }

        latch.await();


    }

    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, double range)
    {
        List<VecPairedComparable<V, Double>> knn = new ArrayList<VecPairedComparable<V, Double>>();

        //Find the best representative r_q
        double tmp;
        double bestDist = Double.POSITIVE_INFINITY;
        int bestRep = 0;
        for (int i = 0; i < R.size(); i++)
            if ((tmp = dm.dist(query, R.get(i))) < bestDist)
            {
                bestRep = i;
                bestDist = tmp;
            }
        if(bestDist <= range)
            knn.add(new VecPairedComparable<V, Double>(R.get(bestRep), bestDist));

        for (V v : ownedVecs.get(bestRep))
            if((tmp = dm.dist(query, v)) <= range)
            knn.add(new VecPairedComparable<V, Double>(v, tmp));

        Collections.sort(knn);
        return knn;
    }

    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, int neighbors)
    {
        BoundedSortedList<VecPairedComparable<V,Double>> knn =
                new BoundedSortedList<VecPairedComparable<V,Double>>(neighbors);

        //Find the best representative r_q
        double tmp;
        double bestDist = Double.POSITIVE_INFINITY;
        int bestRep = 0;
        for (int i = 0; i < R.size(); i++)
            if ((tmp = dm.dist(query, R.get(i))) < bestDist)
            {
                bestRep = i;
                bestDist = tmp;
            }
        knn.add(new VecPairedComparable<V, Double>(R.get(bestRep), bestDist));

        for (V v : ownedVecs.get(bestRep))
            knn.add(new VecPairedComparable<V, Double>(v, dm.dist(query, v)));


        return knn;
    }

    @Override
    public int size()
    {
        return R.size()*s;
    }

    @Override
    public RandomBallCoverOneShot<V> clone()
    {
        return new RandomBallCoverOneShot<V>(this);
    }
    
    public static class RandomBallCoverOneShotFactory<V extends Vec> implements VectorCollectionFactory<V>
    {

        @Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric)
        {
            return new RandomBallCoverOneShot<V>(source, distanceMetric);
        }

        @Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric, ExecutorService threadpool)
        {
            return new RandomBallCoverOneShot<V>(source, distanceMetric, threadpool);
        }

        @Override
        public VectorCollectionFactory<V> clone()
        {
            return new RandomBallCoverOneShotFactory<V>();
        }
    }
}
