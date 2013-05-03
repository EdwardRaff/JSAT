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
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.ListUtils;
import static jsat.utils.SystemInfo.LogicalCores;

/**
 * An implementation of the exact search for the Random Ball Cover algorithm. 
 * Unlike most algorithms, it attempts to satisfy queries in <i>O(sqrt(n))</i> 
 * time. It does this to be more efficient in its computation and easily 
 * parallelizable. Construction time is <i>O(n<sup>3/2</sup>)</i>. <br>
 * Unlike the original paper, which assumes single queries will be run in 
 * parallel, the algorithm has been modified to perform additional pruning and 
 * to support range queries. 
 * <br><br>
 * See: Cayton, L. (2012). <i>Accelerating Nearest Neighbor Search on Manycore 
 * Systems</i>. 2012 IEEE 26th International Parallel and Distributed Processing
 * Symposium, 402–413. doi:10.1109/IPDPS.2012.45
 *
 * @author Edward Raff
 */
public class RandomBallCover<V extends Vec> implements VectorCollection<V>
{
    private DistanceMetric dm;
    /**
     * The indices match with their representatives in R
     */
    private List<List<V>> ownedVecs;
    /**
     * The indices match with their representatives in R and in ownedVecs. Each 
     * value indicates the distance of the point to its owner. They are not in 
     * any order
     */
    private List<DoubleList> ownedRDists;
    /**
     * The list of representatives
     */
    private List<V> R;
    private int size;
    
    /**
     * Distance from representative i to its farthest neighbor it owns
     */
    double[] repRadius;

    /**
     * Creates a new Random Ball Cover
     * @param vecs the vectors to place into the RBC
     * @param dm the distance metric to use
     * @param execServ the source of threads for parallel construction
     */
    public RandomBallCover(List<V> vecs, DistanceMetric dm, ExecutorService execServ)
    {
        this.dm = dm;
        this.size = vecs.size();
        try
        {
            setUp(vecs, execServ);
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
     * Creates a new Random Ball Cover
     * @param vecs the vectors to place into the RBC
     * @param dm  the distance metric to use
     */
    public RandomBallCover(List<V> vecs, DistanceMetric dm)
    {
        this(vecs, dm, new FakeExecutor());
    }

    /**
     * Copy constructor
     * @param other the RandomBallCover to create a copy of
     */
    private RandomBallCover(RandomBallCover<V> other)
    {
        this.dm = other.dm.clone();
        this.ownedVecs = new ArrayList<List<V>>(other.ownedVecs.size());
        this.ownedRDists = new ArrayList<DoubleList>(other.ownedRDists.size());
        for(int i = 0; i < other.ownedRDists.size(); i++)
        {
            this.ownedRDists.add(new DoubleList(other.ownedRDists.get(i)));
            this.ownedVecs.add(new ArrayList<V>(other.ownedVecs.get(i)));
        }
        this.R = new ArrayList<V>(other.R);
        this.repRadius = Arrays.copyOf(other.repRadius, other.repRadius.length);
    }

    private void setUp(List<V> allVecs, ExecutorService execServ) throws InterruptedException
    {
        int repCount = (int) Math.max(1, Math.sqrt(allVecs.size()));;
        Collections.shuffle(allVecs);
        
        R = allVecs.subList(0, repCount);
        repRadius = new double[R.size()];
        ownedRDists = new ArrayList<DoubleList>(repRadius.length);
        allVecs = allVecs.subList(repCount, allVecs.size());
        ownedVecs = new ArrayList<List<V>>(repCount);

        for (int i = 0; i < repCount; i++)
        {
            ownedVecs.add(new ArrayList<V>(repCount));
            ownedRDists.add(new DoubleList(repCount));
        }

        final CountDownLatch latch = new CountDownLatch(LogicalCores);
        for (final List<V> subSet : ListUtils.splitList(allVecs, LogicalCores))
            execServ.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    double tmp;
                    for (V v : subSet)
                    {
                        int bestRep = 0;
                        double bestDist = dm.dist(R.get(0), v);
                        for (int potentialRep = 1; potentialRep < R.size(); potentialRep++)
                            if ((tmp = dm.dist(R.get(potentialRep), v)) < bestDist)
                            {
                                bestDist = tmp;
                                bestRep = potentialRep;
                            }

                        synchronized (ownedVecs.get(bestRep))
                        {
                            ownedVecs.get(bestRep).add(v);
                            ownedRDists.get(bestRep).add(bestDist);
                            repRadius[bestRep] = Math.max(repRadius[bestRep], bestDist);
                        }
                    }

                    latch.countDown();
                }
            });

        latch.await();


    }

    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, double range)
    {
        List<VecPairedComparable<V, Double>> knn = new ArrayList<VecPairedComparable<V, Double>>();

        //Find the best representative r_q, and add its owned children to knn list. 
        double[] queryRDists = new double[R.size()];

        for (int i = 0; i < R.size(); i++)
            if ((queryRDists[i] = dm.dist(query, R.get(i))) <= range)
                knn.add(new VecPairedComparable<V, Double>(R.get(i), queryRDists[i]));

        //k-nn search through the rest of the data set
        for (int i = 0; i < R.size(); i++)
        {
            //Prune our representatives that are jsut too far
            if (queryRDists[i] > range + repRadius[i])
                continue;

            //Add any new nn imediatly, hopefully shrinking the bound before
            //the next representative is tested
            double dist;
            for (int j = 0; j < ownedVecs.get(i).size(); j++)
            {
                double rDist = ownedRDists.get(i).getD(j);
                if (queryRDists[i] > range + rDist)//first inqueality ona  per point basis
                    continue;
                V v = ownedVecs.get(i).get(j);
                if ((dist = dm.dist(query, v)) <= range)
                    knn.add(new VecPairedComparable<V, Double>(v, dist));
            }
        }

        Collections.sort(knn);
        return knn;
    }

    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, int neighbors)
    {
        BoundedSortedList<VecPairedComparable<V, Double>> knn = new BoundedSortedList<VecPairedComparable<V, Double>>(neighbors);

        //Find the best representative r_q, and add its owned children to knn list. 
        double[] queryRDists = new double[R.size()];
        int bestRep = 0;
        for (int i = 0; i < R.size(); i++)
            if ((queryRDists[i] = dm.dist(query, R.get(i))) < queryRDists[i])
                bestRep = i;
        knn.add(new VecPairedComparable<V, Double>(R.get(bestRep), queryRDists[bestRep]));

        for (V v : ownedVecs.get(bestRep))
            knn.add(new VecPairedComparable<V, Double>(v, dm.dist(query, v)));

        //k-nn search through the rest of the data set
        for (int i = 0; i < R.size(); i++)
        {
            if (i == bestRep)
                continue;

            //Prune out representatives that are just too far
            if (queryRDists[i] > knn.last().getPair() + repRadius[i])
                continue;
            else if (queryRDists[i] > 3 * queryRDists[bestRep])
                continue;

            //Add any new nn imediatly, hopefully shrinking the bound before
            //the next representative is tested
            knn.add(new VecPairedComparable<V, Double>(R.get(i), queryRDists[i]));
            for (int j = 0; j < ownedVecs.get(i).size(); j++)
            {
                double rDist = ownedRDists.get(i).getD(j);
                //Check the first inequality on a per point basis
                if (queryRDists[i] > knn.last().getPair() + rDist)
                    continue;
                V v = ownedVecs.get(i).get(j);

                knn.add(new VecPairedComparable<V, Double>(v, dm.dist(query, v)));
            }
        }

        return knn;
    }

    @Override
    public int size()
    {
        return size;
    }

    @Override
    public RandomBallCover<V> clone()
    {
        return new RandomBallCover<V>(this);
    }
    
    public static class RandomBallCoverFactory<V extends Vec> implements VectorCollectionFactory<V>
    {

        @Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric)
        {
            return new RandomBallCover<V>(source, distanceMetric);
        }

        @Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric, ExecutorService threadpool)
        {
            return new RandomBallCover<V>(source, distanceMetric, threadpool);
        }

        @Override
        public VectorCollectionFactory<V> clone()
        {
            return new RandomBallCoverFactory<V>();
        }
        
    }
}
