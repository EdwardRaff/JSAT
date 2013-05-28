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
import jsat.utils.IntList;
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
    private List<List<Integer>> ownedVecs;
    /**
     * The indices match with their representatives in R and in ownedVecs. Each 
     * value indicates the distance of the point to its owner. They are not in 
     * any order
     */
    private List<DoubleList> ownedRDists;
    /**
     * The list of representatives
     */
    private List<Integer> R;
    private int size;
    private List<V> allVecs;
    private List<Double> distCache;
    
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
        this.allVecs = new ArrayList<V>(vecs);
        if(execServ instanceof FakeExecutor)
            this.distCache = dm.getAccelerationCache(allVecs);
        else
            this.distCache = dm.getAccelerationCache(allVecs, execServ);
        IntList allIndices = new IntList(vecs.size());
        ListUtils.addRange(allIndices, 0, size, 1);
        try
        {
            setUp(allIndices, execServ);
        }
        catch (InterruptedException ex)
        {
            try
            {
                setUp(allIndices, new FakeExecutor());
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
        this.ownedVecs = new ArrayList<List<Integer>>(other.ownedVecs.size());
        this.ownedRDists = new ArrayList<DoubleList>(other.ownedRDists.size());
        for(int i = 0; i < other.ownedRDists.size(); i++)
        {
            this.ownedRDists.add(new DoubleList(other.ownedRDists.get(i)));
            this.ownedVecs.add(new IntList(other.ownedVecs.get(i)));
        }
        this.R = new IntList(other.R);
        this.repRadius = Arrays.copyOf(other.repRadius, other.repRadius.length);
    }

    private void setUp(List<Integer> vecIndices, ExecutorService execServ) throws InterruptedException
    {
        int repCount = (int) Math.max(1, Math.sqrt(vecIndices.size()));;
        Collections.shuffle(vecIndices);
        
        R = vecIndices.subList(0, repCount);
        repRadius = new double[R.size()];
        ownedRDists = new ArrayList<DoubleList>(repRadius.length);
        vecIndices = vecIndices.subList(repCount, vecIndices.size());
        ownedVecs = new ArrayList<List<Integer>>(repCount);

        for (int i = 0; i < repCount; i++)
        {
            ownedVecs.add(new IntList(repCount));
            ownedRDists.add(new DoubleList(repCount));
        }

        final CountDownLatch latch = new CountDownLatch(LogicalCores);
        for (final List<Integer> subSet : ListUtils.splitList(vecIndices, LogicalCores))
            execServ.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    double tmp;
                    for (int v : subSet)
                    {
                        int bestRep = 0;
                        double bestDist = dm.dist(v, R.get(0), allVecs, distCache);
                        for (int potentialRep = 1; potentialRep < R.size(); potentialRep++)
                            if ((tmp = dm.dist(v, R.get(potentialRep), allVecs, distCache)) < bestDist)
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
        
        List<Double> qi = dm.getQueryInfo(query);

        //Find the best representative r_q, and add its owned children to knn list. 
        double[] queryRDists = new double[R.size()];

        for (int i = 0; i < R.size(); i++)
            if ((queryRDists[i] = dm.dist(R.get(i), query, qi, allVecs, distCache)) <= range)
                knn.add(new VecPairedComparable<V, Double>(allVecs.get(R.get(i)), queryRDists[i]));

        //k-nn search through the rest of the data set
        for (int i = 0; i < R.size(); i++)
        {
            //Prune our representatives that are just too far
            if (queryRDists[i] > range + repRadius[i])
                continue;

            //Add any new nn imediatly, hopefully shrinking the bound before
            //the next representative is tested
            double dist;
            for (int j = 0; j < ownedVecs.get(i).size(); j++)
            {
                double rDist = ownedRDists.get(i).getD(j);
                if (queryRDists[i] > range + rDist)//first inqueality on a per point basis
                    continue;
                V v = allVecs.get(ownedVecs.get(i).get(j));
                if ((dist = dm.dist(ownedVecs.get(i).get(j), query, qi, allVecs, distCache)) <= range)
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
        
        List<Double> qi = dm.getQueryInfo(query);

        //Find the best representative r_q, and add its owned children to knn list. 
        double[] queryRDists = new double[R.size()];
        int bestRep = 0;
        for (int i = 0; i < R.size(); i++)
            if ((queryRDists[i] = dm.dist(R.get(i), query, qi, allVecs, distCache)) < queryRDists[i])
                bestRep = i;
        knn.add(new VecPairedComparable<V, Double>(allVecs.get(R.get(bestRep)), queryRDists[bestRep]));

        for (int v : ownedVecs.get(bestRep))
            knn.add(new VecPairedComparable<V, Double>(allVecs.get(v), dm.dist(v, query, qi, allVecs, distCache)));

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
            knn.add(new VecPairedComparable<V, Double>(allVecs.get(R.get(i)), queryRDists[i]));
            for (int j = 0; j < ownedVecs.get(i).size(); j++)
            {
                double rDist = ownedRDists.get(i).getD(j);
                //Check the first inequality on a per point basis
                if (queryRDists[i] > knn.last().getPair() + rDist)
                    continue;
                int indx = ownedVecs.get(i).get(j);
                V v = allVecs.get(indx);

                knn.add(new VecPairedComparable<V, Double>(v, dm.dist(indx, query, qi, allVecs, distCache) ));
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
