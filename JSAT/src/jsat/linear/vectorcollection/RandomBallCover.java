package jsat.linear.vectorcollection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.exceptions.FailedToFitException;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.VecPairedComparable;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.utils.*;
import static jsat.utils.SystemInfo.LogicalCores;
import static java.lang.Math.*;

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
public class RandomBallCover<V extends Vec> implements IncrementalCollection<V>
{

    private static final long serialVersionUID = 2437771973228849200L;
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
                throw new FailedToFitException(ex1);
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
    
    public RandomBallCover(DistanceMetric dm)
    {
        this.dm = dm;
        this.size = 0;
        this.allVecs = new ArrayList<V>();
        if(dm.supportsAcceleration())
            this.distCache = new DoubleList();
        this.R = new IntList();
    }

    /**
     * Copy constructor
     * @param other the RandomBallCover to create a copy of
     */
    private RandomBallCover(RandomBallCover<V> other)
    {
        this.dm = other.dm.clone();
        this.size = other.size;
        if(other.allVecs != null)
            this.allVecs = new ArrayList<V>(other.allVecs);
        if(other.distCache != null)
            this.distCache = new DoubleList(other.distCache);
        if(other.ownedVecs != null)
            this.ownedVecs = new ArrayList<List<Integer>>(other.ownedVecs.size());
        if(other.ownedRDists != null)
            this.ownedRDists = new ArrayList<DoubleList>(other.ownedRDists.size());
        if(other.ownedRDists != null)
            for(int i = 0; i < other.ownedRDists.size(); i++)
            {
                this.ownedRDists.add(new DoubleList(other.ownedRDists.get(i)));
                this.ownedVecs.add(new IntList(other.ownedVecs.get(i)));
            }
        this.R = new IntList(other.R);
        if(other.repRadius != null)
            this.repRadius = Arrays.copyOf(other.repRadius, other.repRadius.length);
    }

    private void setUp(List<Integer> vecIndices, ExecutorService execServ) throws InterruptedException
    {
        int repCount = (int) Math.max(1, Math.sqrt(vecIndices.size()));
        Collections.shuffle(vecIndices);
        
        R = new IntList(vecIndices.subList(0, repCount));
        repRadius = new double[R.size()];
        ownedRDists = new ArrayList<DoubleList>(repRadius.length);
        vecIndices = new IntList(vecIndices.subList(repCount, vecIndices.size()));
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
        
        List<ProbailityMatch<Integer>> results = search_(query, range);
        
        for(ProbailityMatch<Integer> r : results)
            knn.add(new VecPairedComparable<V, Double>(allVecs.get(r.getMatch()), r.getProbability()));
        
        return knn;
    }
    
    /**
     * Performs a range search but returns the integers indecies of the vectors, rather than the vector objects
     * @param query
     * @param range 
     */
    private List<ProbailityMatch<Integer>> search_(Vec query, double range)
    {
        List<ProbailityMatch<Integer>> knn = new ArrayList<ProbailityMatch<Integer>>();
        
        List<Double> qi = dm.getQueryInfo(query);
        
        if(repRadius == null)//brute force search b/c small collection
        {
            for(int i = 0; i < allVecs.size(); i++)
                knn.add(new ProbailityMatch<Integer>(dm.dist(i, query, qi, allVecs, distCache), i));
            return knn;
        }

        //Find the best representative r_q, and add its owned children to knn list. 
        double[] queryRDists = new double[R.size()];

        for (int i = 0; i < R.size(); i++)
            if ((queryRDists[i] = dm.dist(R.get(i), query, qi, allVecs, distCache)) <= range)
                knn.add(new ProbailityMatch<Integer>(queryRDists[i], R.get(i)));

        IndexTable sorted = new IndexTable(queryRDists);
        //k-nn search through the rest of the data set
        for (int i_indx = 0; i_indx < R.size(); i_indx++)
        {
            int i = sorted.index(i_indx);
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
                if ((dist = dm.dist(ownedVecs.get(i).get(j), query, qi, allVecs, distCache)) <= range)
                    knn.add(new ProbailityMatch<Integer>(dist, ownedVecs.get(i).get(j)));
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
        
        if(repRadius == null)//brute force search b/c small collection
        {
            for(int i = 0; i < allVecs.size(); i++)
                knn.add(new VecPairedComparable<V, Double>(allVecs.get(i),
                        dm.dist(i, query, qi, allVecs, distCache)));
            return knn;
        }

        //Find the best representative r_q, and add its owned children to knn list. 
        double[] queryRDists = new double[R.size()];
        Arrays.fill(queryRDists, Double.MAX_VALUE);
        int bestRep = 0;
        for (int i = 0; i < R.size(); i++)
            if ((queryRDists[i] = dm.dist(R.get(i), query, qi, allVecs, distCache)) < queryRDists[bestRep])
                bestRep = i;
        //Other cluster reps R will get a chance to be added to the list later
        knn.add(new VecPairedComparable<V, Double>(allVecs.get(R.get(bestRep)), queryRDists[bestRep]));
        
        //need k'th nearest representative R for bounds check
        IndexTable it = new IndexTable(queryRDists);
        int kth_best_rept;
        if(neighbors < R.size())//need teh k'th closest, but if less than K we can't sure that bound
            kth_best_rept = it.index(neighbors-1);
        else//You are asking for too many neighbors, we can't use the 2nd bound
            kth_best_rept = -1;//if somone uses this we will get an IndexOutOfBound, telling us about the bug! 

        for (int v : ownedVecs.get(bestRep))
            knn.add(new VecPairedComparable<V, Double>(allVecs.get(v), dm.dist(v, query, qi, allVecs, distCache)));
        
        //k-nn search through the rest of the data set
        for (int sorted_order = 1; sorted_order < R.size(); sorted_order++)
        {//start at 1 b/c we brute forced the closest rep first
            final int i = it.index(sorted_order);

            if(knn.size() == neighbors)//no prunnig until we reach k-nns
            {
                //Prune out representatives that are just too far
                if (queryRDists[i] > knn.last().getPair() + repRadius[i])
                    continue;
                //check to make sure we can use this bound before attempting
                else if (kth_best_rept >= 0 && queryRDists[i] > 3 * queryRDists[kth_best_rept])
                    continue;
            }

            //Add any new nn imediatly, hopefully shrinking the bound before
            //the next representative is tested
            knn.add(new VecPairedComparable<V, Double>(allVecs.get(R.get(i)), queryRDists[i]));
            final List<Integer> L_i_index = ownedVecs.get(i);
            final DoubleList L_i_radius = ownedRDists.get(i);
            for (int j = 0; j < ownedVecs.get(i).size(); j++)
            {
                double rDist = L_i_radius.getD(j);
                //Check the first inequality on a per point basis
                if (knn.size() == neighbors && queryRDists[i] > knn.last().getPair() + rDist)
                    continue;
                int indx = L_i_index.get(j);
                V v = allVecs.get(indx);

                knn.add(new VecPairedComparable<V, Double>(v, dm.dist(indx, query, qi, allVecs, distCache) ));
            }
        }

        return knn;
    }
    
    @Override
    public void insert(V x)
    {
        final int new_indx = allVecs.size();
        allVecs.add(x);
        List<Double> qi = dm.getQueryInfo(x);
        if(distCache != null)
            distCache.addAll(qi);
        size++;
        if(size < 10)//brute force for now
        {
            R.add(new_indx);//use R for brute force
            return;
        }
        else if(repRadius == null)//initial normal build
        {
            try
            {
                R.add(new_indx);
                setUp(new IntList(R), new FakeExecutor());
            }
            catch (InterruptedException ex)
            {
                throw new FailedToFitException(ex);
            }
            return;
        }
        //else, normal addition

        //Find the best representative 
        double[] queryRDists = new double[R.size()];
        Arrays.fill(queryRDists, Double.MAX_VALUE);
        int bestRep = 0;
        for (int i = 0; i < R.size(); i++)
            if ((queryRDists[i] = dm.dist(R.get(i), x, qi, allVecs, distCache)) < queryRDists[bestRep])
                bestRep = i;
        //Add new point and update information
        ownedVecs.get(bestRep).add(new_indx);
        ownedRDists.get(bestRep).add(queryRDists[bestRep]);
        repRadius[bestRep] = Math.max(repRadius[bestRep], queryRDists[bestRep]);
        
        if(pow(ceil(sqrt(size)), 2) != size)
            return;//we are done
        //else, expand R set
        int new_r_vec_indx = -1;
        {//lets randomly sample a point that isn't a rep
            int ran_val = new Random().nextInt(size-R.size()-1);
            int R_pos = 0;
            while(ran_val >= 0)
            {
                if(ran_val >= ownedVecs.get(R_pos).size())
                    ran_val -= ownedVecs.get(R_pos++).size();
                else//found the list to grab from
                {
                    new_r_vec_indx = ownedVecs.get(R_pos).remove(ran_val);
                    ownedRDists.get(R_pos).remove(ran_val);
                    //update radius
                    repRadius[R_pos] = 0;
                    for(double d : ownedRDists.get(R_pos))
                        repRadius[R_pos] = Math.max(repRadius[R_pos], d);
                    //stop loop
                    break;
                    
                }
            }
            
        }
        //We now have a new rep, we need to find the people it will own
        double max_radius = 0;
        for(double d : repRadius)
            max_radius = Math.max(max_radius, d);
        List<ProbailityMatch<Integer>> potentialChildren = search_(allVecs.get(new_r_vec_indx), max_radius);
        //add new R to set after to avoid search issues
        repRadius = Arrays.copyOf(repRadius, repRadius.length+1);
        R.add(new_r_vec_indx);
        ownedRDists.add(new DoubleList());
        ownedVecs.add(new IntList());
        final int r_new = R.size()-1;
        
        /*
         * Existing structure of RBC bookkeping dosn't lend itself to the insertion case. 
         * B/c the R set expansion is rare, we don't modify that internal structure. 
         * Instead, we will create a new temporary strucutre to store things 
         * based on the index of the data point. This gives us easy direct 
         * indexing ability. We then fix-up the RBC structure at the end. 
         */
        int[] whoOwnsMe = new int[allVecs.size()];
        Arrays.fill(whoOwnsMe, -1);
        double[] distToMyOwner = new double[allVecs.size()];
        for (int i = 0; i < R.size() - 1; i++)//technicaly this is O(n), but its really fast - so who cares
        {
            List<Integer> L_ry = ownedVecs.get(i);
            for (int j = 0; j < L_ry.size(); j++)
            {
                whoOwnsMe[L_ry.get(j)] = i;
                distToMyOwner[L_ry.get(j)] = ownedRDists.get(i).getD(j);
            }
        }
        boolean[] R_is_dirty = new boolean[R.size()];
        Arrays.fill(R_is_dirty, false);
        R_is_dirty[r_new] = true;
        
        for(ProbailityMatch<Integer> pc : potentialChildren)
        {
            double d_y_r_new = pc.getProbability();
            int y_indx = pc.getMatch();
            //find who owns y_indx
            int r_y = whoOwnsMe[y_indx];
            if(r_y == -1)//Represantative, skip
                continue;
            
            double d_y_ry = distToMyOwner[y_indx];
            
            if(d_y_ry > d_y_r_new)//change ownership
            {
                R_is_dirty[r_y] = true;
                whoOwnsMe[y_indx] = r_new;
                distToMyOwner[y_indx] = d_y_r_new;
            }
        }
        //update representative radi
        for (int r_indx = 0; r_indx < R.size(); r_indx++)
            if(R_is_dirty[r_indx])//clear vecs so we can re-populate
            {
                repRadius[r_indx] = 0;
                ownedRDists.get(r_indx).clear();
                ownedVecs.get(r_indx).clear();
            }
        for(int i = 0; i < whoOwnsMe.length; i++)
        {
            int r_i = whoOwnsMe[i];
            if(r_i == -1)//Represantative, skip
                continue;
            if(R_is_dirty[r_i])
            {
                repRadius[r_i] = Math.max(repRadius[r_i], distToMyOwner[i]);
                ownedRDists.get(r_i).add(distToMyOwner[i]);
                ownedVecs.get(r_i).add(i);
            }
        }
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

        private static final long serialVersionUID = 2707446304590604519L;

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
