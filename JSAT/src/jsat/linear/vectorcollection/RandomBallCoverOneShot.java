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
 * portion of the vectors that should have been included. <br>
 * <br>
 * See: Cayton, L. (2012). <i>Accelerating Nearest Neighbor Search on Manycore
 * Systems</i>. 2012 IEEE 26th International Parallel and Distributed Processing
 * Symposium, 402–413. doi:10.1109/IPDPS.2012.45
 *
 * @author Edward Raff
 */
public class RandomBallCoverOneShot<V extends Vec> implements VectorCollection<V> {

  public static class RandomBallCoverOneShotFactory<V extends Vec> implements VectorCollectionFactory<V> {

    /**
     *
     */
    private static final long serialVersionUID = 7658115337969827371L;

    @Override
    public VectorCollectionFactory<V> clone() {
      return new RandomBallCoverOneShotFactory<V>();
    }

    @Override
    public VectorCollection<V> getVectorCollection(final List<V> source, final DistanceMetric distanceMetric) {
      return new RandomBallCoverOneShot<V>(source, distanceMetric);
    }

    @Override
    public VectorCollection<V> getVectorCollection(final List<V> source, final DistanceMetric distanceMetric,
        final ExecutorService threadpool) {
      return new RandomBallCoverOneShot<V>(source, distanceMetric, threadpool);
    }
  }

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
   *
   * @param vecs
   *          the vectors to place into the RBC
   * @param dm
   *          the distance metric to use
   */
  public RandomBallCoverOneShot(final List<V> vecs, final DistanceMetric dm) {
    this(vecs, dm, (int) Math.sqrt(vecs.size()));
  }

  /**
   * Creates a new one-shot version of the Random Cover Ball.
   *
   * @param vecs
   *          the vectors to place into the RBC
   * @param dm
   *          the distance metric to use
   * @param execServ
   *          the source of threads for parallel construction
   */
  public RandomBallCoverOneShot(final List<V> vecs, final DistanceMetric dm, final ExecutorService execServ) {
    this(vecs, dm, (int) Math.sqrt(vecs.size()), execServ);
  }

  /**
   * Creates a new one-shot version of the Random Cover Ball.
   *
   * @param vecs
   *          the vectors to place into the RBC
   * @param dm
   *          the distance metric to use
   * @param s
   *          the number of points to be claimed by each representative.
   */
  public RandomBallCoverOneShot(final List<V> vecs, final DistanceMetric dm, final int s) {
    this(vecs, dm, s, new FakeExecutor());
  }

  /**
   * Creates a new one-shot version of the Random Cover Ball.
   *
   * @param vecs
   *          the vectors to place into the RBC
   * @param dm
   *          the distance metric to use
   * @param s
   *          the number of points to be claimed by each representative.
   * @param execServ
   *          the source of threads for parallel construction
   */
  public RandomBallCoverOneShot(final List<V> vecs, final DistanceMetric dm, final int s,
      final ExecutorService execServ) {
    this.dm = dm;
    this.s = s;
    this.allVecs = new ArrayList<V>(vecs);
    if (execServ instanceof FakeExecutor) {
      distCache = dm.getAccelerationCache(allVecs);
    } else {
      distCache = dm.getAccelerationCache(vecs, execServ);
    }
    final IntList allIndices = new IntList(allVecs.size());
    ListUtils.addRange(allIndices, 0, allVecs.size(), 1);
    try {
      setUp(allIndices, execServ);
    } catch (final InterruptedException ex) {
      try {
        setUp(allIndices, new FakeExecutor());
      } catch (final InterruptedException ex1) {
        // Wont happen with a fake executor, nothing to through the interupted
        // exception in that case
      }
    }
  }

  /**
   * Copy constructor
   *
   * @param other
   *          the RandomBallCover to create a copy of
   */
  private RandomBallCoverOneShot(final RandomBallCoverOneShot<V> other) {
    this.dm = other.dm.clone();
    this.ownedVecs = new ArrayList<List<Integer>>(other.ownedVecs.size());
    for (int i = 0; i < other.ownedVecs.size(); i++) {
      this.ownedVecs.add(new IntList(other.ownedVecs.get(i)));
    }
    this.R = new IntList(other.R);
    this.repRadius = Arrays.copyOf(other.repRadius, other.repRadius.length);
    this.s = other.s;
    if (other.distCache != null) {
      this.distCache = new DoubleList(other.distCache);
    }
    if (other.allVecs != null) {
      this.allVecs = new ArrayList<V>(other.allVecs);
    }
  }

  @Override
  public RandomBallCoverOneShot<V> clone() {
    return new RandomBallCoverOneShot<V>(this);
  }

  @Override
  public List<? extends VecPaired<V, Double>> search(final Vec query, final double range) {
    final List<VecPairedComparable<V, Double>> knn = new ArrayList<VecPairedComparable<V, Double>>();

    final List<Double> qi = dm.getQueryInfo(query);
    // Find the best representative r_q
    double tmp;
    double bestDist = Double.POSITIVE_INFINITY;
    int bestRep = 0;
    for (int i = 0; i < R.size(); i++) {
      if ((tmp = dm.dist(R.get(i), query, qi, allVecs, distCache)) < bestDist) {
        bestRep = i;
        bestDist = tmp;
      }
    }
    if (bestDist <= range) {
      knn.add(new VecPairedComparable<V, Double>(allVecs.get(R.get(bestRep)), bestDist));
    }

    for (final int v : ownedVecs.get(bestRep)) {
      if ((tmp = dm.dist(v, query, qi, allVecs, distCache)) <= range) {
        knn.add(new VecPairedComparable<V, Double>(allVecs.get(v), tmp));
      }
    }

    Collections.sort(knn);
    return knn;
  }

  @Override
  public List<? extends VecPaired<V, Double>> search(final Vec query, final int neighbors) {
    final BoundedSortedList<VecPairedComparable<V, Double>> knn = new BoundedSortedList<VecPairedComparable<V, Double>>(
        neighbors);

    final List<Double> qi = dm.getQueryInfo(query);
    // Find the best representative r_q
    double tmp;
    double bestDist = Double.POSITIVE_INFINITY;
    int bestRep = 0;
    for (int i = 0; i < R.size(); i++) {
      if ((tmp = dm.dist(R.get(i), query, qi, allVecs, distCache)) < bestDist) {
        bestRep = i;
        bestDist = tmp;
      }
    }
    knn.add(new VecPairedComparable<V, Double>(allVecs.get(R.get(bestRep)), bestDist));

    for (final int v : ownedVecs.get(bestRep)) {
      knn.add(new VecPairedComparable<V, Double>(allVecs.get(v), dm.dist(v, query, qi, allVecs, distCache)));
    }

    return knn;
  }

  private void setUp(final List<Integer> allIndices, final ExecutorService execServ) throws InterruptedException {
    final int repCount = (int) Math.max(1, Math.sqrt(allIndices.size()));
    ;
    Collections.shuffle(allIndices);

    R = allIndices.subList(0, repCount);
    repRadius = new double[R.size()];
    final List<Integer> allRemainingVecs = allIndices.subList(repCount, allIndices.size());
    ownedVecs = new ArrayList<List<Integer>>(repCount);

    for (int i = 0; i < repCount; i++) {
      ownedVecs.add(new IntList(s));
    }

    final CountDownLatch latch = new CountDownLatch(R.size());
    for (int i = 0; i < R.size(); i++) {
      final int Ri = R.get(i);
      final List<Integer> ROwned = ownedVecs.get(i);
      execServ.submit(new Runnable() {
        @Override
        public void run() {
          final BoundedSortedList<ProbailityMatch<VecPaired<V, Integer>>> nearest = new BoundedSortedList<ProbailityMatch<VecPaired<V, Integer>>>(
              s, s);
          for (final int v : allRemainingVecs) {
            nearest.add(new ProbailityMatch<VecPaired<V, Integer>>(dm.dist(v, Ri, allVecs, distCache),
                new VecPaired<V, Integer>(allVecs.get(v), v)));
          }

          for (final ProbailityMatch<VecPaired<V, Integer>> pmv : nearest) {
            ROwned.add(pmv.getMatch().getPair());
          }

          latch.countDown();
        }
      });
    }

    latch.await();

  }

  @Override
  public int size() {
    return R.size() * s;
  }
}
