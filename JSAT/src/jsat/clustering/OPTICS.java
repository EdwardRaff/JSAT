package jsat.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.DefaultVectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollectionUtils;
import jsat.math.OnLineStatistics;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.IntList;
import jsat.utils.IntSet;

/**
 * An Implementation of the OPTICS algorithm, which is a generalization of
 * {@link DBSCAN}. OPTICS creates an ordering of the ports, and then clusters
 * can be extracted from this ordering in numerous different ways. <br>
 * NOTE: The original clustering method proposed in the paper is fairly
 * complicated, and its implementation is not yet complete. Though it does
 * perform some amount of clustering, it may not return the expected results.
 * <br>
 * <br>
 * See original paper<br>
 * Ankerst, M., Breunig, M., Kriegel, H.-P.,&amp;Sander, J. (1999).
 * <a href="http://dl.acm.org/citation.cfm?id=304187"><i>OPTICS: ordering points
 * to identify the clustering structure</i></a>. Proceedings of the 1999 ACM
 * SIGMOD international conference on Management of data (Vol. 28, pp. 49–60).
 * Philadelphia, Pennsylvania: ACM.
 *
 * @author Edward Raff
 */
public class OPTICS extends ClustererBase implements Parameterized {

  /**
   * Enum to indicate which method of extracting clusters should be used on the
   * reachability plot.
   *
   */
  public enum ExtractionMethod {
    /**
     * Uses the original clustering method proposed in the OPTICS paper.<br>
     * NOTE: Implementation not yet complete
     */
    XI_STEEP_ORIGINAL, /**
                        * Forms clusters in the reachability plot by drawing a
                        * line across it, and using the separations to mark
                        * clusters
                        */
    THRESHHOLD, /**
                 * Forms clusters in the reachability plot by drawing a line
                 * across it, and using the separations to mark clusters. It
                 * then de-noises points by checking their nearest neighbors for
                 * consensus
                 */
    THRESHHOLD_FIXUP
  }

  /**
   * Private class for keeping track of heiarchies of clusters
   */
  private class OPTICSCluster {

    int start, end;
    List<OPTICSCluster> subClusters;

    public OPTICSCluster(final int start, final int end) {
      this.start = start;
      this.end = end;
      subClusters = new ArrayList<OPTICSCluster>(5);
    }

    /**
     *
     * @param other
     * @return
     */
    public boolean contains(final OPTICSCluster other) {
      return start <= other.start && other.end <= end;
    }

    @Override
    public String toString() {
      return "{" + start + "," + end + "}";
    }

  }

  private static final long serialVersionUID = -1093772096278544211L;

  private static final int NOISE = -1;
  private static double UNDEFINED = Double.POSITIVE_INFINITY;

  /**
   * The default value for xi is {@value #DEFAULT_XI}
   */
  public static final double DEFAULT_XI = 0.005;

  /**
   * The default number of points to consider is {@value #DEFAULT_MIN_POINTS}.
   */
  public static final int DEFAULT_MIN_POINTS = 10;

  /**
   * The default method used to extract clusters in OPTICS
   */
  public static final ExtractionMethod DEFAULT_EXTRACTION_METHOD = ExtractionMethod.THRESHHOLD_FIXUP;
  private DistanceMetric dm;
  private VectorCollectionFactory<VecPaired<Vec, Integer>> vcf = new DefaultVectorCollectionFactory<VecPaired<Vec, Integer>>();
  private VectorCollection<VecPaired<Vec, Integer>> vc;
  private double radius = 1;
  private int minPts;
  private double[] core_distance;
  /**
   * Stores the reachability distance of each point in the order they were first
   * observed in the data set. After clustering is finished, it is altered to be
   * in the reachability order used in clustering
   */
  private double[] reach_d;

  /**
   * Whether or not the given data point has been processed
   */
  private boolean[] processed;
  private Vec[] allVecs;
  private double xi;

  // XXX useless assignment
  private double one_min_xi;// = 1.0-xi;

  private ExtractionMethod extractionMethod = DEFAULT_EXTRACTION_METHOD;

  /**
   * The objects contained in OrderSeeds are sorted by their
   * reachability-distance to the closest core object from which they have been
   * directly density reachable.
   *
   * The paired double is their distance, the paired integer the the vector's
   * index in the data set
   *
   * This is only used during building. We should probably refactor this out
   */
  private PriorityQueue<Integer> orderdSeeds;

  /**
   * Creates a new OPTICS cluster object. Because the radius of OPTICS is not
   * sensitive, it is estimated from the data and set to a sufficiently large
   * value. The {@link EuclideanDistance} will be used as the metric.
   */
  public OPTICS() {
    this(DEFAULT_MIN_POINTS);
  }

  /**
   * Creates a new OPTICS cluster object. Because the radius of OPTICS is not
   * sensitive, it is estimated from the data and set to a sufficiently large
   * value.
   *
   * @param dm
   *          the distance metric to use
   * @param minPts
   *          the minimum number of points for reachability
   */
  public OPTICS(final DistanceMetric dm, final int minPts) {
    this(dm, minPts, DEFAULT_XI);
  }

  /**
   * Creates a new OPTICS cluster object. Because the radius of OPTICS is not
   * sensitive, it is estimated from the data and set to a sufficiently large
   * value.
   *
   * @param dm
   *          the distance metric to use
   * @param minPts
   *          the minimum number of points for reachability
   * @param xi
   *          the xi value
   */
  public OPTICS(final DistanceMetric dm, final int minPts, final double xi) {
    setDistanceMetric(dm);
    setMinPts(minPts);
    setXi(xi);
  }

  /**
   * Creates a new OPTICS cluster object. Because the radius of OPTICS is not
   * sensitive, it is estimated from the data and set to a sufficiently large
   * value. The {@link EuclideanDistance} will be used as the metric.
   *
   * @param minPts
   *          the minimum number of points for reachability
   */
  public OPTICS(final int minPts) {
    this(new EuclideanDistance(), minPts);
  }

  public OPTICS(final OPTICS toCopy) {
    dm = toCopy.dm.clone();
    vc = toCopy.vc.clone();
    minPts = toCopy.minPts;
    if (toCopy.core_distance != null) {
      core_distance = Arrays.copyOf(toCopy.core_distance, toCopy.core_distance.length);
    }

    if (toCopy.reach_d != null) {
      reach_d = Arrays.copyOf(toCopy.reach_d, toCopy.reach_d.length);
    }

    if (toCopy.processed != null) {
      processed = Arrays.copyOf(toCopy.processed, toCopy.processed.length);
    }

    if (toCopy.allVecs != null) {
      allVecs = new Vec[toCopy.allVecs.length];
      for (int i = 0; i < toCopy.allVecs.length; i++) {
        allVecs[i] = toCopy.allVecs[i].clone();
      }
    }
    xi = toCopy.xi;
    orderdSeeds = toCopy.orderdSeeds;
    radius = toCopy.radius;
  }

  @Override
  public OPTICS clone() {
    return new OPTICS(this);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final ExecutorService threadpool, final int[] designations) {
    return cluster(dataSet, designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, int[] designations) {
    if (dataSet.getNumNumericalVars() < 1) {
      throw new ClusterFailureException("OPTICS requires numeric features, and non are present.");
    }

    final int n = dataSet.getSampleSize();
    if (designations == null) {
      designations = new int[n];
    }

    Arrays.fill(designations, NOISE);
    orderdSeeds = new PriorityQueue<Integer>(n, new Comparator<Integer>() {

      @Override
      public int compare(final Integer o1, final Integer o2) {
        return Double.compare(reach_d[o1], reach_d[o2]);
      }
    });
    core_distance = new double[n];
    reach_d = new double[n];
    Arrays.fill(reach_d, UNDEFINED);
    processed = new boolean[n];
    allVecs = new Vec[n];
    final List<VecPaired<Vec, Integer>> pairedVecs = new ArrayList<VecPaired<Vec, Integer>>(n);
    for (int i = 0; i < allVecs.length; i++) {
      allVecs[i] = dataSet.getDataPoint(i).getNumericalValues();
      pairedVecs.add(new VecPaired<Vec, Integer>(allVecs[i], i));
    }
    vc = vcf.getVectorCollection(pairedVecs, dm);

    // Estimate radius value
    final OnLineStatistics stats = VectorCollectionUtils.getKthNeighborStats(vc, allVecs, minPts + 1);

    radius = stats.getMean() + stats.getStandardDeviation() * 3;

    final List<Integer> orderedFile = new IntList(n);

    // Main clustering loop
    for (int i = 0; i < dataSet.getSampleSize(); i++) {
      if (processed[i]) {
        continue;
      }
      final Vec vec = dataSet.getDataPoint(i).getNumericalValues();
      expandClusterOrder(i, vec, orderedFile);
    }

    int clustersFound;
    if (extractionMethod == ExtractionMethod.THRESHHOLD) {
      clustersFound = threshHoldExtractCluster(orderedFile, designations);
    } else if (extractionMethod == ExtractionMethod.THRESHHOLD_FIXUP) {
      clustersFound = threshHoldFixExtractCluster(orderedFile, designations);
    } else if (extractionMethod == ExtractionMethod.XI_STEEP_ORIGINAL) {
      clustersFound = xiSteepClusterExtract(n, orderedFile, designations);
    }

    // Sort reachability values
    final double[] newReach = new double[reach_d.length];
    Arrays.fill(newReach, Double.POSITIVE_INFINITY);
    for (int i = 0; i < orderedFile.size(); i++) {
      newReach[i] = reach_d[orderedFile.get(i)];
    }
    reach_d = newReach;

    return designations;
  }

  private boolean downPoint(final int index1, final int index2) {
    return reach_d[index1] * one_min_xi <= reach_d[index2];
  }

  private void expandClusterOrder(final int curIndex, final Vec vec, final List<Integer> orderedFile) {
    List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors = vc.search(vec, radius);
    final VecPaired<Vec, Integer> object = new VecPaired<Vec, Integer>(vec, curIndex);

    reach_d[curIndex] = UNDEFINED;// NaN used for undefined
    processed[curIndex] = true;
    setCoreDistance(neighbors, curIndex);
    orderedFile.add(curIndex);

    if (!Double.isInfinite(core_distance[curIndex])) {
      orderedSeedsUpdate(neighbors, curIndex);
      while (!orderdSeeds.isEmpty()) {
        final int curObjectIndex = orderdSeeds.poll();
        neighbors = vc.search(allVecs[curObjectIndex], radius);
        processed[curObjectIndex] = true;
        setCoreDistance(neighbors, curObjectIndex);
        orderedFile.add(curObjectIndex);
        if (!Double.isInfinite(core_distance[curObjectIndex])) {
          orderedSeedsUpdate(neighbors, curObjectIndex);
        }
      }
    }

  }

  @SuppressWarnings("unused")
  private void extractClusteringDBSCAN(final List<Integer> orderedFile, final double e, final int[] designations) {
    int clusterID = NOISE;
    for (int i = 0; i < orderedFile.size(); i++) {
      final int trueObjIndex = orderedFile.get(i);
      if (Double.isInfinite(reach_d[trueObjIndex]) || reach_d[trueObjIndex] > e) {
        if (core_distance[trueObjIndex] <= e) {
          clusterID++;
          designations[trueObjIndex] = clusterID;
        } else {
          designations[trueObjIndex] = NOISE;
        }
      } else {
        designations[trueObjIndex] = clusterID;
      }
    }
    throw new UnsupportedOperationException("Not yet implemented");
  }

  private void filterSDASet(final Set<Integer> sdaSet, final double mib, final double[] mibVals,
      final List<Integer> orderedFile) {
    for (final Iterator<Integer> iter = sdaSet.iterator(); iter.hasNext();) {
      final int sdaIndx = orderedFile.get(iter.next());
      if (reach_d[sdaIndx] * one_min_xi <= mib) {
        iter.remove();
      } else {
        mibVals[sdaIndx] = Math.max(mib, mibVals[sdaIndx]);// TODO mibFill?
      }
    }
  }

  /**
   * Returns the distance metric used to compute distances in the algorithm.
   *
   * @return the distance metric used
   */
  public DistanceMetric getDistanceMetric() {
    return dm;
  }

  /**
   * Returns the method used to extract clusters from the reachability plot.
   *
   * @return the clustering method used
   */
  public ExtractionMethod getExtractionMethod() {
    return extractionMethod;
  }

  /**
   * Sets the minimum number of points needed to compute the core distance.
   *
   * @return the number of points to compute reachability and core distance
   */
  public int getMinPts() {
    return minPts;
  }

  @Override
  public Parameter getParameter(final String paramName) {
    return Parameter.toParameterMap(getParameters()).get(paramName);
  }

  @Override
  public List<Parameter> getParameters() {
    return Parameter.getParamsFromMethods(this);
  }

  /**
   * Returns a copy of the reachability array in correct reachability order.
   * Some values that are not density reachability could have a value of
   * {@link Double#POSITIVE_INFINITY}.
   *
   * @return an array of the reachability values
   */
  public double[] getReachabilityArray() {
    return Arrays.copyOf(reach_d, reach_d.length);
  }

  /**
   * Returns the xi value used in {@link ExtractionMethod#XI_STEEP_ORIGINAL} to
   * produce cluster results.
   *
   * @return the xi value used
   */
  public double getXi() {
    return xi;
  }

  private void orderedSeedsUpdate(final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors,
      final int centerObjectIndex) {
    final double c_dist = core_distance[centerObjectIndex];
    for (int i = 1; i < neighbors.size(); i++)// '0' index is a self reference,
                                              // skip it
    {
      final VecPaired<VecPaired<Vec, Integer>, Double> neighbor = neighbors.get(i);
      final int objIndex = neighbor.getVector().getPair();
      if (processed[objIndex]) {
        continue;
      }
      final double new_r_dist = Math.max(c_dist, neighbor.getPair());
      if (Double.isInfinite(reach_d[objIndex])) {
        reach_d[objIndex] = new_r_dist;
        // r_dists[objIndex] = new_r_dist;
        orderdSeeds.add(objIndex);
      } else if (new_r_dist < reach_d[objIndex]) // Object already in
                                                 // OrderSeeds, but we can do
                                                 // better
      {
        reach_d[objIndex] = new_r_dist;
        // r_dists[objIndex] = new_r_dist;
        orderdSeeds.remove(objIndex);
        orderdSeeds.add(objIndex);
      }

    }

  }

  private void setCoreDistance(final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors,
      final int curIndex) {
    if (neighbors.size() < minPts + 1) {// +1 b/c we dont count oursleves, which
                                        // will get returned
      core_distance[curIndex] = UNDEFINED;
    } else {
      // 0 is us, 1 is the nearest neighbor
      core_distance[curIndex] = neighbors.get(minPts).getPair();
    }
  }

  /**
   * Sets the distance metric used to compute distances in the algorithm.
   *
   * @param dm
   *          the distance metric to use
   */
  public void setDistanceMetric(final DistanceMetric dm) {
    this.dm = dm;
  }

  /**
   * Sets the method used to extract clusters from the reachability plot.
   *
   * @param extractionMethod
   *          the clustering method
   */
  public void setExtractionMethod(final ExtractionMethod extractionMethod) {
    this.extractionMethod = extractionMethod;
  }

  /**
   * Sets the minimum number of points needed to compute the core distance.
   * Higher values tend to smooth out the reachability plot.
   *
   * @param minPts
   *          the number of points to compute reachability and core distance
   */
  public void setMinPts(final int minPts) {
    this.minPts = minPts;
  }

  /**
   * Sets the {@link VectorCollectionFactory} used to produce acceleration
   * structures for the OPTICS computation.
   *
   * @param vcf
   *          the vector collection factory to use
   */
  public void setVCF(final VectorCollectionFactory<VecPaired<Vec, Integer>> vcf) {
    this.vcf = vcf;
  }

  /**
   * Sets the xi value used in {@link ExtractionMethod#XI_STEEP_ORIGINAL} to
   * produce cluster results.
   *
   * @param xi
   *          the value in the range (0, 1)
   * @throws ArithmeticException
   *           if the value is not in the appropriate range
   */
  public void setXi(final double xi) {
    if (xi <= 0 || xi >= 1 || Double.isNaN(xi)) {
      throw new ArithmeticException("xi must be in the range (0, 1) not " + xi);
    }
    this.xi = xi;
    one_min_xi = 1.0 - xi;
  }

  /**
   * Finds clusters by segmenting the reachability plot witha line that is the
   * mean reachability distance times
   *
   * @param orderedFile
   *          the ordering of the data points
   * @param designations
   *          the storage array for their cluster assignment
   * @return the number of clusters found
   */
  private int threshHoldExtractCluster(final List<Integer> orderedFile, final int[] designations) {
    int clustersFound = 0;
    final OnLineStatistics stats = new OnLineStatistics();
    for (final double r : reach_d) {
      if (!Double.isInfinite(r)) {
        stats.add(r);
      }
    }

    final double thresh = stats.getMean() + stats.getStandardDeviation();
    for (int i = 0; i < orderedFile.size(); i++) {
      if (reach_d[orderedFile.get(i)] >= thresh) {
        continue;
      }
      // Everything in between is part of the cluster
      while (i < orderedFile.size() && reach_d[orderedFile.get(i)] < thresh) {
        designations[i++] = clustersFound;
      }
      // Climb up to the top of the hill, everything we climbed over is part of
      // the cluster
      while (i + 1 < orderedFile.size() && reach_d[orderedFile.get(i)] < reach_d[orderedFile.get(i + 1)]) {
        designations[i++] = clustersFound;
      }
      clustersFound++;
    }
    return clustersFound;
  }

  private int threshHoldFixExtractCluster(final List<Integer> orderedFile, final int[] designations) {
    final int clustersFound = threshHoldExtractCluster(orderedFile, designations);

    for (int i = 0; i < orderedFile.size(); i++) {
      if (designations[i] != NOISE) {
        continue;
      }
      // Check if all the neighbors have a consensus on the cluster class
      // (ignoring noise)
      final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors = vc.search(allVecs[i],
          minPts / 2 + 1);
      int CLASS = -1;// -1 for not set, -2 for conflic

      for (final VecPaired<VecPaired<Vec, Integer>, Double> v : neighbors) {
        final int subC = designations[v.getVector().getPair()];
        if (subC == NOISE) {
          // ignore
          // ignore
        } else if (CLASS == -1) {// First class set
          CLASS = subC;
        } else if (CLASS != subC) {// Conflict
          CLASS = -2;// No consensus, we wont change the noise label
        }
      }

      if (CLASS != -2) {
        designations[i] = CLASS;
      }
    }

    return clustersFound;
  }

  private boolean upPoint(final int index1, final int index2) {
    return reach_d[index1] <= reach_d[index2] * one_min_xi;
  }

  /**
   * Extracts clusters using the original xi steep algorithm from the OPTICS
   * paper
   *
   * @param n
   *          original number of data points
   * @param orderedFile
   *          the correct order of the points in the reachability plot
   * @param designations
   *          the array to store the final class designations
   * @return the number of clusters found
   */
  private int xiSteepClusterExtract(final int n, final List<Integer> orderedFile, final int[] designations) {
    /// Now obtain clustering
    /// Extract CLustering
    int clustersFound = 0;
    final Set<Integer> sdaSet = new IntSet();
    int orderIndex = 0;
    double mib = 0;
    final double[] mibVals = new double[n];

    final List<OPTICSCluster> clusters = new ArrayList<OPTICSCluster>();
    final List<Integer> allSteepUp = new IntList();
    final List<Integer> allSDA = new IntList();
    /*
     * Ugly else if to increment orderIndex counter and avoid geting stuck in
     * infinite loops. Can I write that a better way?
     */
    while (orderIndex < orderedFile.size() - 1) {

      int curIndex = orderedFile.get(orderIndex);
      mib = Math.max(mib, reach_d[curIndex]);

      if (orderIndex + 1 < orderedFile.size()) {
        int nextIndex = orderedFile.get(orderIndex + 1);
        if (!downPoint(curIndex, nextIndex)) // IF(start of a steep down area D
                                             // at index)
        {
          filterSDASet(sdaSet, mib, mibVals, orderedFile);

          sdaSet.add(orderIndex);
          allSDA.add(orderIndex);

          while (orderIndex + 1 < orderedFile.size()) {
            orderIndex++;
            curIndex = nextIndex;
            if (orderIndex + 1 >= orderedFile.size()) {
              break;
            }
            nextIndex = orderedFile.get(orderIndex + 1);
            if (downPoint(curIndex, nextIndex)) {
              break;
            }
          }

          mib = reach_d[curIndex];
        } else if (!upPoint(curIndex, nextIndex)) // ELSE IF(start of steep up
                                                  // area U at index)
        {

          filterSDASet(sdaSet, mib, mibVals, orderedFile);
          if (!sdaSet.isEmpty()) {
            allSteepUp.add(orderIndex);
          }

          while (orderIndex + 1 < orderedFile.size()) {
            orderIndex++;
            curIndex = nextIndex;
            if (orderIndex + 1 >= orderedFile.size()) {
              break;
            }
            nextIndex = orderedFile.get(orderIndex + 1);
            if (upPoint(curIndex, nextIndex)) {
              break;
            }
          }

          mib = reach_d[curIndex];

          for (final int sdaOrdered : sdaSet) {
            final int sdaIndx = orderedFile.get(sdaOrdered);
            if (!(orderIndex - sdaOrdered >= minPts)) {// Fail 3a
              continue;
            } else if (mib * one_min_xi < mibVals[sdaIndx]) {
              continue;
            }
            if (sdaOrdered > orderIndex) {
              continue;
            }
            final OPTICSCluster newClust = new OPTICSCluster(sdaOrdered, orderIndex + 1);
            OPTICSCluster tmp;
            for (final Iterator<OPTICSCluster> clustIter = clusters.iterator(); clustIter.hasNext();) {
              if (newClust.contains(tmp = clustIter.next())) {
                clustIter.remove();
                newClust.subClusters.add(tmp);
              }
            }
            clusters.add(newClust);
          }

        } else {
          orderIndex++;
        }

      } else {
        orderIndex++;
      }

    }
    for (final OPTICSCluster oc : clusters) {
      for (final int i : orderedFile.subList(oc.start, oc.end)) {
        if (designations[i] < 0) {
          designations[i] = clustersFound;
        }
      }
      clustersFound++;
    }
    return clustersFound;
  }
}
