package jsat.clustering.hierarchical;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;

import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.ClusterFailureException;
import jsat.clustering.KClusterer;
import jsat.clustering.KClustererBase;
import jsat.clustering.evaluation.ClusterEvaluation;

/**
 * DivisiveGlobalClusterer is a hierarchical clustering method that works by
 * splitting the data set into sub trees from the top down. Unlike many top-up
 * methods, such as {@link SimpleHAC}, top-down methods require another
 * clustering method to perform the splitting at each iteration. If the base
 * method is not deterministic, then the top-down method will not be
 * deterministic. <br>
 * Like many HAC methods, DivisiveGlobalClusterer will store the merge order of
 * the clusters so that the clustering results for many <i>k</i> can be
 * obtained. It is limited to the range of clusters successfully computed
 * before. <br>
 * <br>
 * Specifically, DivisiveGlobalClusterer greedily chooses the cluster to split
 * based on an evaluation of all resulting clusters after a split. Because of
 * this global search of the world, DivisiveLocalClusterer has can make a good
 * estimate of the number of clusters in the data set. The quality of this
 * result is dependent on the accuracy of the {@link ClusterEvaluation} used.
 * This quality comes at the cost of execution speed, as more and more large
 * evaluations of the whole dataset are needed at each iteration. If execution
 * speed is more important, {@link DivisiveLocalClusterer} should be used
 * instead, which requires only a fixed number of evaluations per iteration.
 *
 * @author Edward Raff
 */
public class DivisiveGlobalClusterer extends KClustererBase {

  private static final long serialVersionUID = -9117751530105155090L;
  private final KClusterer baseClusterer;
  private final ClusterEvaluation clusterEvaluation;

  private int[] splitList;
  private int[] fullDesignations;
  private DataSet originalDataSet;

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public DivisiveGlobalClusterer(final DivisiveGlobalClusterer toCopy) {
    baseClusterer = toCopy.baseClusterer.clone();
    clusterEvaluation = toCopy.clusterEvaluation.clone();
    if (toCopy.splitList != null) {
      splitList = Arrays.copyOf(toCopy.splitList, toCopy.splitList.length);
    }
    if (toCopy.fullDesignations != null) {
      fullDesignations = Arrays.copyOf(toCopy.fullDesignations, toCopy.fullDesignations.length);
    }
    originalDataSet = toCopy.originalDataSet.shallowClone();
  }

  public DivisiveGlobalClusterer(final KClusterer baseClusterer, final ClusterEvaluation clusterEvaluation) {
    this.baseClusterer = baseClusterer;
    this.clusterEvaluation = clusterEvaluation;
  }

  @Override
  public DivisiveGlobalClusterer clone() {
    return new DivisiveGlobalClusterer(this);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final ExecutorService threadpool, final int[] designations) {
    return cluster(dataSet, 2, (int) Math.sqrt(dataSet.getSampleSize()), threadpool, designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int clusters, final ExecutorService threadpool,
      final int[] designations) {
    return cluster(dataSet, clusters, clusters, threadpool, designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int lowK, final int highK, final ExecutorService threadpool,
      int[] designations) {
    if (designations == null) {
      designations = new int[dataSet.getSampleSize()];
    }
    /**
     * Is used to copy the value of designations and then alter to test the
     * quality of a potential new clustering
     */
    final int[] fakeWorld = new int[dataSet.getSampleSize()];

    /**
     * For each current cluster, we store the clustering results if we attempt
     * to split it into two. <br>
     * Each row needs to be re-set since the clustering methods will use the
     * length of the cluster size
     */
    final int[][] subDesignation = new int[highK][];
    /**
     * Stores the index from the sub data set into the full data set
     */
    final int[][] originalPositions = new int[highK][dataSet.getSampleSize()];
    /**
     * List of Lists for holding the data points of each cluster in
     */
    final List<List<DataPoint>> pointsInCluster = new ArrayList<List<DataPoint>>(highK);
    for (int i = 0; i < highK; i++) {
      pointsInCluster.add(new ArrayList<DataPoint>(dataSet.getSampleSize()));
    }

    /**
     * Stores the dissimilarity of the splitting of the cluster with the same
     * index value. Negative value indicates not set. Special values:<br>
     * <ul>
     * <li>NEGATIVE_INFINITY : value never used</li>
     * <li>-1 : clustering computed, but no current global evaluation</li>
     * <li>>=0 : cluster computed, current value is the evaluation for using
     * this split</li>
     * </ul>
     */
    final double[] splitEvaluation = new double[highK];
    Arrays.fill(splitEvaluation, Double.NEGATIVE_INFINITY);

    /**
     * Records the order in which items were split
     */
    splitList = new int[highK * 2 - 2];
    int bestK = -1;
    double bestKEval = Double.POSITIVE_INFINITY;

    // k is the current number of clusters, & the ID of the next cluster
    for (int k = 1; k < highK; k++) {
      double bestSplitVal = Double.POSITIVE_INFINITY;
      int bestID = -1;

      for (int z = 0; z < k; z++)// TODO it might be better to do this loop in
                                 // parallel
      {
        if (Double.isNaN(splitEvaluation[z])) {
          continue;
        } else if (splitEvaluation[z] == Double.NEGATIVE_INFINITY) // at most 2
                                                                   // will hit
                                                                   // this per
                                                                   // loop
        {// Need to compute a split for that cluster & set up helper structures
          final List<DataPoint> clusterPointsZ = pointsInCluster.get(z);
          clusterPointsZ.clear();
          for (int i = 0; i < dataSet.getSampleSize(); i++) {
            if (designations[i] != z) {
              continue;
            }
            originalPositions[z][clusterPointsZ.size()] = i;
            clusterPointsZ.add(dataSet.getDataPoint(i));
          }
          subDesignation[z] = new int[clusterPointsZ.size()];
          if (clusterPointsZ.isEmpty()) // Empty cluster? How did that happen...
          {
            splitEvaluation[z] = Double.NaN;
            continue;
          }
          final SimpleDataSet subDataSet = new SimpleDataSet(clusterPointsZ);

          try {
            // if (threadpool == null)
            baseClusterer.cluster(subDataSet, 2, subDesignation[z]);
            // else
            // baseClusterer.cluster(subDataSet, 2, threadpool,
            // subDesignation[z]);
          } catch (final ClusterFailureException ex) {
            splitEvaluation[z] = Double.NaN;
            continue;
          }
        }

        System.arraycopy(designations, 0, fakeWorld, 0, fakeWorld.length);
        for (int i = 0; i < subDesignation[z].length; i++) {
          if (subDesignation[z][i] == 1) {
            fakeWorld[originalPositions[z][i]] = k;
          }
        }
        try {
          splitEvaluation[z] = clusterEvaluation.evaluate(fakeWorld, dataSet);
        } catch (final Exception ex)// Can occur if one of the clusters has size
                                    // zeros
        {
          splitEvaluation[z] = Double.NaN;
          continue;
        }

        if (splitEvaluation[z] < bestSplitVal) {
          bestSplitVal = splitEvaluation[z];
          bestID = z;
        }
      }

      // We now know which cluster we should use the split of
      for (int i = 0; i < subDesignation[bestID].length; i++) {
        if (subDesignation[bestID][i] == 1) {
          designations[originalPositions[bestID][i]] = k;
        }
      }

      // The original clsuter id, and the new one should be set to -Inf
      splitEvaluation[bestID] = splitEvaluation[k] = Double.NEGATIVE_INFINITY;

      // Store a split list
      splitList[(k - 1) * 2] = bestID;
      splitList[(k - 1) * 2 + 1] = k;
      if (lowK - 1 <= k && k <= highK - 1) // Should we stop?
      {
        if (bestSplitVal < bestKEval) {
          bestKEval = bestSplitVal;
          bestK = k;
          System.out.println("Best k is now " + k + " at " + bestKEval);
        }
      }
    }

    fullDesignations = Arrays.copyOf(designations, designations.length);

    // Merge the split clusters back to the one that had the best score
    for (int k = splitList.length / 2 - 1; k >= bestK; k--) {
      if (splitList[k * 2] == splitList[k * 2 + 1]) {
        continue;// Happens when we bail out early
      }
      for (int j = 0; j < designations.length; j++) {
        if (designations[j] == splitList[k * 2 + 1]) {
          designations[j] = splitList[k * 2];
        }
      }
    }

    originalDataSet = dataSet;
    return designations;
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int lowK, final int highK, final int[] designations) {
    return cluster(dataSet, lowK, highK, null, designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int clusters, final int[] designations) {
    return cluster(dataSet, clusters, clusters, designations);
  }

  @Override
  public int[] cluster(final DataSet dataSet, final int[] designations) {
    return cluster(dataSet, 2, (int) Math.sqrt(dataSet.getSampleSize()), designations);
  }

  /**
   * Returns the clustering results for a specific <i>k</i> number of clusters
   * for a previously computed data set. If the data set did not compute up to
   * the value <i>k</i> <tt>null</tt> will be returned.
   *
   * @param targetK
   *          the number of clusters to get the result for.
   * @return an array containing the assignments for each cluster in the
   *         original data set.
   * @throws ClusterFailureException
   *           if no prior data set had been clustered
   */
  public int[] clusterSplit(final int targetK) {
    if (originalDataSet == null) {
      throw new ClusterFailureException("No prior cluster stored");
    }
    final int[] newDesignations = Arrays.copyOf(fullDesignations, fullDesignations.length);
    // Merge the split clusters back to the one that had the best score
    for (int k = splitList.length / 2 - 1; k >= targetK; k--) {
      if (splitList[k * 2] == splitList[k * 2 + 1]) {
        continue;// Happens when we bail out early
      }
      for (int j = 0; j < newDesignations.length; j++) {
        if (newDesignations[j] == splitList[k * 2 + 1]) {
          newDesignations[j] = splitList[k * 2];
        }
      }
    }
    return newDesignations;
  }

}
