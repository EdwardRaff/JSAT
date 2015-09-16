package jsat.classifiers.trees;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.ExecutorService;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.trees.ImpurityScore.ImpurityMeasure;
import jsat.math.OnLineStatistics;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.IntList;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;

/**
 * The ExtraTree is an Extremely Randomized Tree. Splits are chosen at random,
 * and the features that are selected are also chosen at random for each new
 * node in the tree. <br>
 * If set to randomly select one feature for each node, it becomes a <i>Totally
 * Randomized Tree</i><br>
 * See: <br>
 * Geurts, P., Ernst, D.,&amp;Wehenkel, L. (2006). <i>Extremely randomized
 * trees </i>. Machine learning, 63(1), 3–42. doi:10.1007/s10994-006-6226-1
 *
 * @author Edward Raff
 */
public class ExtraTree implements Classifier, Regressor, TreeLearner, Parameterized {
  // TODO in both of the train methods, 2 passes are done for numeric features.
  // This can be done in one pass by fiding the min/max when we split, and
  // passing that info in the argument parameters

  /**
   * Base node for regression and classification
   */
  private static abstract class NodeBase extends TreeNodeVisitor {

    /**
     *
     */
    private static final long serialVersionUID = 6783491817922690901L;
    protected TreeNodeVisitor[] children;

    public NodeBase() {
    }

    public NodeBase(final int children) {
      this.children = new TreeNodeVisitor[children];
    }

    public NodeBase(final NodeBase toClone) {
      if (toClone.children != null) {
        children = new TreeNodeVisitor[toClone.children.length];
        for (int i = 0; i < toClone.children.length; i++) {
          if (toClone.children[i] != null) {
            children[i] = toClone.children[i].clone();
          }
        }
      }
    }

    @Override
    public int childrenCount() {
      return children.length;
    }

    @Override
    public void disablePath(final int child) {
      if (!isLeaf()) {
        children[child] = null;
      }
    }

    @Override
    public TreeNodeVisitor getChild(final int child) {
      if (child < 0 || child > childrenCount()) {
        return null;
      }
      return children[child];
    }

    @Override
    public boolean isLeaf() {
      if (children == null) {
        return true;
      }
      for (final TreeNodeVisitor children1 : children) {
        if (children1 != null) {
          return false;
        }
      }
      return true;
    }

    @Override
    public boolean isPathDisabled(final int child) {
      if (isLeaf()) {
        return true;
      }
      return children[child] == null;
    }
  }

  /**
   * Base node for classification
   */
  private static class NodeC extends NodeBase {

    /**
     *
     */
    private static final long serialVersionUID = -3977497656918695759L;
    private final CategoricalResults crResult;

    /**
     * Creates a new leaf node
     *
     * @param crResult
     *          the results to return
     */
    public NodeC(final CategoricalResults crResult) {
      super();
      this.crResult = crResult;
      children = null;
    }

    /**
     * Creates a new node with children that start out with null (path disabled)
     *
     * @param crResult
     *          the results to return
     * @param children
     *          the number of children this node has
     */
    public NodeC(final CategoricalResults crResult, final int children) {
      super(children);
      this.crResult = crResult;
    }

    public NodeC(final NodeC toClone) {
      super(toClone);
      crResult = toClone.crResult.clone();
    }

    @Override
    public TreeNodeVisitor clone() {
      return new NodeC(this);
    }

    @Override
    public int getPath(final DataPoint dp) {
      return -1;
    }

    @Override
    public CategoricalResults localClassify(final DataPoint dp) {
      return crResult;
    }
  }

  /**
   * Node for classification that splits on a categorical feature
   */
  private static class NodeCCat extends NodeC {

    /**
     *
     */
    private static final long serialVersionUID = 7413428280703235600L;
    /**
     * Categorical attribute to split on
     */
    private final int catAtt;
    /**
     * The cat values that go to the left branch, or null if no binary cats are
     * being used
     */
    private int[] leftBranch;

    public NodeCCat(final int catAtt, final int children, final CategoricalResults crResult) {
      super(crResult, children);
      this.catAtt = catAtt;
      leftBranch = null;
    }

    public NodeCCat(final int catAtt, final Set<Integer> left, final CategoricalResults crResult) {
      super(crResult, 2);
      this.catAtt = catAtt;
      leftBranch = new int[left.size()];
      int pos = 0;
      for (final int i : left) {
        leftBranch[pos++] = i;
      }
      Arrays.sort(leftBranch);
    }

    public NodeCCat(final NodeCCat toClone) {
      super(toClone);
      catAtt = toClone.catAtt;
      if (toClone.leftBranch != null) {
        leftBranch = Arrays.copyOf(toClone.leftBranch, toClone.leftBranch.length);
      }
    }

    @Override
    public TreeNodeVisitor clone() {
      return new NodeCCat(this);
    }

    @Override
    public int getPath(final DataPoint dp) {
      final int[] catVals = dp.getCategoricalValues();
      if (leftBranch == null) {
        return catVals[catAtt];
      } else if (Arrays.binarySearch(leftBranch, catVals[catAtt]) < 0) {
        return 1;
      } else {
        return 0;
      }
    }

  }

  /**
   * Node for classification that splits on a numeric feature
   */
  private static class NodeCNum extends NodeC {

    /**
     *
     */
    private static final long serialVersionUID = 3967180517059509869L;
    private final int numerAtt;
    private final double threshold;

    public NodeCNum(final int numerAtt, final double threshold, final CategoricalResults crResult) {
      super(crResult, 2);
      this.numerAtt = numerAtt;
      this.threshold = threshold;
    }

    public NodeCNum(final NodeCNum toClone) {
      super(toClone);
      numerAtt = toClone.numerAtt;
      threshold = toClone.threshold;
    }

    @Override
    public TreeNodeVisitor clone() {
      return new NodeCNum(this);
    }

    @Override
    public int getPath(final DataPoint dp) {
      final double val = dp.getNumericalValues().get(numerAtt);
      if (val <= threshold) {
        return 0;
      } else {
        return 1;
      }
    }
  }

  /**
   * Base node for regression
   */
  private static class NodeR extends NodeBase {

    /**
     *
     */
    private static final long serialVersionUID = -2461046505444129890L;
    private final double result;

    /**
     * Creates a new leaf node
     *
     * @param result
     *          the result to return
     */
    public NodeR(final double result) {
      super();
      this.result = result;
    }

    /**
     * Creates a new node with children that start out with null (path disabled)
     *
     * @param crResult
     *          the results to return
     * @param children
     *          the number of children this node has
     */
    public NodeR(final double result, final int children) {
      super(children);
      this.result = result;
    }

    public NodeR(final NodeR toClone) {
      super(toClone);
      result = toClone.result;
    }

    @Override
    public TreeNodeVisitor clone() {
      return new NodeR(this);
    }

    @Override
    public int getPath(final DataPoint dp) {
      return -1;
    }

    @Override
    public double localRegress(final DataPoint dp) {
      return result;
    }
  }

  private static class NodeRCat extends NodeR {

    /**
     *
     */
    private static final long serialVersionUID = 5868393594474661054L;
    /**
     * Categorical attribute to split on
     */
    private final int catAtt;
    /**
     * The cat values that go to the left branch, or null if no binary cats are
     * being used
     */
    private int[] leftBranch;

    public NodeRCat(final int catAtt, final int children, final double result) {
      super(result, children);
      this.catAtt = catAtt;
      leftBranch = null;
    }

    public NodeRCat(final int catAtt, final Set<Integer> left, final double result) {
      super(result, 2);
      this.catAtt = catAtt;
      leftBranch = new int[left.size()];
      int pos = 0;
      for (final int i : left) {
        leftBranch[pos++] = i;
      }
      Arrays.sort(leftBranch);
    }

    public NodeRCat(final NodeRCat toClone) {
      super(toClone);
      catAtt = toClone.catAtt;
      if (toClone.leftBranch != null) {
        leftBranch = Arrays.copyOf(toClone.leftBranch, toClone.leftBranch.length);
      }
    }

    @Override
    public TreeNodeVisitor clone() {
      return new NodeRCat(this);
    }

    @Override
    public int getPath(final DataPoint dp) {
      final int[] catVals = dp.getCategoricalValues();
      if (leftBranch == null) {
        return catVals[catAtt];
      } else if (Arrays.binarySearch(leftBranch, catVals[catAtt]) < 0) {
        return 1;
      } else {
        return 0;
      }
    }
  }

  /**
   * Base node for regression that splits no a numeric feature
   */
  private static class NodeRNum extends NodeR {

    /**
     *
     */
    private static final long serialVersionUID = -6775472771777960211L;
    private final int numerAtt;
    private final double threshold;

    public NodeRNum(final int numerAtt, final double threshold, final double result) {
      super(result, 2);
      this.numerAtt = numerAtt;
      this.threshold = threshold;
    }

    public NodeRNum(final NodeRNum toClone) {
      super(toClone);
      numerAtt = toClone.numerAtt;
      threshold = toClone.threshold;
    }

    @Override
    public TreeNodeVisitor clone() {
      return new NodeRNum(this);
    }

    @Override
    public int getPath(final DataPoint dp) {
      final double val = dp.getNumericalValues().get(numerAtt);
      if (val <= threshold) {
        return 0;
      } else {
        return 1;
      }
    }
  }

  private static final long serialVersionUID = 7433728970041876327L;

  /**
   * Add lists to a list of lists
   *
   * @param <T>
   *          the content type of the list
   * @param listsToAdd
   *          the number of lists to add
   * @param reusableLists
   *          available pre allocated lists for reuse
   * @param aSplit
   *          the list of lists to add to
   */
  static private <T> void fillList(final int listsToAdd, final Stack<List<T>> reusableLists,
      final List<List<T>> aSplit) {
    for (int j = 0; j < listsToAdd; j++) {
      if (reusableLists.isEmpty()) {
        aSplit.add(new ArrayList<T>());
      } else {
        aSplit.add(reusableLists.pop());
      }
    }
  }

  /**
   * Adds the lists from the list of lists to the stack of reusable lists. The
   * lists will not be removed from <tt>aSplit</tt>, despite being added to
   * <tt>resuableLists</tt>
   *
   * @param <T>
   *          the content type of the list
   * @param reusableLists
   *          available pre allocated lists for reuse
   * @param aSplit
   *          the lists of lists to add all elements of to the reusable lists
   */
  static private <T> void fillStack(final Stack<List<T>> reusableLists, final List<List<T>> aSplit) {
    for (final List<T> list : aSplit) {
      list.clear();
      reusableLists.push(list);
    }
  }

  private int stopSize;

  private int selectionCount;

  private CategoricalData predicting;

  private boolean binaryCategoricalSplitting = true;

  private ImpurityScore.ImpurityMeasure impMeasure = ImpurityMeasure.NMI;

  private TreeNodeVisitor root;

  /**
   * Creates a new Extra Tree that will use all features in the training set
   */
  public ExtraTree() {
    this(Integer.MAX_VALUE, 5);
  }

  /**
   * Creates a new Extra Tree
   *
   * @param selectionCount
   *          the number of features to select
   * @param stopSize
   *          the stop size
   */
  public ExtraTree(final int selectionCount, final int stopSize) {
    this.stopSize = stopSize;
    this.selectionCount = selectionCount;
    impMeasure = ImpurityMeasure.NMI;
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    return root.classify(data);
  }

  @Override
  public ExtraTree clone() {
    final ExtraTree clone = new ExtraTree(selectionCount, stopSize);
    clone.impMeasure = impMeasure;
    clone.binaryCategoricalSplitting = binaryCategoricalSplitting;
    if (predicting != null) {
      clone.predicting = predicting.clone();
    }
    if (root != null) {
      clone.root = root.clone();
    }
    return clone;
  }

  private ImpurityScore[] createScores(final int count) {
    final ImpurityScore[] scores = new ImpurityScore[count];
    for (int j = 0; j < scores.length; j++) {
      scores[j] = new ImpurityScore(predicting.getNumOfCategories(), impMeasure);
    }

    return scores;
  }

  private OnLineStatistics[] createStats(final int count) {
    final OnLineStatistics[] stats = new OnLineStatistics[count];
    for (int i = 0; i < stats.length; i++) {
      stats[i] = new OnLineStatistics();
    }
    return stats;
  }

  /**
   * Returns the impurity measure in use
   *
   * @return the impurity measure in use
   */
  public ImpurityMeasure getImpurityMeasure() {
    return impMeasure;
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
   * Returns the number of random features chosen at each level in the tree
   *
   * @return the number of random features to chose
   */
  public int getSelectionCount() {
    return selectionCount;
  }

  /**
   * Returns the stopping size for tree growth
   *
   * @return the stopping size for tree growth
   */
  public int getStopSize() {
    return stopSize;
  }

  @Override
  public TreeNodeVisitor getTreeNodeVisitor() {
    return root;
  }

  /**
   * Returns whether or not binary splitting is used for nominal features
   *
   * @return whether or not binary splitting is used for nominal features
   */
  public boolean isBinaryCategoricalSplitting() {
    return binaryCategoricalSplitting;
  }

  @Override
  public double regress(final DataPoint data) {
    return root.regress(data);
  }

  /**
   * The normal implementation of ExtraTree always produces binary splits,
   * including for categorical features. If set to <tt>false</tt> categorical
   * features will expand out for each value in the category. This reduces the
   * randomness of the tree.
   *
   * @param binaryCategoricalSplitting
   *          whether or not to use the original splitting algorithm, or to
   *          fully expand nominal features
   */
  public void setBinaryCategoricalSplitting(final boolean binaryCategoricalSplitting) {
    this.binaryCategoricalSplitting = binaryCategoricalSplitting;
  }

  /**
   * Sets the impurity measure used during classification tree construction to
   * select the best of the features.
   *
   * @param impurityMeasure
   *          the impurity measure to use
   */
  public void setImpurityMeasure(final ImpurityMeasure impurityMeasure) {
    impMeasure = impurityMeasure;
  }

  /**
   * The ExtraTree will select the best of a random subset of features at each
   * level, this sets the number of random features to select. If set larger
   * than the number of features in the training set, all features will be
   * eligible for selection at every level.
   *
   * @param selectionCount
   *          the number of random features to select
   */
  public void setSelectionCount(final int selectionCount) {
    this.selectionCount = selectionCount;
  }

  /**
   * Sets the stopping size for tree growth. When a node has less than or equal
   * to <tt>stopSize</tt> data points to train from, it terminates and produces
   * a leaf node.
   *
   * @param stopSize
   *          the size of the testing set to refuse to split
   */
  public void setStopSize(final int stopSize) {
    if (stopSize <= 0) {
      throw new ArithmeticException("The stopping size must be a positive value");
    }
    this.stopSize = stopSize;
  }

  @Override
  public boolean supportsWeightedData() {
    return true;
  }

  /**
   * Creates a new tree top down
   *
   * @param setScore
   *          the impurity score for the set of data points being evaluated
   * @param subSet
   *          the set of data points to perform a split on
   * @param features
   *          the features available to split on
   * @param catInfo
   *          the categorical information
   * @param rand
   *          the source of randomness
   * @param reusableLists
   *          a stack of already allocated lists that can be added and removed
   *          from
   * @return the new top node created for the given data
   */
  private TreeNodeVisitor train(final OnLineStatistics setScore, final List<DataPointPair<Double>> subSet,
      final List<Integer> features, final CategoricalData[] catInfo, final Random rand,
      final Stack<List<DataPointPair<Double>>> reusableLists) {
    // Should we stop? Stop split(S)
    if (subSet.size() < stopSize || setScore.getVarance() <= 0.0 || Double.isNaN(setScore.getVarance())) {
      return new NodeR(setScore.getMean());
    }

    double bestGain = Double.NEGATIVE_INFINITY;
    double bestThreshold = Double.NaN;
    int bestAttribute = -1;
    OnLineStatistics[] bestScores = null;
    List<List<DataPointPair<Double>>> bestSplit = null;
    Set<Integer> bestLeftSide = null;

    /*
     * TODO use smarter random feature selection based on how many features we
     * need relative to how many are available
     */
    Collections.shuffle(features);

    // It is possible, if we test all attribute - that one was categorical and
    // no longer an option
    final int goTo = Math.min(selectionCount, features.size());
    for (int i = 0; i < goTo; i++) {
      double gain;
      double threshold = Double.NaN;
      Set<Integer> leftSide = null;
      OnLineStatistics[] stats;
      final int a = features.get(i);

      List<List<DataPointPair<Double>>> aSplit;

      if (a < catInfo.length) {
        final int vals = catInfo[a].getNumOfCategories();

        if (binaryCategoricalSplitting || vals == 2) {
          stats = createStats(2);
          final Set<Integer> catsValsInUse = new IntSet(vals * 2);
          for (final DataPointPair<Double> dpp : subSet) {
            catsValsInUse.add(dpp.getDataPoint().getCategoricalValue(a));
          }
          if (catsValsInUse.size() == 1) {
            return new NodeR(setScore.getMean());
          }
          leftSide = new IntSet(vals);
          final int toUse = rand.nextInt(catsValsInUse.size() - 1) + 1;
          ListUtils.randomSample(catsValsInUse, leftSide, toUse, rand);
          // Now we have anything in leftSide is path 0, we can do the bining

          aSplit = new ArrayList<List<DataPointPair<Double>>>(2);
          fillList(2, reusableLists, aSplit);
          for (final DataPointPair<Double> dpp : subSet) {
            final DataPoint dp = dpp.getDataPoint();
            final int dest = leftSide.contains(dpp.getDataPoint().getCategoricalValue(a)) ? 0 : 1;
            stats[dest].add(dpp.getPair(), dp.getWeight());
            aSplit.get(dest).add(dpp);
          }

        } else// split on each value
        {
          stats = createStats(vals);
          // Bin all the points to get their scores
          aSplit = new ArrayList<List<DataPointPair<Double>>>(vals);
          fillList(vals, reusableLists, aSplit);
          for (final DataPointPair<Double> dpp : subSet) {
            final DataPoint dp = dpp.getDataPoint();
            stats[dp.getCategoricalValue(a)].add(dpp.getPair(), dp.getWeight());
            aSplit.get(dp.getCategoricalValue(a)).add(dpp);
          }
        }
      } else {
        final int numerA = a - catInfo.length;
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (final DataPointPair<Double> dpp : subSet) {
          final double val = dpp.getVector().get(numerA);
          min = Math.min(min, val);
          max = Math.max(max, val);
        }

        // Uniform random threshold
        threshold = rand.nextDouble() * (max - min) + min;
        stats = createStats(2);

        aSplit = new ArrayList<List<DataPointPair<Double>>>(2);
        fillList(2, reusableLists, aSplit);
        for (final DataPointPair<Double> dpp : subSet) {
          final double val = dpp.getVector().get(numerA);

          final int toAddTo = val <= threshold ? 0 : 1;

          aSplit.get(toAddTo).add(dpp);
          stats[toAddTo].add(dpp.getPair(), dpp.getDataPoint().getWeight());
        }

      }

      gain = 1;
      final double varNorm = setScore.getVarance();
      final double varSum = setScore.getSumOfWeights();
      for (final OnLineStatistics stat : stats) {
        gain -= stat.getSumOfWeights() / varSum * (stat.getVarance() / varNorm);
      }
      if (gain > bestGain) {
        bestGain = gain;
        bestAttribute = a;
        bestThreshold = threshold;
        bestScores = stats;
        if (bestSplit != null) {
          fillStack(reusableLists, bestSplit);
        }
        bestSplit = aSplit;
        bestLeftSide = leftSide;
      } else {
        fillStack(reusableLists, aSplit);
      }

    }

    // Best attribute has been selected
    // We are no longer using the full array of all values
    fillStack(reusableLists, Arrays.asList(subSet));
    NodeBase toReturn;
    if (bestAttribute >= 0) {
      if (bestAttribute < catInfo.length) {
        if (bestSplit.size() == 2) {
          toReturn = new NodeRCat(bestAttribute, bestLeftSide, setScore.getMean());
        } else {
          toReturn = new NodeRCat(goTo, bestSplit.size(), setScore.getMean());
          features.remove(new Integer(bestAttribute));// Feature nolonger viable
                                                      // in this case
        }
      } else {
        toReturn = new NodeRNum(bestAttribute - catInfo.length, bestThreshold, setScore.getMean());
      }

      for (int i = 0; i < toReturn.children.length; i++) {
        toReturn.children[i] = train(bestScores[i], bestSplit.get(i), features, catInfo, rand, reusableLists);
      }
      return toReturn;
    }
    return new NodeR(setScore.getMean());
  }

  @Override
  public void train(final RegressionDataSet dataSet) {
    final Random rand = new Random();
    final Stack<List<DataPointPair<Double>>> reusableLists = new Stack<List<DataPointPair<Double>>>();
    final IntList features = new IntList(dataSet.getNumFeatures());
    ListUtils.addRange(features, 0, dataSet.getNumFeatures(), 1);

    final List<DataPointPair<Double>> data = dataSet.getAsDPPList();

    final OnLineStatistics score = new OnLineStatistics();
    for (final DataPointPair<Double> dpp : data) {
      score.add(dpp.getPair(), dpp.getDataPoint().getWeight());
    }

    root = train(score, data, features, dataSet.getCategories(), rand, reusableLists);
  }

  @Override
  public void train(final RegressionDataSet dataSet, final ExecutorService threadPool) {
    train(dataSet);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    final Random rand = new Random();
    final Stack<List<DataPointPair<Integer>>> reusableLists = new Stack<List<DataPointPair<Integer>>>();
    final IntList features = new IntList(dataSet.getNumFeatures());
    ListUtils.addRange(features, 0, dataSet.getNumFeatures(), 1);

    final List<DataPointPair<Integer>> data = dataSet.getAsDPPList();

    predicting = dataSet.getPredicting();
    final ImpurityScore score = new ImpurityScore(predicting.getNumOfCategories(), impMeasure);
    for (final DataPointPair<Integer> dpp : data) {
      score.addPoint(dpp.getDataPoint(), dpp.getPair());
    }

    root = trainC(score, data, features, dataSet.getCategories(), rand, reusableLists);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    trainC(dataSet);
  }

  /**
   * Creates a new tree top down
   *
   * @param setScore
   *          the impurity score for the set of data points being evaluated
   * @param subSet
   *          the set of data points to perform a split on
   * @param features
   *          the features available to split on
   * @param catInfo
   *          the categorical information
   * @param rand
   *          the source of randomness
   * @param reusableLists
   *          a stack of already allocated lists that can be added and removed
   *          from
   * @return the new top node created for the given data
   */
  private TreeNodeVisitor trainC(final ImpurityScore setScore, final List<DataPointPair<Integer>> subSet,
      final List<Integer> features, final CategoricalData[] catInfo, final Random rand,
      final Stack<List<DataPointPair<Integer>>> reusableLists) {
    // Should we stop? Stop split(S)
    if (subSet.size() < stopSize || setScore.getScore() == 0.0) {
      if (subSet.isEmpty()) {
        return null;
      }
      return new NodeC(setScore.getResults());
    }

    double bestGain = Double.NEGATIVE_INFINITY;
    double bestThreshold = Double.NaN;
    int bestAttribute = -1;
    ImpurityScore[] bestScores = null;
    List<List<DataPointPair<Integer>>> bestSplit = null;
    Set<Integer> bestLeftSide = null;

    /*
     * TODO use smarter random feature selection based on how many features we
     * need relative to how many are available
     */
    Collections.shuffle(features);

    // It is possible, if we test all attribute - that one was categorical and
    // no longer an option
    final int goTo = Math.min(selectionCount, features.size());
    for (int i = 0; i < goTo; i++) {
      double gain;
      double threshold = Double.NaN;
      Set<Integer> leftSide = null;
      ImpurityScore[] scores;
      final int a = features.get(i);

      List<List<DataPointPair<Integer>>> aSplit;

      if (a < catInfo.length) {
        final int vals = catInfo[a].getNumOfCategories();

        if (binaryCategoricalSplitting || vals == 2) {
          scores = createScores(2);
          final Set<Integer> catsValsInUse = new IntSet(vals * 2);
          for (final DataPointPair<Integer> dpp : subSet) {
            catsValsInUse.add(dpp.getDataPoint().getCategoricalValue(a));
          }
          if (catsValsInUse.size() == 1) {
            return new NodeC(setScore.getResults());
          }
          leftSide = new IntSet(vals);
          final int toUse = rand.nextInt(catsValsInUse.size() - 1) + 1;
          ListUtils.randomSample(catsValsInUse, leftSide, toUse, rand);
          // Now we have anything in leftSide is path 0, we can do the bining

          aSplit = new ArrayList<List<DataPointPair<Integer>>>(2);
          fillList(2, reusableLists, aSplit);
          for (final DataPointPair<Integer> dpp : subSet) {
            final DataPoint dp = dpp.getDataPoint();
            final int dest = leftSide.contains(dpp.getDataPoint().getCategoricalValue(a)) ? 0 : 1;
            scores[dest].addPoint(dp, dpp.getPair());
            aSplit.get(dest).add(dpp);
          }

        } else// split on each value
        {
          scores = createScores(vals);
          // Bin all the points to get their scores
          aSplit = new ArrayList<List<DataPointPair<Integer>>>(vals);
          fillList(vals, reusableLists, aSplit);
          for (final DataPointPair<Integer> dpp : subSet) {
            final DataPoint dp = dpp.getDataPoint();
            scores[dp.getCategoricalValue(a)].addPoint(dp, dpp.getPair());
            aSplit.get(dp.getCategoricalValue(a)).add(dpp);
          }
        }
      } else {
        final int numerA = a - catInfo.length;
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (final DataPointPair<Integer> dpp : subSet) {
          final double val = dpp.getVector().get(numerA);
          min = Math.min(min, val);
          max = Math.max(max, val);
        }

        // Uniform random threshold
        threshold = rand.nextDouble() * (max - min) + min;
        scores = createScores(2);

        aSplit = new ArrayList<List<DataPointPair<Integer>>>(2);
        fillList(2, reusableLists, aSplit);
        for (final DataPointPair<Integer> dpp : subSet) {
          final double val = dpp.getVector().get(numerA);

          final int toAddTo = val <= threshold ? 0 : 1;

          aSplit.get(toAddTo).add(dpp);
          scores[toAddTo].addPoint(dpp.getDataPoint(), dpp.getPair());
        }

      }

      gain = ImpurityScore.gain(setScore, scores);
      if (gain > bestGain) {
        bestGain = gain;
        bestAttribute = a;
        bestThreshold = threshold;
        bestScores = scores;
        if (bestSplit != null) {
          fillStack(reusableLists, bestSplit);
        }
        bestSplit = aSplit;
        bestLeftSide = leftSide;
      } else {
        fillStack(reusableLists, aSplit);
      }

    }

    // Best attribute has been selected
    // We are no longer using the full array of all values
    fillStack(reusableLists, Arrays.asList(subSet));
    NodeBase toReturn;
    if (bestAttribute < catInfo.length) {
      if (bestSplit.size() == 2) {
        toReturn = new NodeCCat(bestAttribute, bestLeftSide, setScore.getResults());
      } else {
        toReturn = new NodeCCat(goTo, bestSplit.size(), setScore.getResults());
        features.remove(new Integer(bestAttribute));// Feature nolonger viable
                                                    // in this case
      }
    } else {
      toReturn = new NodeCNum(bestAttribute - catInfo.length, bestThreshold, setScore.getResults());
    }
    for (int i = 0; i < toReturn.children.length; i++) {
      toReturn.children[i] = trainC(bestScores[i], bestSplit.get(i), features, catInfo, rand, reusableLists);
    }
    return toReturn;
  }
}
