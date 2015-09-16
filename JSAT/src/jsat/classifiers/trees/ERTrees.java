package jsat.classifiers.trees;

import static java.lang.Math.max;
import static java.lang.Math.round;
import static java.lang.Math.sqrt;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;

import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.exceptions.FailedToFitException;
import jsat.regression.RegressionDataSet;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;

/**
 * Extra Randomized Trees (ERTrees) is an ensemble method built on top of
 * {@link ExtraTree}. The randomness of the trees provides incredibly high
 * variance, yet a low bias. The sum of many randomized trees proves to be a
 * powerful and fast learner. <br>
 * The default settings are those suggested in the paper. However, the default
 * stop size suggested (especially for classification) is often too small. You
 * may want to consider increasing it if the accuracy is too low. <br>
 * See: <br>
 * Geurts, P., Ernst, D.,&amp;Wehenkel, L. (2006). <i>Extremely randomized
 * trees </i>. Machine learning, 63(1), 3–42. doi:10.1007/s10994-006-6226-1
 *
 * @author Edward Raff
 */
public class ERTrees extends ExtraTree {

  private class ForrestPlanter implements Runnable {

    int start;
    int end;

    DataSet dataSet;
    CountDownLatch latch;

    public ForrestPlanter(final int start, final int end, final DataSet dataSet, final CountDownLatch latch) {
      this.start = start;
      this.end = end;
      this.dataSet = dataSet;
      this.latch = latch;
    }

    @Override
    public void run() {
      if (dataSet instanceof ClassificationDataSet) {
        final ClassificationDataSet cds = (ClassificationDataSet) dataSet;
        for (int i = start; i < end; i++) {
          forrest[i] = baseTree.clone();
          forrest[i].trainC(cds);
        }
      } else if (dataSet instanceof RegressionDataSet) {
        final RegressionDataSet rds = (RegressionDataSet) dataSet;
        for (int i = start; i < end; i++) {
          forrest[i] = baseTree.clone();
          forrest[i].train(rds);
        }
      } else {
        throw new RuntimeException("BUG: Please report");
      }

      latch.countDown();
    }
  }

  private static final long serialVersionUID = 7139392253403373132L;

  // NOTE ExtraTrees uses the dynamic reflection, so extening it the new
  // getter/setter paris are automatically picked up
  private ExtraTree baseTree = new ExtraTree();
  private boolean useDefaultSelectionCount = true;

  private boolean useDefaultStopSize = true;
  private CategoricalData predicting;

  private ExtraTree[] forrest;

  private int forrestSize;

  /**
   * Creates a new Extremely Randomized Trees learner
   */
  public ERTrees() {
    this(100);
  }

  /**
   * Creates a new Extremely Randomized Trees learner
   *
   * @param forrestSize
   *          the number of trees to construct
   */
  public ERTrees(final int forrestSize) {
    this.forrestSize = forrestSize;
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    final CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());

    for (final ExtraTree tree : forrest) {
      cr.incProb(tree.classify(data).mostLikely(), 1.0);
    }
    cr.normalize();
    return cr;

  }

  @Override
  public ERTrees clone() {
    final ERTrees clone = new ERTrees();
    clone.forrestSize = forrestSize;
    clone.useDefaultSelectionCount = useDefaultSelectionCount;
    clone.useDefaultStopSize = useDefaultStopSize;
    clone.baseTree = baseTree.clone();
    if (predicting != null) {
      clone.predicting = predicting.clone();
    }
    if (forrest != null) {
      clone.forrest = new ExtraTree[forrest.length];
      for (int i = 0; i < forrest.length; i++) {
        clone.forrest[i] = forrest[i].clone();
      }
    }

    return clone;
  }

  private void doTraining(final ExecutorService threadPool, final DataSet dataSet) throws FailedToFitException {
    forrest = new ExtraTree[forrestSize];
    final int chunkSize = forrestSize / SystemInfo.LogicalCores;
    int extra = forrestSize % SystemInfo.LogicalCores;

    int planted = 0;

    final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
    while (planted < forrestSize) {
      final int start = planted;
      int end = start + chunkSize;
      if (extra-- > 0) {
        end++;
      }
      planted = end;
      threadPool.submit(new ForrestPlanter(start, end, dataSet, latch));
    }

    try {
      latch.await();
    } catch (final InterruptedException ex) {
      throw new FailedToFitException(ex);
    }
  }

  public int getForrestSize() {
    return forrestSize;
  }

  @Override
  public TreeNodeVisitor getTreeNodeVisitor() {
    throw new UnsupportedOperationException("Can not get the tree node vistor becase ERTrees is really a ensemble");
  }

  /**
   * Returns if the default heuristic for the selection count is used
   *
   * @return if the default heuristic for the selection count is used
   */
  public boolean getUseDefaultSelectionCount() {
    return useDefaultSelectionCount;
  }

  /**
   * Returns if the default heuristic for the stop size is used
   *
   * @return if the default heuristic for the stop size is used
   */
  public boolean getUseDefaultStopSize() {
    return useDefaultStopSize;
  }

  @Override
  public double regress(final DataPoint data) {
    double mean = 0.0;
    for (final ExtraTree tree : forrest) {
      mean += tree.regress(data);
    }
    return mean / forrest.length;
  }

  public void setForrestSize(final int forrestSize) {
    this.forrestSize = forrestSize;
  }

  /**
   * Sets whether or not to use the default heuristic for the number of random
   * features to select as candidates for each node. If <tt>true</tt> the value
   * of selectionCount will be modified during training, using sqrt(n) features
   * for classification and all features for regression. Otherwise, whatever
   * value set before hand will be used.
   *
   * @param useDefaultSelectionCount
   *          whether or not to use the heuristic version
   */
  public void setUseDefaultSelectionCount(final boolean useDefaultSelectionCount) {
    this.useDefaultSelectionCount = useDefaultSelectionCount;
  }

  /**
   * Sets whether or not to us the default heuristic for the number of points to
   * force a new node to be a leaf. If <tt>true</tt> the value for stopSize will
   * be altered during training, set to 2 for classification and 5 for
   * regression. Otherwise, whatever value set beforehand will be used.
   *
   * @param useDefaultStopSize
   *          whether or not to use the heuristic version
   */
  public void setUseDefaultStopSize(final boolean useDefaultStopSize) {
    this.useDefaultStopSize = useDefaultStopSize;
  }

  @Override
  public boolean supportsWeightedData() {
    return true;
  }

  @Override
  public void train(final RegressionDataSet dataSet) {
    train(dataSet, new FakeExecutor());
  }

  @Override
  public void train(final RegressionDataSet dataSet, final ExecutorService threadPool) {
    if (useDefaultSelectionCount) {
      baseTree.setSelectionCount(dataSet.getNumFeatures());
    }
    if (useDefaultStopSize) {
      baseTree.setStopSize(5);
    }

    doTraining(threadPool, dataSet);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    trainC(dataSet, new FakeExecutor());
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    if (useDefaultSelectionCount) {
      baseTree.setSelectionCount((int) max(round(sqrt(dataSet.getNumFeatures())), 1));
    }
    if (useDefaultStopSize) {
      baseTree.setStopSize(2);
    }

    predicting = dataSet.getPredicting();

    doTraining(threadPool, dataSet);
  }
}
