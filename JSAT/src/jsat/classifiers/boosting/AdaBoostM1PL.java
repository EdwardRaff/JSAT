package jsat.classifiers.boosting;

import static jsat.utils.SystemInfo.LogicalCores;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.MajorityVote;
import jsat.exceptions.FailedToFitException;
import jsat.utils.DoubleList;
import jsat.utils.IndexTable;

/**
 * An extension to the original AdaBoostM1 algorithm for parallel training. This
 * comes at an increase in classification time. <br>
 * See: <i>Scalable and Parallel Boosting with MapReduce</i>, Indranil Palit and
 * Chandan K. Reddy, IEEE Transactions on Knowledge and Data Engineering
 *
 *
 * @author Edward Raff
 */
public class AdaBoostM1PL extends AdaBoostM1 {

  private static final long serialVersionUID = 1027211688101553766L;

  public AdaBoostM1PL(final Classifier weakLearner, final int maxIterations) {
    super(weakLearner, maxIterations);
  }

  @Override
  public AdaBoostM1PL clone() {
    final AdaBoostM1PL copy = new AdaBoostM1PL(getWeakLearner().clone(), getMaxIterations());
    if (hypWeights != null) {
      copy.hypWeights = new DoubleList(hypWeights);
    }
    if (hypoths != null) {
      copy.hypoths = new ArrayList<Classifier>(hypoths.size());
      for (int i = 0; i < hypoths.size(); i++) {
        copy.hypoths.add(hypoths.get(i).clone());
      }
    }
    if (predicting != null) {
      copy.predicting = predicting.clone();
    }
    return copy;
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    super.trainC(dataSet, null);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    predicting = dataSet.getPredicting();

    // Contains the Boostings we performed on subsets of the data
    final List<Future<AdaBoostM1>> futureBoostings = new ArrayList<Future<AdaBoostM1>>(LogicalCores);

    // We want an even, random split of the data into groups for each learner,
    // the CV set does that for us!
    final List<ClassificationDataSet> subSets = dataSet.cvSet(LogicalCores);
    for (int i = 0; i < LogicalCores; i++) {
      final AdaBoostM1 learner = new AdaBoostM1(getWeakLearner().clone(), getMaxIterations());
      final ClassificationDataSet subDataSet = subSets.get(i);
      futureBoostings.add(threadPool.submit(new Callable<AdaBoostM1>() {

        @Override
        public AdaBoostM1 call() throws Exception {
          learner.trainC(subDataSet);
          return learner;
        }
      }));
    }

    try {
      final List<AdaBoostM1> boosts = new ArrayList<AdaBoostM1>(LogicalCores);
      final List<List<Double>> boostWeights = new ArrayList<List<Double>>(LogicalCores);
      final List<List<Classifier>> boostWeakLearners = new ArrayList<List<Classifier>>(LogicalCores);
      // Contains the tables to view the weights in sorted order
      final List<IndexTable> sortedViews = new ArrayList<IndexTable>(LogicalCores);
      for (final Future<AdaBoostM1> futureBoost : futureBoostings) {
        final AdaBoostM1 boost = futureBoost.get();
        boosts.add(boost);
        sortedViews.add(new IndexTable(boost.hypWeights));
        boostWeights.add(boost.hypWeights);
        boostWeakLearners.add(boost.hypoths);

      }

      // Now we merge the results into our new classifer
      final int T = boosts.get(0).getMaxIterations();
      hypoths = new ArrayList<Classifier>(T);
      hypWeights = new DoubleList(T);
      for (int i = 0; i < T; i++) {
        final Classifier[] toMerge = new Classifier[LogicalCores];
        double weight = 0.0;
        for (int m = 0; m < LogicalCores; m++) {
          final int mSortedIndex = sortedViews.get(m).index(i);
          toMerge[m] = boostWeakLearners.get(m).get(mSortedIndex);
          weight += boostWeights.get(m).get(mSortedIndex);
        }
        weight /= LogicalCores;
        hypWeights.add(weight);
        hypoths.add(new MajorityVote(toMerge));
      }
    } catch (final Exception ex) {
      throw new FailedToFitException(ex);
    }
  }

}
