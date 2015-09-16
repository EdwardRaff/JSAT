package jsat.clustering.evaluation;

import java.util.List;

import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.utils.DoubleList;

/**
 * Normalized Mutual Information (NMI) is a measure to evaluate a cluster based
 * on the true class labels for the data set. The NMI normally returns a value
 * in [0, 1], where 0 indicates the clustering appears random, and 1 indicate
 * the clusters perfectly match the class labels. To match the
 * {@link ClusterEvaluation} interface, the value returned by evaluate will be
 * 1.0-NMI . <br>
 * <b>NOTE:</b> Because the NMI needs to know the true class labels, only
 * {@link #evaluate(int[], jsat.DataSet) } will work, since it provides the data
 * set as an argument. The dataset given must be an instance of
 * {@link ClassificationDataSet}
 *
 * @author Edward Raff
 */
public class NormalizedMutualInformation implements ClusterEvaluation {

  @Override
  public NormalizedMutualInformation clone() {
    return new NormalizedMutualInformation();
  }

  @Override
  public double evaluate(final int[] designations, final DataSet dataSet) {
    if (!(dataSet instanceof ClassificationDataSet)) {
      throw new RuntimeException("NMI can only be calcuate for classification data sets");
    }
    final ClassificationDataSet cds = (ClassificationDataSet) dataSet;
    double nmiNumer = 0.0;
    double nmiC = 0.0;
    double nmiK = 0.0;

    final DoubleList kPriors = new DoubleList();

    for (int i = 0; i < cds.getSampleSize(); i++) {
      final int ki = designations[i];
      if (ki < 0) {// outlier, not clustered
        continue;
      }
      while (kPriors.size() <= ki) {
        kPriors.add(0.0);
      }
      kPriors.set(ki, kPriors.get(ki) + cds.getDataPoint(i).getWeight());
    }

    double N = 0.0;
    for (int i = 0; i < kPriors.size(); i++) {
      N += kPriors.get(i);
    }
    for (int i = 0; i < kPriors.size(); i++) {
      kPriors.set(i, kPriors.get(i) / N);
      final double pKi = kPriors.get(i);
      if (pKi > 0) {
        nmiK += -pKi * Math.log(pKi);
      }
    }

    final double[] cPriors = cds.getPriors();

    final double[][] ck = new double[cPriors.length][kPriors.size()];

    for (int i = 0; i < cds.getSampleSize(); i++) {
      final int ci = cds.getDataPointCategory(i);
      final int kj = designations[i];
      if (kj < 0) {// outlier, ignore
        continue;
      }

      ck[ci][kj] += cds.getDataPoint(i).getWeight();
    }

    for (int i = 0; i < cPriors.length; i++) {
      final double pCi = cPriors[i];
      if (pCi <= 0.0) {
        continue;
      }
      final double logPCi = Math.log(pCi);
      for (int j = 0; j < kPriors.size(); j++) {
        final double pKj = kPriors.get(j);
        if (pKj <= 0.0) {
          continue;
        }
        final double pCiKj = ck[i][j] / N;
        if (pCiKj <= 0.0) {
          continue;
        }
        nmiNumer += pCiKj * (Math.log(pCiKj) - Math.log(pKj) - logPCi);
      }
      nmiC += -pCi * logPCi;
    }

    return 1.0 - nmiNumer / ((nmiC + nmiK) / 2);
  }

  @Override
  public double evaluate(final List<List<DataPoint>> dataSets) {
    throw new UnsupportedOperationException(
        "NMI requires the true data set" + " labels, call evaluate(int[] designations, DataSet dataSet)" + " instead");
  }

}
