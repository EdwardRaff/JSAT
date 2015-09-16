package jsat.classifiers.neuralnetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;

import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.neuralnetwork.activations.ActivationLayer;
import jsat.classifiers.neuralnetwork.activations.ReLU;
import jsat.classifiers.neuralnetwork.activations.SoftmaxLayer;
import jsat.classifiers.neuralnetwork.initializers.ConstantInit;
import jsat.classifiers.neuralnetwork.initializers.GaussianNormalInit;
import jsat.classifiers.neuralnetwork.regularizers.Max2NormRegularizer;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.math.optimization.stochastic.AdaDelta;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * This class provides a neural network based on Geoffrey Hinton's <b>D</b>eep
 * <b>Re</b>ctified <b>D</b>ropout <b>N</b>ets. It is parameterized to be
 * "simpler" in that the default batch size and gradient updating method should
 * require no tuning to get decent results<br>
 * <br>
 * NOTE: Training neural networks is computationally expensive, you may want to
 * consider a GPU implementation from another source.
 *
 * @author Edward Raff
 */
public class DReDNetSimple implements Classifier, Parameterized {

  private static final long serialVersionUID = -342281027279571332L;
  private SGDNetworkTrainer network;
  private int[] hiddenSizes;
  private int batchSize = 256;
  private int epochs = 100;

  /**
   * Create a new DReDNet that uses the specified number of hidden layers. A
   * batch size of 256 and 100 epochs will be used.
   *
   * @param hiddenLayerSizes
   *          the length indicates the number of hidden layers, and the value in
   *          each index is the number of neurons in that layer
   */
  public DReDNetSimple(final int... hiddenLayerSizes) {
    setHiddenSizes(hiddenLayerSizes);
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    final Vec x = data.getNumericalValues();
    final Vec y = network.feedfoward(x);
    return new CategoricalResults(y.arrayCopy());
  }

  @Override
  public DReDNetSimple clone() {
    final DReDNetSimple clone = new DReDNetSimple(hiddenSizes);
    if (network != null) {
      clone.network = network.clone();
    }
    clone.batchSize = batchSize;
    clone.epochs = epochs;
    return clone;
  }

  /**
   *
   * @return the number of data points to use for one gradient computation
   */
  public int getBatchSize() {
    return batchSize;
  }

  /**
   *
   * @return the number of training iterations through the data set
   */
  public int getEpochs() {
    return epochs;
  }

  /**
   *
   * @return the array of hidden layer sizes
   */
  public int[] getHiddenSizes() {
    return hiddenSizes;
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
   * Sets the batch size for updates
   *
   * @param batchSize
   *          the number of items to compute the gradient from
   */
  public void setBatchSize(final int batchSize) {
    this.batchSize = batchSize;
  }

  /**
   * Sets the number of epochs to perform
   *
   * @param epochs
   *          the number of training iterations through the whole data set
   */
  public void setEpochs(final int epochs) {
    if (epochs <= 0) {
      throw new IllegalArgumentException("Number of epochs must be positive");
    }
    this.epochs = epochs;
  }

  /**
   * Sets the hidden layer sizes for this network. The size of the array is the
   * number of hidden layers and the value in each index denotes the size of
   * that layer.
   *
   * @param hiddenSizes
   */
  public void setHiddenSizes(final int[] hiddenSizes) {
    for (int i = 0; i < hiddenSizes.length; i++) {
      if (hiddenSizes[i] <= 0) {
        throw new IllegalArgumentException(
            "Hidden layer " + i + " must contain a positive number of neurons, not " + hiddenSizes[i]);
      }
    }
    this.hiddenSizes = Arrays.copyOf(hiddenSizes, hiddenSizes.length);
  }

  private void setup(final ClassificationDataSet dataSet) {
    network = new SGDNetworkTrainer();
    final int[] sizes = new int[hiddenSizes.length + 2];
    sizes[0] = dataSet.getNumNumericalVars();
    System.arraycopy(hiddenSizes, 0, sizes, 1, hiddenSizes.length);
    sizes[sizes.length - 1] = dataSet.getClassSize();
    network.setLayerSizes(sizes);

    final List<ActivationLayer> activations = new ArrayList<ActivationLayer>(hiddenSizes.length + 2);
    for (final int size : hiddenSizes) {
      activations.add(new ReLU());
    }
    activations.add(new SoftmaxLayer());
    network.setLayersActivation(activations);
    network.setRegularizer(new Max2NormRegularizer(25));
    network.setWeightInit(new GaussianNormalInit(1e-2));
    network.setBiasInit(new ConstantInit(0.1));

    network.setEta(1.0);
    network.setGradientUpdater(new AdaDelta());

    network.setup();
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet) {
    trainC(dataSet, null);
  }

  @Override
  public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool) {
    setup(dataSet);

    final List<Vec> X = dataSet.getDataVectors();
    final List<Vec> Y = new ArrayList<Vec>(dataSet.getSampleSize());
    for (int i = 0; i < dataSet.getSampleSize(); i++) {
      final SparseVector sv = new SparseVector(dataSet.getClassSize(), 1);
      sv.set(dataSet.getDataPointCategory(i), 1.0);
      Y.add(sv);
    }
    final IntList randOrder = new IntList(X.size());
    ListUtils.addRange(randOrder, 0, X.size(), 1);
    final List<Vec> Xmini = new ArrayList<Vec>(batchSize);
    final List<Vec> Ymini = new ArrayList<Vec>(batchSize);

    for (int epoch = 0; epoch < epochs; epoch++) {
      final long start = System.currentTimeMillis();
      double epochError = 0;
      Collections.shuffle(randOrder);
      for (int i = 0; i < X.size(); i += batchSize) {
        final int to = Math.min(i + batchSize, X.size());
        Xmini.clear();
        Ymini.clear();
        for (int j = i; j < to; j++) {
          Xmini.add(X.get(j));
          Ymini.add(Y.get(j));
        }

        double localErr;
        if (threadPool != null) {
          localErr = network.updateMiniBatch(Xmini, Ymini, threadPool);
        } else {
          localErr = network.updateMiniBatch(Xmini, Ymini);
        }
        epochError += localErr;
      }
      final long end = System.currentTimeMillis();
      // System.out.println("Epoch " + epoch + " had error " + epochError + "
      // took " + (end-start)/1000.0 + " seconds");
    }

    network.finishUpdating();
  }

}
