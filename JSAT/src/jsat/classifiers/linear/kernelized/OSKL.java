package jsat.classifiers.linear.kernelized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import jsat.DataSet;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.kernels.KernelTrick;
import jsat.linear.Vec;
import jsat.lossfunctions.HingeLoss;
import jsat.lossfunctions.LogisticLoss;
import jsat.lossfunctions.LossC;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.random.XORWOW;

/**
 * Online Sparse Kernel Learning by Sampling and Smooth Losses (OSKL) is an
 * online algorithm for learning sparse kernelized solutions to binary
 * classification problems. The number of support vectors is controlled by a a
 * sparsity parameter {@link #setG(double) G} and a specified {@link LossC loss
 * function}. The number of support vectors is bounded by the cumulative loss of
 * the loss function used. <br>
 * <br>
 * The OSKL algorithm is designed for use with smooth loss functions such as the
 * {@link LogisticLoss logistic loss}. However, it can work with non-smooth loss
 * functions such as the {@link HingeLoss hinge loss}. <br>
 * <br>
 * See: Zhang, L., Yi, J., Jin, R., Lin, M.,&amp;He, X. (2013). <i>Online Kernel
 * Learning with a Near Optimal Sparsity Bound</i>. In S. Dasgupta&amp;D.
 * Mcallester (Eds.), Proceedings of the 30th International Conference on
 * Machine Learning (ICML-13) (Vol. 28, pp. 621–629). JMLR Workshop and
 * Conference Proceedings.
 *
 * @author Edward Raff
 */
public class OSKL extends BaseUpdateableClassifier implements BinaryScoreClassifier, Parameterized {

  private static final long serialVersionUID = 4207594016856230134L;

  /**
   * Guesses the distribution to use for the R parameter
   *
   * @param d
   *          the dataset to get the guess for
   * @return the guess for the R parameter
   * @see #setR(double)
   */
  public static Distribution guessR(final DataSet d) {
    return new LogUniform(1, 1e5);
  }

  @ParameterHolder
  private KernelTrick k;
  private double eta;
  private double R;
  private double G;
  private double curSqrdNorm;
  private final LossC lossC;

  private boolean useAverageModel = true;
  // Data used for capturing the average
  private int t;
  /**
   * Last time alphaAverage was updated
   */
  private int last_t;
  private int burnIn;

  /**
   * Store the average of the weights over time
   */
  private DoubleList alphaAveraged;
  private List<Vec> vecs;
  private DoubleList alphas;
  private DoubleList inputKEvals;
  private List<Double> accelCache;

  private Random rand;

  /**
   * Creates a new OSKL learner using the {@link LogisticLoss}. The parameters
   * {@link #setG(double) } and {@link #setEta(double) } are set based on the
   * original papers suggestions to produced a less sparse model that should be
   * more accurate
   *
   * @param k
   *          the kernel to use
   * @param R
   *          the maximum allowed norm for the model
   */
  public OSKL(final KernelTrick k, final double R) {
    this(k, 0.9, 1, R);
  }

  /**
   * Creates a new OSKL learner using the {@link LogisticLoss}
   *
   * @param k
   *          the kernel to use
   * @param eta
   *          the learning rate to use
   * @param G
   *          the sparsification parameter
   * @param R
   *          the maximum allowed norm for the model
   */
  public OSKL(final KernelTrick k, final double eta, final double G, final double R) {
    this(k, eta, G, R, new LogisticLoss());
  }

  /**
   * Creates a new OSKL learner
   *
   * @param k
   *          the kernel to use
   * @param eta
   *          the learning rate to use
   * @param G
   *          the sparsification parameter
   * @param R
   *          the maximum allowed norm for the model
   * @param lossC
   *          the loss function to use
   */
  public OSKL(final KernelTrick k, final double eta, final double G, final double R, final LossC lossC) {
    setKernel(k);
    setEta(eta);
    setR(R);
    setG(G);
    this.lossC = lossC;
  }

  /**
   * Copy constructor
   *
   * @param toCopy
   *          the object to copy
   */
  public OSKL(final OSKL toCopy) {
    k = toCopy.k.clone();
    eta = toCopy.eta;
    R = toCopy.R;
    G = toCopy.G;
    curSqrdNorm = toCopy.curSqrdNorm;
    lossC = toCopy.lossC.clone();
    t = toCopy.t;
    last_t = toCopy.last_t;
    useAverageModel = toCopy.useAverageModel;
    burnIn = toCopy.burnIn;
    if (toCopy.vecs != null) {
      vecs = new ArrayList<Vec>();
      for (final Vec v : toCopy.vecs) {
        vecs.add(v.clone());
      }
      alphas = new DoubleList(toCopy.alphas);
      alphaAveraged = new DoubleList(toCopy.alphaAveraged);
      inputKEvals = new DoubleList(toCopy.inputKEvals);
    }
    if (toCopy.accelCache != null) {
      accelCache = new DoubleList(toCopy.accelCache);
    }

    rand = new XORWOW();
  }

  @Override
  public CategoricalResults classify(final DataPoint data) {
    final Vec x = data.getNumericalValues();
    return lossC.getClassification(score(x, k.getQueryInfo(x)));
  }

  @Override
  public OSKL clone() {
    return new OSKL(this);
  }

  /**
   * Returns the number of burn in rounds
   *
   * @return the number of burn in rounds
   */
  public int getBurnIn() {
    return burnIn;
  }

  /**
   * Returns the learning rate in use
   *
   * @return the learning rate in use
   */
  public double getEta() {
    return eta;
  }

  /**
   * Returns the sparsification parameter
   *
   * @return the sparsification parameter
   */
  public double getG() {
    return G;
  }

  /**
   * Returns the kernel to use
   *
   * @return the kernel to use
   */
  public KernelTrick getKernel() {
    return k;
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
   * Returns the maximum allowed norm for the model learned
   *
   * @return the maximum allowed norm for the model learned
   */
  public double getR() {
    return R;
  }

  @Override
  public double getScore(final DataPoint dp) {
    final Vec x = dp.getNumericalValues();
    return score(x, k.getQueryInfo(x));
  }

  /**
   * Returns the number of data points accepted as support vectors
   *
   * @return the number of support vectors in the model
   */
  public int getSupportVectorCount() {
    if (vecs == null) {
      return 0;
    } else {
      return vecs.size();
    }
  }

  /**
   * Returns {@code true} if the average of all models is being used, or
   * {@code false} if the last model is used
   *
   * @return {@code true} if the average of all models is being used, or
   *         {@code false} if the last model is used
   */
  public boolean isUseAverageModel() {
    return useAverageModel;
  }

  private double score(final Vec x, final List<Double> qi) {
    DoubleList alphToUse;
    if (useAverageModel && t > burnIn) {
      updateAverage();
      alphToUse = alphaAveraged;
    } else {
      alphToUse = alphas;
    }
    return k.evalSum(vecs, accelCache, alphToUse.getBackingArray(), x, qi, 0, alphToUse.size());
  }

  /**
   * Computes the score and saves the results of the kernel computations in
   * {@link #inputKEvals}. The first value in the list will be the self kernel
   * product
   *
   * @param x
   *          the input vector
   * @param qi
   *          the query information for the vector
   * @return the dot product in the kernel space
   */
  private double scoreSaveEval(final Vec x, final List<Double> qi) {
    inputKEvals.clear();
    inputKEvals.add(k.eval(0, 0, Arrays.asList(x), qi));
    double sum = 0;
    for (int i = 0; i < alphas.size(); i++) {
      final double k_ix = k.eval(i, x, qi, vecs, accelCache);
      inputKEvals.add(k_ix);
      sum += alphas.getD(i) * k_ix;
    }
    return sum;
  }

  /**
   * Sets the number of update calls to consider as part of the "burn in" phase.
   * The averaging of the model will not start until after the burn in phase.
   * <br>
   * If the classification or score is requested before the burn in phase is
   * completed, the latest model will be used as is.
   *
   * @param burnIn
   *          the number of updates to ignore before averaging. Must be non
   *          negative.
   */
  public void setBurnIn(final int burnIn) {
    if (burnIn < 0) {
      throw new IllegalArgumentException("Burn in must be non negative, not " + burnIn);
    }
    this.burnIn = burnIn;
  }

  ;

  /**
   * Sets the learning rate to use for training. The original paper suggests
   * setting &eta; = 0.9/{@link #setG(double) G}
   *
   * @param eta
   *          the positive learning rate to use
   */
  public void setEta(final double eta) {
    if (eta <= 0 || Double.isNaN(eta) || Double.isInfinite(eta)) {
      throw new IllegalArgumentException("Eta must be positive, not " + eta);
    }
    this.eta = eta;
  }

  /**
   * Sets the sparsification parameter G. Increasing G reduces the number of
   * updates to the model, which increases sparsity but may reduce accuracy.
   * Decreasing G increases the update rate reducing sparsity. The original
   * paper tests values of G &isin; {1, 2, 4, 10}
   *
   * @param G
   *          the sparsification parameter in [1, &infin;)
   */
  public void setG(final double G) {
    if (G < 1 || Double.isInfinite(G) || Double.isNaN(G)) {
      throw new IllegalArgumentException("G must be in [1, Infinity), not " + G);
    }
    this.G = G;
  }

  /**
   * Sets the kernel to use
   *
   * @param k
   *          the kernel to use
   */
  public void setKernel(final KernelTrick k) {
    this.k = k;
  }

  /**
   * Sets the maximum allowed norm of the model. The original paper suggests
   * values in the range 10<sup>x</sup> for <i>x</i> &isin; {0, 1, 2, 3, 4, 5}.
   *
   * @param R
   *          the maximum allowed norm for the model
   */
  public void setR(final double R) {
    if (R <= 0 || Double.isNaN(R) || Double.isInfinite(R)) {
      throw new IllegalArgumentException("R must be positive, not " + R);
    }
    this.R = R;
  }

  @Override
  public void setUp(final CategoricalData[] categoricalAttributes, final int numericAttributes,
      final CategoricalData predicting) {
    rand = new XORWOW();
    vecs = new ArrayList<Vec>();
    alphas = new DoubleList();
    alphaAveraged = new DoubleList();
    t = 0;
    last_t = 0;
    inputKEvals = new DoubleList();
    if (k.supportsAcceleration()) {
      accelCache = new DoubleList();
    } else {
      accelCache = null;
    }
    curSqrdNorm = 0;
  }

  /**
   * Sets whether or not the average of all intermediate models is used or if
   * the most recent model is used when performing classification
   *
   * @param useAverageModel
   *          {@code true} to use the average model, {@code false} to use the
   *          last model update
   */
  public void setUseAverageModel(final boolean useAverageModel) {
    this.useAverageModel = useAverageModel;
  }

  @Override
  public boolean supportsWeightedData() {
    return false;
  }

  @Override
  public void update(final DataPoint dataPoint, final int targetClass) {
    final Vec x_t = dataPoint.getNumericalValues();
    final List<Double> qi = k.getQueryInfo(x_t);
    final double score = scoreSaveEval(x_t, qi);
    final double y_t = targetClass * 2 - 1;
    // 4: Compute the derivative ℓ′(yt, ft(xt))
    final double lossD = lossC.getDeriv(score, y_t);
    t++;
    // Step 5: Sample a binary random variable Zt with
    if (rand.nextDouble() > Math.abs(lossD) / G) {
      return;// "failed", no update
    }
    final double alpha_t = -eta * Math.signum(lossD) * G;
    // Update the squared norm
    curSqrdNorm += alpha_t * alpha_t * inputKEvals.getD(0);
    for (int i = 0; i < alphas.size(); i++) {
      curSqrdNorm += 2 * alpha_t * alphas.getD(i) * inputKEvals.getD(i + 1);
    }
    // add values
    alphas.add(alpha_t);
    vecs.add(x_t);
    if (accelCache != null) {
      accelCache.addAll(qi);
    }
    // update online alpha averages for current & old SVs
    alphaAveraged.add(0.0);// implicit zero for time we didn't have new SVs
    updateAverage();
    // project alphas to maintain norm if needed
    if (curSqrdNorm > R * R) {
      final double coeff = R / Math.sqrt(curSqrdNorm);
      alphas.getVecView().mutableMultiply(coeff);
      curSqrdNorm *= coeff * coeff;
    }
  }

  /**
   * Updates the average model to reflect the current time average
   */
  private void updateAverage() {
    if (t == last_t || t < burnIn) {
      return;
    } else if (last_t < burnIn) // first update since done burning
    {
      for (int i = 0; i < alphaAveraged.size(); i++) {
        alphaAveraged.set(i, alphas.get(i));
      }
    }
    final double w = t - last_t;// time elapsed
    for (int i = 0; i < alphaAveraged.size(); i++) {
      final double delta = alphas.getD(i) - alphaAveraged.getD(i);
      alphaAveraged.set(i, alphaAveraged.getD(i) + delta * w / t);
    }
    last_t = t;// average done
  }
}
