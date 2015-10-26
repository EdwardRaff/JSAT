package jsat.classifiers.linear;

import static java.lang.Math.*;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;

/**
 * Implements the iterative and single threaded stochastic solver for 
 * L<sub>1</sub> regularized linear regression problems SMIDAS (Stochastic 
 * Mirror Descent Algorithm mAde Sparse). It performs very well when the number 
 * of features is large relative to or greater than the number of data points. 
 * It also works decently on smaller sparse data sets. <br>
 * Using the squared loss is equivalent to LASSO regression, and the LOG loss 
 * is equivalent to logistic regression. <br>
 * <br>
 * Note: This implementation requires all feature values to be in the range 
 * [-1, 1]. By default scaling is performed to [0,1] to preserve sparseness. If
 * your data is dense or has negative and positive feature values, scaling to 
 * [-1, 1] may perform better. 
 * See {@link #setReScale(boolean) }<br>
 * <br>
 * See:<br>
 * <a href="http://eprints.pascal-network.org/archive/00005418/">Shalev-Shwartz,
 * S.,&amp;Tewari, A. (2009). <i>Stochastic Methods for L<sub>1</sub>-regularized 
 * Loss Minimization</i>. 26th International Conference on Machine Learning 
 * (Vol. 12, pp. 929–936).</a>
 * 
 * @author Edward Raff
 */
public class SMIDAS extends StochasticSTLinearL1
{

	private static final long serialVersionUID = -4888083541600164597L;
	private double eta;
    
    /**
     * Creates a new SMIDAS learner
     * @param eta the learning rate for each iteration
     */
    public SMIDAS(final double eta)
    {
        this(eta, DEFAULT_EPOCHS, DEFAULT_REG, DEFAULT_LOSS);
    }

    /**
     * Creates a new SMIDAS learner
     * @param eta the learning rate for each iteration
     * @param epochs the number of learning iterations
     * @param lambda the regularization penalty
     * @param loss the loss function to use
     */
    public SMIDAS(final double eta, final int epochs, final double lambda, final Loss loss)
    {
        this(eta, epochs, lambda, loss, true);
    }

    /**
     * Creates a new SMIDAS learner
     * @param eta the learning rate for each iteration
     * @param epochs the number of learning iterations
     * @param lambda the regularization penalty
     * @param loss the loss function to use
     * @param reScale whether or not to rescale the feature values 
     */
    public SMIDAS(final double eta, final int epochs, final double lambda, final Loss loss, final boolean reScale)
    {
        setEta(eta);
        setEpochs(epochs);
        setLambda(lambda);
        setLoss(loss);
        setReScale(reScale);
    }

    /**
     * Sets the learning rate used during training
     * @param eta the learning rate to use
     */
    public void setEta(final double eta)
    {
        if(Double.isNaN(eta) || Double.isInfinite(eta) || eta <= 0) {
          throw new ArithmeticException("convergence parameter must be a positive value");
        }
        this.eta = eta;
    }

    /**
     * Returns the current learning rate used during training
     * @return the learning rate in use
     */
    public double getEta()
    {
        return eta;
    }
    
    @Override
    public CategoricalResults classify(final DataPoint data)
    {
        if(w == null) {
          throw new UntrainedModelException("Model has not been trained");
        }
        final Vec x = data.getNumericalValues();
        return loss.classify(wDot(x));
    }

    @Override
    public double regress(final DataPoint data)
    {
        if(w == null) {
          throw new UntrainedModelException("Model has not been trained");
        }
        final Vec x = data.getNumericalValues();
        return loss.regress(wDot(x));
    }
    
    @Override
    public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet)
    {
        if(dataSet.getNumNumericalVars() < 3) {
          throw new FailedToFitException("SMIDAS requires at least 3 features");
        } else if(dataSet.getClassSize() != 2) {
          throw new FailedToFitException("SMIDAS only supports binary classification problems");
        }
        final Vec[] x = setUpVecs(dataSet);
        
        final Vec obvMinV = DenseVector.toDenseVec(obvMin);
        final Vec obvMaxV = DenseVector.toDenseVec(obvMax);
        final Vec multitpliers = new DenseVector(obvMaxV.length());
        multitpliers.mutableAdd(maxScaled-minScaled);
        multitpliers.mutablePairwiseDivide(obvMaxV.subtract(obvMinV));
        
        boolean allZeroMins = true;
        for(final double min : obvMin) {
          if (min != 0) {
            allZeroMins = false;
          }
        }
        final double[] target = new double[x.length];
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            //Copy and scale each value
            if(allZeroMins && minScaled == 0.0)
            {
                x[i].mutablePairwiseMultiply(multitpliers);
            }
            else//destroy all sparsity and our dreams
            {
                x[i] = x[i].subtract(obvMinV);
                x[i].mutablePairwiseMultiply(multitpliers);
                x[i].mutableAdd(minScaled);
            }
            target[i] = dataSet.getDataPointCategory(i)*2-1;
        }
        train(x, target);
    }

    @Override
    public void train(final RegressionDataSet dataSet, final ExecutorService threadPool)
    {
        train(dataSet);
    }

    @Override
    public void train(final RegressionDataSet dataSet)
    {
        if(dataSet.getNumNumericalVars() < 3) {
          throw new FailedToFitException("SMIDAS requires at least 3 features");
        }
        final Vec[] x = setUpVecs(dataSet);
        
        final Vec obvMinV = DenseVector.toDenseVec(obvMin);
        final Vec obvMaxV = DenseVector.toDenseVec(obvMax);
        final Vec multitpliers = new DenseVector(obvMaxV.length());
        multitpliers.mutableAdd(maxScaled-minScaled);
        multitpliers.mutablePairwiseDivide(obvMaxV.subtract(obvMinV));
        
        boolean allZeroMins = true;
        for(final double min : obvMin) {
          if (min != 0) {
            allZeroMins = false;
          }
        }
        final double[] target = new double[x.length];
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            if(allZeroMins && minScaled == 0.0)
            {
                x[i].mutablePairwiseMultiply(multitpliers);
            }
            else
            {
                //Copy and scale each value
                x[i] = x[i].subtract(obvMinV);
                x[i].mutablePairwiseMultiply(multitpliers);
                x[i].mutableAdd(minScaled);
            }
            target[i] = dataSet.getTargetValue(i);
        }
        
        train(x, target);
    }
    
    private void train(final Vec[] x, final double[] y)
    {
        final int m = x.length;
        final int d = x[0].length();
        final double p = 2*Math.log(d);
        
        final Vec theta = new DenseVector(d);
        double theta_bias = 0;
        double lossScore = 0;
        w = new DenseVector(d);
        
        final Random rand = new Random();
        
        for(int t = 0; t < epochs; t++)
        {
            final int i = rand.nextInt(m);
            
            lossScore = loss.deriv(w.dot(x[i])+bias, y[i]);
            
            theta.mutableSubtract(eta*lossScore, x[i]);
            theta_bias -= eta*lossScore;

            for(final IndexValue iv : theta)
            {
                final int j = iv.getIndex();
                final double theta_j = iv.getValue();//theta.get(j);
                theta.set(j, signum(theta_j)*max(0, abs(theta_j)-eta*lambda));
            }
            theta_bias = signum(theta_bias)*max(0, abs(theta_bias)-eta*lambda);
            
            final double thetaNorm = theta.pNorm(p);
            if(thetaNorm > 0)
            {
                //w = f^-1(theta)
                final double logThetaNorm = log(thetaNorm);
                for(int j = 0; j < w.length(); j++)
                {
                    final double theta_j = theta.get(j);
                    w.set(j, signum(theta_j) * exp((p-1) * log(abs(theta_j)) - (p-2) * logThetaNorm));
                }
                bias = signum(theta_bias)*exp((p-1) * log(abs(theta_bias)) - (p-2) * logThetaNorm);
            }
            else
            {
                theta.zeroOut();
                theta_bias = 0;
                w.zeroOut();
                bias = 0;
            }
        }
    }
    
    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }
    
    @Override
    public SMIDAS clone()
    {
        final SMIDAS clone = new SMIDAS(eta, epochs, lambda, loss, reScale);
        if(this.w != null) {
          clone.w = this.w.clone();
        }
        clone.bias = this.bias;
        clone.minScaled = this.minScaled;
        clone.maxScaled = this.maxScaled;
        if(this.obvMin != null) {
          clone.obvMin = Arrays.copyOf(this.obvMin, this.obvMin.length);
        }
        if(this.obvMax != null) {
          clone.obvMax = Arrays.copyOf(this.obvMax, this.obvMax.length);
        }
        return clone;
    }

    private Vec[] setUpVecs(final DataSet dataSet)
    {
        obvMin = new double[dataSet.getNumNumericalVars()];
        Arrays.fill(obvMin, Double.POSITIVE_INFINITY);
        obvMax = new double[dataSet.getNumNumericalVars()];
        Arrays.fill(obvMax, Double.NEGATIVE_INFINITY);
        final Vec[] x = new Vec[dataSet.getSampleSize()];    
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            x[i] = dataSet.getDataPoint(i).getNumericalValues();

            for(final IndexValue iv : x[i])
            {
                final int j = iv.getIndex();
                final double v = iv.getValue();
                obvMin[j] = Math.min(obvMin[j], v);
                obvMax[j] = Math.max(obvMax[j], v);
            }
        }
        
        if(x[0].isSparse()) {//Assume implicit min zeros from sparsity
          for (int i = 0; i < obvMin.length; i++) {
            obvMin[i] = Math.min(obvMin[i], 0);
          }
        }
        
        if(!reScale)
        {
            for(final double min : obvMin) {
              if (min < -1) {
                throw new FailedToFitException("Values must be in the range [-1,1], " + min + " violation encountered");
              }
            }
            for(final double max : obvMax) {
              if (max > 1) {
                throw new FailedToFitException("Values must be in the range [-1,1], " + max + " violation encountered");
              }
            }
            
        }
        return x;
    }
    
}
