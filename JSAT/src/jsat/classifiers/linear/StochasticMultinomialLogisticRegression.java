package jsat.classifiers.linear;

import static java.lang.Math.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.SimpleWeightVectorModel;
import jsat.classifiers.*;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.ConstantVector;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.math.MathTricks;
import jsat.math.decayrates.DecayRate;
import jsat.math.decayrates.ExponetialDecay;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * This is a Stochastic implementation of Multinomial Logistic Regression. It 
 * supports regularization from several different priors, and performs prior 
 * updates in a lazy fashion to avoid destroying the sparsity of training 
 * inputs. 
 * <br>
 * Algorithm is based on the technical report:<br>
 * Carpenter, B. (2008). <i>Lazy Sparse Stochastic Gradient Descent for 
 * Regularized Mutlinomial Logistic Regression</i>. Retrieved from
 * http://lingpipe-blog.com/lingpipe-white-papers/
 * 
 * @author Edward Raff
 */
public class StochasticMultinomialLogisticRegression implements Classifier, Parameterized, SimpleWeightVectorModel
{   

	private static final long serialVersionUID = -492707881682847556L;
	private int epochs;
    private boolean clipping = true;
    private double regularization;
    private double tolerance = 1e-4;
    private double initialLearningRate;
    private double alpha = 0.5;
    private DecayRate learningRateDecay = new ExponetialDecay();
    private Prior prior;
    private boolean standardized = true;
    private boolean useBias = true;
    private int miniBatchSize = 1;
    
    private Vec[] B;
    private double[] biases;

    /**
     * Creates a new Stochastic Multinomial Logistic Regression object
     * @param initialLearningRate the initial learning rate to use
     * @param epochs the maximum number of training epochs to go through
     * @param regularization the scale factor applied to the regularization term
     * @param prior the prior to use for regularization
     */
    public StochasticMultinomialLogisticRegression(double initialLearningRate, int epochs, double regularization, Prior prior)
    {
        setEpochs(epochs);
        setRegularization(regularization);
        setInitialLearningRate(initialLearningRate);
        setPrior(prior);
    }

    /**
     * Creates a new Stochastic Multinomial Logistic Regression that uses a 
     * {@link Prior#GAUSSIAN} prior with a regularization scale of 1e-6. 
     * 
     * @param initialLearningRate the initial learning rate to use
     * @param epochs the maximum number of training epochs to go through
     */
    public StochasticMultinomialLogisticRegression(double initialLearningRate, int epochs)
    {
        this(initialLearningRate, epochs, 1e-6, Prior.GAUSSIAN);
    }

    /**
     * Creates a new Stochastic Multinomial Logistic Regression that uses a 
     * {@link Prior#GAUSSIAN} prior with a regularization scale of 1e-6. It 
     * will do at most 50 epochs with a learning rate of 0.1
     */
    public StochasticMultinomialLogisticRegression()
    {
        this(0.1, 50);
    }

    /**
     * Copy constructor
     * @param toClone the classifier to create a copy of
     */
    protected StochasticMultinomialLogisticRegression(StochasticMultinomialLogisticRegression toClone)
    {
        this.epochs = toClone.epochs;
        this.clipping = toClone.clipping;
        this.regularization = toClone.regularization;
        this.tolerance = toClone.tolerance;
        this.initialLearningRate = toClone.initialLearningRate;
        this.alpha = toClone.alpha;
        this.learningRateDecay = toClone.learningRateDecay;
        this.prior = toClone.prior;
        this.standardized = toClone.standardized;
        
        if(toClone.B != null)
        {
            this.B = new Vec[toClone.B.length];
            for(int i = 0; i < toClone.B.length; i++)
                this.B[i] = toClone.B[i].clone();
        }
        
        if(toClone.biases != null)
            this.biases = Arrays.copyOf(toClone.biases, toClone.biases.length);
    }
    
    /**
     * Represents a prior of the coefficients that can be applied to perform 
     * regularization. 
     */
    public enum Prior 
    {
        /**
         * A Gaussian prior, this is equivalent to L<sub>2</sub> regularization. 
         */
        GAUSSIAN
        {
            @Override
            protected double gradientError(double b_i, double s_i)
            {
                return - b_i/s_i;
            }
            
            @Override
            protected double logProb(double b_i, double s_i)
            {
                return -0.5*log(2*PI*s_i)-2*b_i*b_i*s_i/2;
            }
            
        },
        /**
         * A Laplace prior, this is equivalent to L<sub>1</sub> regularization
         */
        LAPLACE
        {
            @Override
            protected double gradientError(double b_i, double s_i)
            {
                return - sqrt(2)*signum(b_i)/sqrt(s_i);
            }
            
            @Override
            protected double logProb(double b_i, double s_i)
            {
                return -signum(b_i)*sqrt(2)*b_i/sqrt(s_i)-0.5*log(2*s_i);
            }
        },
        /**
         * This is the Elastic Net prior, and it uses the extra 
         * {@link #setAlpha(double) alpha} parameter. This prior is a mix of 
         * both {@link #LAPLACE} and {@link #GAUSSIAN}. Alpha should be in the 
         * range [0,1]. Alpha weight will be applied to the Laplace prior, and
         * (1-alpha) weight will be applied to the Gaussian prior. The extreme 
         * values of this collapse into the Laplace and Gaussian priors. 
         */
        ELASTIC
        {
            
            @Override
            protected double gradientError(double b_i, double s_i)
            {
                throw new UnsupportedOperationException();
            }
            
            @Override
            protected double gradientError(double b_i, double s_i, double alpha)
            {
                return alpha*LAPLACE.gradientError(b_i, s_i) 
                        + (1-alpha)*GAUSSIAN.gradientError(b_i, s_i);
            }
            
            @Override
            protected double logProb(double b_i, double s_i)
            {
                return Double.NaN;
            }
            
            @Override
            protected double logProb(double b_i, double s_i, double alpha)
            {
                return alpha*LAPLACE.logProb(b_i, s_i)
                        + (1-alpha)*GAUSSIAN.logProb(b_i, s_i);
            }
        },
        /**
         * This is a prior from the Cauchy (student-t) distribution, and it uses
         * the extra {@link #setAlpha(double) alpha} parameter. Alpha should be 
         * in the range (0, Infty).  
         */
        CAUCHY
        {
            
            @Override
            protected double gradientError(double b_i, double s_i)
            {
                throw new UnsupportedOperationException();
            }
            
            @Override
            protected double gradientError(double b_i, double s_i, double alpha)
            {
                return - 2*b_i/(b_i*b_i+alpha*alpha);
            }
            
            @Override
            protected double logProb(double b_i, double s_i)
            {
                return Double.NaN;
            }
            
            @Override
            protected double logProb(double b_i, double s_i, double alpha)
            {
                return -log(PI)+log(alpha)-log(b_i*b_i+alpha*alpha);
            }
            
        },
        /**
         * This is the Uniform prior. The uniform prior is equivalent to 
         * no regularization. 
         */
        UNIFORM
        {
            @Override
            protected double gradientError(double b_i, double s_i)
            {
                return 0;
            }
            
            @Override
            protected double logProb(double b_i, double s_i)
            {
                return 0;
            }
        };
        
        abstract protected double gradientError(double b_i, double s_i);
        
        protected double gradientError(double b_i, double s_i, double alpha)
        {
            return gradientError(b_i, s_i);
        }
        
        abstract protected double logProb(double b_i, double s_i);
        
        protected double logProb(double b_i, double s_i, double alpha)
        {
            return logProb(b_i, s_i);
        }
    }

    /**
     * Sets whether or not to learn the bias term for a model. If no bias term
     * is in use, the model learned must pass through the origin of the world. 
     * The use of the bias term is very important for low dimensional problems, 
     * but less so for many higher dimensional problems. 
     * @param useBias {@code true} if the bias term should be used, 
     * {@code false} otherwise
     */
    public void setUseBias(boolean useBias)
    {
        this.useBias = useBias;
    }

    /**
     * Returns {@code true} if the bias term is in use
     * @return {@code true} if the bias term is in use
     */
    public boolean isUseBias()
    {
        return useBias;
    }

    /**
     * Sets the maximum number of epochs that occur in each iteration. Each 
     * epoch goes through the whole data set once. 
     * 
     * @param epochs the maximum number of epochs to train
     */
    public void setEpochs(int epochs)
    {
        if(epochs <= 0)
            throw new IllegalArgumentException("Number of epochs must be positive");
        this.epochs = epochs;
    }

    /**
     * Returns the maximum number of epochs
     * @return the maximum number of epochs
     */
    public int getEpochs()
    {
        return epochs;
    }

    /**
     * Sets the extra parameter alpha. This is used for some priors that take 
     * an extra parameter. This is {@link Prior#CAUCHY} and 
     * {@link Prior#ELASTIC}. If these two priors are not in use, the value is
     * ignored. 
     * 
     * @param alpha the extra parameter value to use. Must be positive
     */
    public void setAlpha(double alpha)
    {
        if(alpha < 0 || Double.isNaN(alpha) || Double.isInfinite(alpha))
            throw new IllegalArgumentException("Extra parameter must be non negative, not " + alpha);
        this.alpha = alpha;
    }

    /**
     * Returns the extra parameter value
     * @return the extra parameter value
     */
    public double getAlpha()
    {
        return alpha;
    }

    /**
     * Sets whether or not the clip changes in coefficient values caused by 
     * regularization so that they can not make the coefficients go from 
     * positive to negative or negative to positive. If clipping is on, the 
     * value will go to zero instead. If off, the value will be allowed to 
     * change signs. <br>
     * If there is no regularization, this has no impact. 
     * 
     * @param clipping {@code true} if clipping should be used, false otherwise
     */
    public void setClipping(boolean clipping)
    {
        this.clipping = clipping;
    }

    /**
     * Returns whether or not coefficient clipping is on. 
     * @return {@code true} if clipping is on. 
     */
    public boolean isClipping()
    {
        return clipping;
    }

    /**
     * Sets the initial learning rate to use for the first epoch. The learning 
     * rate will decay according to the 
     * {@link #setLearningRateDecay(jsat.math.decayrates.DecayRate) decay rate}
     * in use. 
     * 
     * @param initialLearningRate the initial learning rate to use
     */
    public void setInitialLearningRate(double initialLearningRate)
    {
        if(initialLearningRate <= 0 || Double.isInfinite(initialLearningRate) || Double.isNaN(initialLearningRate))
            throw new IllegalArgumentException("Learning rate must be a positive constant, not " + initialLearningRate);
        this.initialLearningRate = initialLearningRate;
    }

    /**
     * Returns the current initial learning rate
     * @return the learning rate in use
     */
    public double getInitialLearningRate()
    {
        return initialLearningRate;
    }

    /**
     * Sets the decay rate used to reduce the learning rate after each epoch. 
     * 
     * @param learningRateDecay the decay rate to use
     */
    public void setLearningRateDecay(DecayRate learningRateDecay)
    {
        this.learningRateDecay = learningRateDecay;
    }

    /**
     * Returns the decay rate in use  
     * @return the decay rate in use
     */
    public DecayRate getLearningRateDecay()
    {
        return learningRateDecay;
    }

    /**
     * Sets the coefficient applied to the regularization penalty at each 
     * update. This is usual set to a small value less than 1. If set to zero, 
     * it effectively turns off the use of regularization. 
     * 
     * @param regularization the non negative regularization coefficient to apply
     */
    public void setRegularization(double regularization)
    {
        if(regularization < 0 || Double.isNaN(regularization) || Double.isInfinite(regularization))
            throw new IllegalArgumentException("Regualrization must be a non negative constant, not " + regularization);
        this.regularization = regularization;
    }

    /**
     * Returns the regularization coefficient in use
     * @return the regularization coefficient in use
     */
    public double getRegularization()
    {
        return regularization;
    }

    /**
     * Sets the prior used to perform regularization 
     * @param prior the prior to use
     */
    public void setPrior(Prior prior)
    {
        this.prior = prior;
    }

    /**
     * Returns the prior used for regularization
     * @return the prior used
     */
    public Prior getPrior()
    {
        return prior;
    }

    /**
     * Sets the tolerance that determines when the training stops early because 
     * the change has become too insignificant. 
     * 
     * @param tolerance the minimum change in log likelihood to stop training
     */
    public void setTolerance(double tolerance)
    {
        this.tolerance = tolerance;
    }

    /**
     * Returns the minimum tolerance for early stopping. 
     * @return the minimum change in log likelihood to stop training
     */
    public double getTolerance()
    {
        return tolerance;
    }

    /**
     * Sets whether or not to perform implicit standardization of the feature 
     * values when performing regularization by the prior. If set on, the input 
     * data will be adjusted to have zero mean and unit variance. This is done 
     * without destroying sparsity. If there is not regularization, this 
     * parameter has no impact. 
     * 
     * @param standardized {@code true} if the input will be standardized, 
     * {@code false} if ti will be left as is. 
     */
    public void setStandardized(boolean standardized)
    {
        this.standardized = standardized;
    }

    /**
     * Returns whether or not the input is standardized for the priors
     * @return {@code true} if the input is standardized for the priors
     */
    public boolean isStandardized()
    {
        return standardized;
    }

    /**
     * Sets the amount of data points used to form each gradient update. 
     * Increasing the batch size can help convergence. By default, a mini batch 
     * size of 1 is used. 
     * 
     * @param miniBatchSize the number of data points used to perform each 
     * update
     */
    public void setMiniBatchSize(int miniBatchSize)
    {
        this.miniBatchSize = miniBatchSize;
    }

    /**
     * Returns the number of data points used to perform each gradient update
     * @return the number of data points used to perform each gradient update
     */
    public int getMiniBatchSize()
    {
        return miniBatchSize;
    }

    @Override
    public Vec getRawWeight(int index)
    {
        if(index == B.length)
            return new ConstantVector(0, B[0].length());
        else
            return B[index];
    }

    @Override
    public double getBias(int index)
    {
        if(index == biases.length)
            return 0;
        else
            return biases[index];
    }
    
    @Override
    public int numWeightsVecs()
    {
        return B.length+1;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(B == null)
            throw new UntrainedModelException("Model has not yet been trained");
        final Vec x = data.getNumericalValues();

        double[] probs = new double[B.length + 1];

        for (int i = 0; i < B.length; i++)
            probs[i] = x.dot(B[i])+biases[i];
        probs[B.length] = 1;
        MathTricks.softmax(probs, false);

        return new CategoricalResults(probs);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        final int n = dataSet.getSampleSize();
        final double N = n;
        final int d = dataSet.getNumNumericalVars();
        if(d < 1)
            throw new FailedToFitException("Data set has no numeric attributes to train on");
        B = new Vec[dataSet.getClassSize()-1];
        biases = new double[B.length];
        for(int i = 0; i < B.length; i++)
            B[i] = new DenseVector(d);
        
        IntList randOrder = new IntList(n);
        ListUtils.addRange(randOrder, 0, n, 1);
        
        Vec means = null, stdDevs = null;
        
        if(standardized)
        {
            Vec[] ms = dataSet.getColumnMeanVariance();
            means = ms[0];
            stdDevs = ms[1];
            stdDevs.applyFunction(MathTricks.sqrtFunc);

            //Now transform it so that stdDevs holds standard deviations, and means is the mean / standDev
            means.pairwiseDivide(stdDevs);

            stdDevs.applyFunction(MathTricks.invsFunc);
        }
        
        
        double[] zs = new double[B.length];
        
        /**
         * Contains the last time each feature was used
         */
        int[] u = new int[d];
        /**
         * Contains the current time. 
         */
        int q = 0;
        
        double prevLogLike = Double.POSITIVE_INFINITY;
        //learing rate in use
        double eta;
        
        for(int iter = 0; iter < epochs; iter++)
        {
            Collections.shuffle(randOrder);
            double logLike = 0;
            eta = learningRateDecay.rate(iter, epochs, initialLearningRate);
            final double etaReg = regularization*eta;
            
            for (int batch = 0; batch < randOrder.size(); batch += miniBatchSize)
            {
                int batchCount = Math.min(miniBatchSize, randOrder.size() - batch);
                double batchFrac = 1.0 / batchCount;
                for (int k = 0; k < batchCount; k++)
                {
                    int j = randOrder.get(batch+k);
                    final int c_j = dataSet.getDataPointCategory(j);
                    final Vec x_j = dataSet.getDataPoint(j).getNumericalValues();

                    //compute softmax
                    for (int i = 0; i < B.length; i++)
                        zs[i] = x_j.dot(B[i]) + biases[i];

                    MathTricks.softmax(zs, true);


                    //lazy apply lost rounds of regularization
                    if (prior != Prior.UNIFORM)
                    {
                        for (IndexValue iv : x_j)
                        {
                            int i = iv.getIndex();
                            if(u[i] == 0)
                                continue;
                            double etaRegScaled = etaReg * (u[i] - q) / N;
                            for (Vec b : B)
                            {
                                double bVal = b.get(i);
                                double bNewVal = bVal;
                                if (standardized)
                                    bNewVal += etaRegScaled * prior.gradientError(bVal * stdDevs.get(i) - means.get(i), 1, alpha);
                                else
                                    bNewVal += etaRegScaled * prior.gradientError(bVal, 1, alpha);

                                if (clipping && signum(bVal) != signum(bNewVal))
                                    b.set(i, 0);
                                else
                                    b.set(i, bNewVal);
                            }
                            u[i] = q;
                        }

                        //No need to do bias here, b/c bias is always up to date
                    }

                    for (int c = 0; c < B.length; c++)
                    {
                        Vec b = B[c];
                        double p_c = zs[c];
                        double log_pc = log(p_c);
                        if (!Double.isInfinite(log_pc))
                            logLike += log_pc;
                        double errScaling = (c == c_j ? 1 : 0) - p_c;
                        b.mutableAdd(batchFrac*eta * errScaling, x_j);
                        if (useBias)
                            biases[c] += batchFrac*eta * errScaling + etaReg * prior.gradientError(biases[c] - 1, 1, alpha);
                    }
                }
                
                q++;
            }

            logLike *= -1;
            if (prior != Prior.UNIFORM)
            {
                for (int i = 0; i < d; i++)
                {
                    if (u[i] - q == 0)
                    {
                        for (Vec b : B)
                            if (standardized)
                                logLike += regularization*prior.logProb(b.get(i) * stdDevs.get(i) - means.get(i), 1, alpha);
                            else
                                logLike += regularization*prior.logProb(b.get(i), 1, alpha);
                        continue;
                    }
                    double etaRegScaled = etaReg * (u[i] - q) / N;
                    for (Vec b : B)
                    {
                        double bVal = b.get(i);
                        if (bVal == 0.0)
                            continue;
                        double bNewVal = bVal;
                        if (standardized)
                            bNewVal += etaRegScaled * prior.gradientError(bVal * stdDevs.get(i) - means.get(i), 1, alpha);
                        else
                            bNewVal += etaRegScaled * prior.gradientError(bVal, 1, alpha);
                        
                        if (clipping && signum(bVal) != signum(bNewVal))
                            b.set(i, 0);
                        else
                            b.set(i, bNewVal);

                        if(standardized)
                            logLike += regularization*prior.logProb(b.get(i) * stdDevs.get(i) - means.get(i), 1, alpha);
                        else
                            logLike += regularization*prior.logProb(b.get(i), 1, alpha);
                    }
                    u[i] = q;
                }
            }
            
            double dif = abs(prevLogLike-logLike)/(abs(prevLogLike)+abs(logLike));
            if(dif < tolerance)
                break;
            else
                prevLogLike = logLike;
            
        }
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }
    
    /**
     * Returns the raw coefficient vector used without the bias term. For a 
     * multinomial Logistic model, there are C-1 coefficient vectors. C is the
     * number of output classes. Altering the returned vector will alter the 
     * model. The i'th index of the vector corresponds to the weight therm for
     * the i'th index in an input. 
     * 
     * @param id which coefficient vector to obtain
     * @return the vector of variable coefficients. 
     */
    public Vec getCoefficientVector(int id)
    {
        return B[id];
    }

    @Override
    public Classifier clone()
    {
        return new StochasticMultinomialLogisticRegression(this);
    }
    
    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
}
