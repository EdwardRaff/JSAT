
package jsat.classifiers.neuralnetwork;

import static java.lang.Math.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.distributions.Normal;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionBase;
import jsat.math.decayrates.DecayRate;
import jsat.math.decayrates.ExponetialDecay;
import jsat.parameters.IntParameter;
import jsat.parameters.ObjectParameter;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.random.RandomUtil;

/**
 * An implementation of a Feed Forward Neural Network (NN) trained by Back
 * Propagation. NNs are powerful classifiers and regressors, but can suffer from
 * slow training time and overfitting. <br>
 * <br>
 * NOTE: This class should generally not be used any more. The
 * {@link DReDNetSimple} provides an easier to use class for most cases that
 * will likely converge faster.
 *
 * @author Edward Raff
 */
public class BackPropagationNet implements Classifier, Regressor, Parameterized
{   

    private static final long serialVersionUID = 335438198218313862L;
    private int inputSize, outputSize;
    private ActivationFunction f = softsignActiv;
    private DecayRate learningRateDecay = new ExponetialDecay();
    private double momentum = 0.1;
    private double weightDecay = 0;
    private int epochs = 1000;
    private double initialLearningRate = 0.2;
    private WeightInitialization weightInitialization = WeightInitialization.TANH_NORMALIZED_INITIALIZATION;
    private double targetBump = 0.1;
    private int batchSize = 10;
    /**
     * Length of the array determines how many layers of hidden units. Value at 
     * each index determines how many neurons are in each hidden layer. 
     */
    private int[] npl;
    /**
     * Matrix of weights for each hidden layer and output layer
     */
    private List<Matrix> Ws;
    /**
     * Bias terms corresponding to each layer
     */
    private List<Vec> bs;
    
    /**
     * Target min and max and scaling multiplier for regression problems to make
     * the target into a range that the activation function can reach
     */
    private double targetMax, targetMin, targetMultiplier;
    
    /**
     * Creates a new back propagation network with one hidden layer of 1024 neurons. 
     * @param npl the array of hidden layer information. The length indicates 
     * how many hidden layers, and the value of each index indicates how many 
     * neurons to place in each hidden layer
     */
    public BackPropagationNet()
    {
        this(1024);
    }

    /**
     * Creates a new back propagation network. 
     * @param npl the array of hidden layer information. The length indicates 
     * how many hidden layers, and the value of each index indicates how many 
     * neurons to place in each hidden layer
     */
    public BackPropagationNet(final int... npl )
    {
        if(npl.length < 1)
            throw new IllegalArgumentException("There must be at least one hidden layer");
        this.npl = npl;
    }

    /**
     * Copy constructor
     * @param toClone the one to copy
     */
    protected BackPropagationNet(BackPropagationNet toClone)
    {
        this(Arrays.copyOf(toClone.npl, toClone.npl.length));
        this.inputSize = toClone.inputSize;
        this.outputSize = toClone.outputSize;
        this.f = toClone.f;
        this.momentum = toClone.momentum;
        this.weightDecay = toClone.weightDecay;
        this.epochs = toClone.epochs;
        this.initialLearningRate = toClone.initialLearningRate;
        this.learningRateDecay = toClone.learningRateDecay;
        this.weightInitialization = toClone.weightInitialization;
        this.targetBump = toClone.targetBump;
        this.targetMax = toClone.targetMax;
        this.targetMin = toClone.targetMin;
        this.targetMultiplier = toClone.targetMultiplier;
        this.batchSize = toClone.batchSize;
        
        if(toClone.Ws != null)
        {
            this.Ws = new ArrayList<Matrix>(toClone.Ws);
            for(int i = 0; i < this.Ws.size(); i++)
                this.Ws.set(i, this.Ws.get(i).clone());
        }
        
        if(toClone.bs != null)
        {
            this.bs = new ArrayList<Vec>(toClone.bs);
            for(int i = 0; i < this.bs.size(); i++)
                this.bs.set(i, this.bs.get(i).clone());
        }
    }

    /**
     * The main work for training the neural network
     * @param dataSet the data set to train from
     */
    private void trainNN(DataSet dataSet)
    {
        //batchSize
        
        List<List<Vec>> activations = new ArrayList<List<Vec>>(batchSize);
        List<List<Vec>> derivatives = new ArrayList<List<Vec>>(batchSize);
        List<List<Vec>> deltas = new ArrayList<List<Vec>>(batchSize);
        
        List<Matrix> updates = new ArrayList<Matrix>(Ws.size());
        
        List<Vec> cur_x = new ArrayList<Vec>(batchSize);
        List<Vec> prev_x = new ArrayList<Vec>(batchSize);
        
        for(int i = 0; i < batchSize; i++)
        {
            activations.add(new ArrayList<Vec>(Ws.size()));
            derivatives.add(new ArrayList<Vec>(Ws.size()));
            deltas.add(new ArrayList<Vec>(Ws.size()));
            
            for(Matrix w : Ws)
            {
                int L = w.rows();
                activations.get(i).add(new DenseVector(L));
                derivatives.get(i).add(new DenseVector(L));
                deltas.get(i).add(new DenseVector(L));
                if(i == 0)
                    updates.add(new DenseMatrix(w.rows(), w.cols()));
            }
        }
        
        IntList iterOrder = new IntList(dataSet.getSampleSize());
        ListUtils.addRange(iterOrder, 0, dataSet.getSampleSize(), 1);
        
        final double bSizeInv = 1.0/batchSize;
        
        for(int epoch = 0; epoch < epochs; epoch++)
        {
            Collections.shuffle(iterOrder);
            final double eta = learningRateDecay.rate(epoch, epochs, initialLearningRate);//learningRate;
            double error = 0.0;
            for(int iter = 0; iter < dataSet.getSampleSize(); iter+=batchSize)
            {
                if(dataSet.getSampleSize() - iter < batchSize)
                    continue;//we have run out of enough sampels to do an update
                
                cur_x.clear();
                
                //Feed batches thought network and get final mistakes
                for(int bi = 0; bi < batchSize; bi++)
                {
                    final int idx = iterOrder.get(iter+bi);
                    Vec x = dataSet.getDataPoint(idx).getNumericalValues();
                    cur_x.add(x);
                    feedForward(x, activations.get(bi), derivatives.get(bi));

                    
                    //Compution of Deltas
                    Vec delta_out = deltas.get(bi).get(npl.length);

                    Vec a_i = activations.get(bi).get(npl.length);
                    Vec d_i = derivatives.get(bi).get(npl.length);
                    
                    error += computeOutputDelta(dataSet, idx, delta_out, a_i, d_i);
                }
                
                //Propigate the collected errors back
                for(int bi = 0; bi < batchSize; bi++)
                {
                    for(int i = Ws.size()-2; i >= 0; i--)
                    {
                        Vec delta = deltas.get(bi).get(i);
                        delta.zeroOut();
                        Matrix W = Ws.get(i+1);
                        W.transposeMultiply(1, deltas.get(bi).get(i+1), delta);
                        delta.mutablePairwiseMultiply(derivatives.get(bi).get(i));
                    }

                    //Apply weight changes
                    for(int i = 1; i < Ws.size(); i++)
                    {
                        Matrix W = Ws.get(i);
                        Vec b = bs.get(i);
                        W.mutableSubtract(eta*weightDecay, W);
                        
                        if(momentum != 0)
                        {
                            Matrix update = updates.get(i);
                            update.mutableMultiply(momentum);
                            Matrix.OuterProductUpdate(update, deltas.get(bi).get(i), activations.get(bi).get(i-1), -eta*bSizeInv);
                            W.mutableAdd(update);
                        }
                        else//update directly
                        {
                            Matrix.OuterProductUpdate(W, deltas.get(bi).get(i), activations.get(bi).get(i-1), -eta*bSizeInv);
                        }
                        
                        b.mutableAdd(-eta*bSizeInv, deltas.get(bi).get(i));
                    }

                    //input layer
                    Matrix W = Ws.get(0);
                    W.mutableSubtract(eta*weightDecay, W);
                    Vec b = bs.get(0);
                    
                    if(momentum != 0)
                    {
                        Matrix update = updates.get(0);
                        update.mutableMultiply(momentum);

                        Matrix.OuterProductUpdate(update, deltas.get(bi).get(0), cur_x.get(bi), -eta*bSizeInv);
                        W.mutableAdd(update);
                    }
                    else//update directly
                    {
                        Matrix.OuterProductUpdate(W, deltas.get(bi).get(0), cur_x.get(bi), -eta*bSizeInv);
                    }
                        
                    b.mutableAdd(-eta*bSizeInv, deltas.get(bi).get(0));
                }
                
            }
        }
    }

    /**
     * Different methods of initializing the weight values before training
     */
    public enum WeightInitialization 
    {
        UNIFORM
        {
            @Override
            public double getWeight(int inputSize, int layerSize, double eta, Random rand)
            {
                return rand.nextDouble()*1.4-0.7;
            }
        },
        GUASSIAN
        {
            @Override
            public double getWeight(int inputSize, int layerSize, double eta, Random rand)
            {
                return Normal.invcdf(rand.nextDouble(), 0, pow(inputSize, -0.5));
            }
        },
        TANH_NORMALIZED_INITIALIZATION
        {
            @Override
            public double getWeight(int inputSize, int layerSize, double eta, Random rand)
            {
                double cnst = sqrt(6.0/(inputSize+layerSize));
                return rand.nextDouble()*cnst*2-cnst;
            }
        };
        
        /**
         * 
         * @param inputSize also referred to as the fan<sub>in</sub>
         * @param layerSize also referred to as the fan<sub>out</sub>
         * @param eta the initial learning rate
         * @param rand the source of randomness
         * @return one weight value
         */
        abstract public double getWeight(int inputSize, int layerSize, double eta, Random rand);
    }

    /**
     * Sets the non negative momentum used in training. 
     * @param momentum the momentum to apply to training
     */
    public void setMomentum(double momentum)
    {
        if(momentum < 0 || Double.isNaN(momentum) || Double.isInfinite(momentum))
            throw new ArithmeticException("Momentum must be non negative, not " + momentum);
        this.momentum = momentum;
    }

    /**
     * Returns the momentum in use
     * @return the momentum
     */
    public double getMomentum()
    {
        return momentum;
    }

    /**
     * Sets the initial learning rate used for the first epoch
     * @param initialLearningRate the positive learning rate to use 
     */
    public void setInitialLearningRate(double initialLearningRate)
    {
        if(initialLearningRate <= 0 || Double.isNaN(initialLearningRate) || Double.isInfinite(initialLearningRate))
            throw new ArithmeticException("Learning rate must be a positive cosntant, not " + initialLearningRate );
        this.initialLearningRate = initialLearningRate;
    }

    /**
     * Returns the learning rate used
     * @return the learning rate used
     */
    public double getInitialLearningRate()
    {
        return initialLearningRate;
    }

    /**
     * Sets the decay rate used to adjust the learning rate after each epoch
     * @param learningRateDecay the decay for the learning rate
     */
    public void setLearningRateDecay(DecayRate learningRateDecay)
    {
        this.learningRateDecay = learningRateDecay;
    }

    /**
     * Returns the decay rate used to adjust the learning rate after each epoch
     * @return the decay rate used for learning
     */
    public DecayRate getLearningRateDecay()
    {
        return learningRateDecay;
    }

    /**
     * Sets the number of epochs of training used. Each epoch goes through the
     * whole data set once. 
     * @param epochs the number of training epochs
     */
    public void setEpochs(int epochs)
    {
        if(epochs < 1)
            throw new ArithmeticException("number of training epochs must be positive, not " + epochs);
        this.epochs = epochs;
    }

    /**
     * Returns the number of epochs of training epochs for learning
     * @return the number of training epochs
     */
    public int getEpochs()
    {
        return epochs;
    }

    /**
     * Sets the weight decay used for each update. The weight decay must be in 
     * the range [0, 1). Weight decay values must often be very small, often 
     * 1e-8 or less. 
     * 
     * @param weightDecay the weight decay to apply when training
     */
    public void setWeightDecay(double weightDecay)
    {
        if(weightDecay < 0 || weightDecay >= 1 || Double.isNaN(weightDecay))
            throw new ArithmeticException("Weight decay must be in [0,1), not " + weightDecay);
        this.weightDecay = weightDecay;
    }

    /**
     * Returns the weight decay used for each update
     * @return the weight decay used. 
     */
    public double getWeightDecay()
    {
        return weightDecay;
    }

    /**
     * Sets how the weights are initialized before training starts
     * @param weightInitialization the method of weight initialization
     */
    public void setWeightInitialization(WeightInitialization weightInitialization)
    {
        this.weightInitialization = weightInitialization;
    }

    /**
     * Returns the method of weight initialization used
     * @return the method of weight initialization used
     */
    public WeightInitialization getWeightInitialization()
    {
        return weightInitialization;
    }

    /**
     * Sets the batch size use to estimate the gradient of the error for 
     * training
     * @param batchSize the number of training instances to use on each update
     */
    public void setBatchSize(int batchSize)
    {
        this.batchSize = batchSize;
    }

    /**
     * Returns the training batch size
     * @return the batch size used for training
     */
    public int getBatchSize()
    {
        return batchSize;
    }

    /**
     * Sets the activation function used for the network 
     * @param f the activation function to use
     */
    public void setActivationFunction(ActivationFunction f)
    {
        this.f = f;
    }

    /**
     * Returns the activation function used for training the network
     * @return the activation function in use
     */
    public ActivationFunction getActivationFunction()
    {
        return f;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(outputSize);
        Vec x = feedForward(data.getNumericalValues());
        
        x.mutableSubtract(f.min()+targetBump);
        
        
        for(int i = 0; i < x.length(); i++)
            cr.setProb(i, Math.max(x.get(i), 0));
        cr.normalize();
        
        return cr;
    }

    @Override
    public double regress(DataPoint data)
    {
        Vec x = feedForward(data.getNumericalValues());
        
        double val = x.get(0);
        
        val = (val - f.min()-targetBump)/targetMultiplier+targetMin;
        
        return val;
    }
    
    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        inputSize = dataSet.getNumNumericalVars();
        outputSize = dataSet.getClassSize();
        
        Random rand = RandomUtil.getRandom();
        
        setUp(rand);
        
        trainNN(dataSet);
        
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        train(dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        targetMax = Double.NEGATIVE_INFINITY;
        targetMin = Double.POSITIVE_INFINITY;
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            double val = dataSet.getTargetValue(i);
            targetMax = Math.max(targetMax, val);
            targetMin = Math.min(targetMin, val);
        }
        
        targetMultiplier = ((f.max()-targetBump)-(f.min()+targetBump))/(targetMax-targetMin);
        
        inputSize = dataSet.getNumNumericalVars();
        outputSize = 1;
        
        Random rand = RandomUtil.getRandom();
        
        setUp(rand);
        
        trainNN(dataSet);
    }
    
    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public BackPropagationNet clone()
    {
        return new BackPropagationNet(this);
    }
    
    /**
     * The neural network needs an activation function for the neurons that is 
     * used to predict from inputs and train the network by propagating the 
     * errors back through the network. 
     */
    public static abstract class  ActivationFunction implements Function
    {
        private static final long serialVersionUID = 8002040194215453918L;

        /**
         * Computes the response of the response of this activation function on 
         * the given input value
         * @param x the input value
         * @return the response value
         */
        abstract public double response(double x);
        
        /**
         * The minimum possible response value
         * @return the min value
         */
        abstract public double min();
        
        /**
         * The maximum possible response value
         * @return the max value
         */
        abstract public double max();
        
        /**
         * Returns the function object for the derivative of this activation 
         * function. The derivative must be calculated using only the output of 
         * the response when given the original input. Meaning: given an input 
         * {@code x}, the value of f'(x) must be computable as g(f(x))
         * 
         * @return the function for computing the derivative of the response
         */
        abstract public Function getD();
        

        @Override
        public double f(double... x)
        {
            return response(x[0]);
        }

        @Override
        public double f(Vec x)
        {
            return response(x.get(0));
        }

    }
    
    /**
     * The logit activation function. This function goes from [0, 1]. It has 
     * more difficultly learning than symmetric activation functions, often 
     * requiring considerably more layers and neurons than other activation 
     * functions. 
     */
    public static final ActivationFunction logitActiv = new ActivationFunction() 
    {

        private static final long serialVersionUID = -5675881412853268432L;

        @Override
        public double response(double x)
        {
            return 1 / (1+exp(-x));
        }

        @Override
        public double min()
        {
            return 0;
        }

        @Override
        public double max()
        {
            return 1;
        }

        @Override
        public Function getD()
        {
            return logitPrime;
        }

        @Override
        public String toString()
        {
            return "Logit";
        }
    };
    
    private static final Function logitPrime = new FunctionBase()
    {
        private static final long serialVersionUID = 7201403465671204173L;

        @Override
        public double f(Vec x)
        {
            double xx = x.get(0);
            return xx * (1 - xx);
        }
    };

    /**
     * The tanh activation function. This function is symmetric in the range of
     * [-1, 1]. It works well for many problems in general. 
     */
    public static final ActivationFunction tanhActiv = new ActivationFunction() 
    {
        private static final long serialVersionUID = 5531922338473526216L;

        @Override
        public double response(double x)
        {
            return tanh(x);
        }

        @Override
        public double min()
        {
            return -1;
        }

        @Override
        public double max()
        {
            return 1;
        }
        
        @Override
        public Function getD()
        {
            return tanhPrime;
        }

        @Override
        public String toString()
        {
            return "Tanh";
        }
    };
    
    private static final Function tanhPrime = new FunctionBase() 
    {
        private static final long serialVersionUID = -7271551720122166947L;

        @Override
        public double f(Vec x)
        {
            double xx = x.get(0);
            return 1-xx*xx;
        }
    };
    
    /**
     * The softsign activation function. This function is symmetric in the range
     * of [-1, 1]. It works well for classification problems, and is very fast 
     * to compute. It sometimes requires more neurons to learn more complicated
     * functions / boundaries. It sometimes has reduced performance in regression
     */
    public static final ActivationFunction softsignActiv = new ActivationFunction() 
    {

        private static final long serialVersionUID = 1618447580574194519L;

        @Override
        public double response(double x)
        {
            return x/(1.0 + abs(x));
        }

        @Override
        public double min()
        {
            return -1;
        }

        @Override
        public double max()
        {
            return 1;
        }

        @Override
        public Function getD()
        {
            return softsignPrime;
        }

        @Override
        public String toString()
        {
            return "Softsign";
        }
    };
    
    private static final Function softsignPrime = new FunctionBase() 
    {
        private static final long serialVersionUID = -6726314880590071199L;

        @Override
        public double f(Vec x)
        {
            double xx = 1-abs(x.get(0));
                    return xx*xx;
        }
    };
    
    /**
     * Creates the weights for the hidden layers and output layer
     * @param rand source of randomness
     */
    private void setUp(Random rand)
    {
        Ws = new ArrayList<Matrix>(npl.length);
        bs = new ArrayList<Vec>(npl.length);
        
        //First Hiden layer takes input raw
        DenseMatrix W = new DenseMatrix(npl[0], inputSize);
        Vec b = new DenseVector(W.rows());
        initializeWeights(W, rand);
        initializeWeights(b, W.cols(), rand);
        Ws.add(W);
        bs.add(b);
        
        //Other Hiden Layers Layers 
        for(int i = 1; i < npl.length; i++)
        {
            W = new DenseMatrix(npl[i], npl[i-1]);
            b = new DenseVector(W.rows());
            initializeWeights(W, rand);
            initializeWeights(b, W.cols(), rand);
            Ws.add(W);
            bs.add(b);
        }
        
        //Output layer
        W = new DenseMatrix(outputSize, npl[npl.length-1]);
        b = new DenseVector(W.rows());
        initializeWeights(W, rand);
        initializeWeights(b, W.cols(), rand);
        Ws.add(W);
        bs.add(b);
        
    }
    
    /**
     * Computes the delta between the networks output for a same and its true value
     * @param dataSet the data set we are learning from
     * @param idx the index into the data set for the current data point
     * @param delta_out the place to store the delta, may already be initialized with random noise 
     * @param a_i the activation of the final output layer for the data point
     * @param d_i the derivative of the activation of the final output layer
     * @return the error that occurred in predicting this data point
     */
    private double computeOutputDelta(DataSet dataSet, final int idx, Vec delta_out, Vec a_i, Vec d_i)
    {
        double error = 0;
        if (dataSet instanceof ClassificationDataSet)
        {
            ClassificationDataSet cds = (ClassificationDataSet) dataSet;
            final int ct = cds.getDataPointCategory(idx);
            for (int i = 0; i < outputSize; i++)
                if (i == ct)
                    delta_out.set(i, f.max() - targetBump);
                else
                    delta_out.set(i, f.min() + targetBump);


            for (int j = 0; j < delta_out.length(); j++)
            {
                double val = delta_out.get(j);
                error += pow((val - a_i.get(j)), 2);
                val = -(val - a_i.get(j)) * d_i.get(j);
                delta_out.set(j, val);
            }
        }
        else if(dataSet instanceof RegressionDataSet)
        {
            RegressionDataSet rds = (RegressionDataSet) dataSet;
            double val = rds.getTargetValue(idx);
            val = f.min()+targetBump + targetMultiplier*(val-targetMin);
            error += pow((val - a_i.get(0)), 2);
            delta_out.set(0, -(val - a_i.get(0)) * d_i.get(0));
        }
        else
        {
            throw new RuntimeException("BUG: please report");
        }
        
        return error;
    }
    
    /**
     * Feeds a vector through the network to get an output 
     * @param input the input to feed forward though the network
     * @param activations the list of allocated vectors to store the activation 
     * outputs for each layer  
     * @param derivatives the list of allocated vectors to store the derivatives
     * of the activations
     */
    private void feedForward(Vec input, List<Vec> activations, List<Vec> derivatives)
    {
        Vec x = input;
        for(int i = 0; i < Ws.size(); i++)
        {
            Matrix W_i = Ws.get(i);
            Vec b_i = bs.get(i);

            Vec a_i = activations.get(i);
            a_i.zeroOut();
            W_i.multiply(x, 1, a_i);
            a_i.mutableAdd(b_i);
            
            a_i.applyFunction(f);
            
            Vec d_i = derivatives.get(i);
            a_i.copyTo(d_i);
            d_i.applyFunction(f.getD());
            
            x = a_i;
        }
    }
    
    /**
     * Feeds an input through the network
     * @param inputthe input vector to feed in
     * @return the output vector for the given input at the final layer
     */
    private Vec feedForward(Vec input)
    {
        Vec x = input;
        for(int i = 0; i < Ws.size(); i++)
        {
            Matrix W_i = Ws.get(i);
            Vec b_i = bs.get(i);

            Vec a_i = W_i.multiply(x);
            a_i.mutableAdd(b_i);
            
            a_i.applyFunction(f);
            
            x = a_i;
        }
        
        return x;
    }
    
    private void initializeWeights(Matrix W, Random rand)
    {
        for(int i = 0; i < W.rows(); i++)
            for(int j = 0; j < W.cols(); j++)
                W.set(i, j, weightInitialization.getWeight(W.cols(), W.rows(), initialLearningRate, rand));
    }
    
    private void initializeWeights(Vec b, int inputSize, Random rand)
    {
        for(int i = 0; i < b.length(); i++)
            b.set(i, weightInitialization.getWeight(inputSize, b.length(), initialLearningRate, rand));
    }
    
    @Override
    public List<Parameter> getParameters()
    {
        ArrayList<Parameter> params = new ArrayList<Parameter>(Parameter.getParamsFromMethods(this));
        for(int i = 0; i < npl.length; i++)
        {
            final int ii = i;
            if(npl[ii] < 1)
                throw new ArithmeticException("There must be a poistive number of hidden neurons in each layer");
            params.add(new IntParameter() 
            {

                private static final long serialVersionUID = -827784019950722754L;

                @Override
                public int getValue()
                {
                    return npl[ii];
                }

                @Override
                public boolean setValue(int val)
                {
                    if(val <= 0)
                        return false;
                    npl[ii] = val;
                    return true;
                }

                @Override
                public String getASCIIName()
                {
                    return "Neurons for Hidden Layer " + ii;
                }
            });
        }
        
        params.add(new ObjectParameter<ActivationFunction>() 
        {

            private static final long serialVersionUID = 6871130865935243583L;

            @Override
            public ActivationFunction getObject()
            {
                return getActivationFunction();
            }

            @Override
            public boolean setObject(ActivationFunction obj)
            {
                setActivationFunction(obj);
                return true;
            }

            @Override
            public List parameterOptions()
            {
                return Arrays.asList(logitActiv, tanhActiv, softsignActiv);
            }

            @Override
            public String getASCIIName()
            {
                return "Activation Function";
            }

        });
        return Collections.unmodifiableList(params);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
}
