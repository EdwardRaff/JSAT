

package jsat.classifiers.neuralnetwork;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.neuralnetwork.activations.ActivationLayer;
import jsat.classifiers.neuralnetwork.initializers.BiastInitializer;
import jsat.classifiers.neuralnetwork.initializers.WeightInitializer;
import jsat.classifiers.neuralnetwork.regularizers.Max2NormRegularizer;
import jsat.classifiers.neuralnetwork.regularizers.WeightRegularizer;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.math.decayrates.DecayRate;
import jsat.math.decayrates.NoDecay;
import jsat.math.optimization.stochastic.GradientUpdater;
import jsat.math.optimization.stochastic.SimpleSGD;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * This class provides a highly configurable and generalized method of training 
 * a neural network using Stochastic Gradient Decent.<br>
 * <br>
 * Note, the API of this class may change in the future. 
 * 
 * @author Edward Raff
 */
public class SGDNetworkTrainer implements Serializable
{

	private static final long serialVersionUID = 5753653181230693131L;
	/**
     * An array where the length indicates the number of layers and the value of
     * each index indicates the number of neurons in that layer. This includes 
     * both the input and output layers
     */
    private int[] layerSizes;
    /**
     * The base learning rate to use
     */
    private double eta;
    /**
     * The dropout probability for the input layers
     */
    private double p_i;
    /**
     * The integer threshold to used when sampling a value from 
     * {@link Random#nextInt() } to get the correct dropout probability
     */
    private int p_i_intThresh;
    /**
     * The dropout probability for the hidden layers. 
     */
    private double p_o;
    /**
     * The integer threshold to used when sampling a value from 
     * {@link Random#nextInt() } to get the correct dropout probability
     */
    private int p_o_intThresh;
    /**
     * The gradient updater to use for updating weights and biases
     */
    private GradientUpdater updater = new SimpleSGD();
    /**
     * The weight regularization method 
     */
    private WeightRegularizer regularizer = new Max2NormRegularizer(15);
    /**
     * The method to initialize all neuron connection weights from
     */
    private WeightInitializer weightInit;
    /**
     * The method to initialize all neuron bias values from
     */
    private BiastInitializer biasInit;
    /**
     * This list contains the neuron weight connection matrix for each layer 
     * after the input layer
     */
    private List<Matrix> W;
    /**
     * This list contains the gradients to update the weight matrices by
     */
    private List<Matrix> W_deltas;
    /**
     * This list contains the gradient updaters used for each layer, where there 
     * is a list of each matrix and each matrix has a list for each row. 
     */
    private List<List<GradientUpdater>> W_updaters;
    /**
     * This list contains the neuron bias connections for each layer after the
     * input layer
     */
    private List<Vec> B;
    /**
     * This list contains the gradients to update the weight biases by
     */
    private List<Vec> B_deltas;
    /**
     * This list contains the gradient updaters used for each set of bias 
     * connections
     */
    private List<GradientUpdater> B_updaters;
    /**
     * This list contains the activation method for each layer after the input 
     * layer
     */
    private List<ActivationLayer> layersActivation;
    /**
     * The decay rate to apply to the base learning rate
     */
    private DecayRate etaDecay = new NoDecay();
    /**
     * The time step, incremented after every mini batch
     */
    private int time;
    
    /**
     * Matrices for storing the activations of each layer 
     */
    private Matrix[] activations;
    private Matrix[] unactivated;
    private Matrix[] deltas;
    
    /**
     * Creates a new SGD network training that uses dropout
     */
    public SGDNetworkTrainer()
    {
        setDropoutInput(0.2);
        setDropoutHidden(0.5);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public SGDNetworkTrainer(SGDNetworkTrainer toCopy)
    {
        this.layerSizes = Arrays.copyOf(toCopy.layerSizes, toCopy.layerSizes.length);
        this.eta = toCopy.eta;
        this.weightInit = toCopy.weightInit.clone();
        this.biasInit = toCopy.biasInit.clone();
        this.regularizer = toCopy.regularizer.clone();
        this.updater = toCopy.updater.clone();
        this.setDropoutInput(toCopy.getDropoutInput());
        this.setDropoutHidden(toCopy.getDropoutHidden());
        if(toCopy.W != null)
        {
            this.W = new ArrayList<Matrix>();
            for(Matrix w : toCopy.W)
                this.W.add(w.clone());
            this.B = new ArrayList<Vec>();
            for(Vec b : toCopy.B)
                this.B.add(b.clone());
        }
        if(toCopy.W_deltas != null)
        {
            this.W_deltas = new ArrayList<Matrix>();
            for(Matrix w : toCopy.W_deltas)
                this.W_deltas.add(w.clone());
            this.B_deltas = new ArrayList<Vec>();
            for(Vec b : toCopy.B_deltas)
                this.B_deltas.add(b.clone());
        }
        if(toCopy.W_updaters != null)
        {
            this.W_updaters = new ArrayList<List<GradientUpdater>>();
            for(List<GradientUpdater> updaters : toCopy.W_updaters)
            {
                List<GradientUpdater> copyUpdaters = new ArrayList<GradientUpdater>(updaters.size());
                this.W_updaters.add(copyUpdaters);
                for(GradientUpdater item : updaters)
                    copyUpdaters.add(item.clone());
            }
            this.B_updaters = new ArrayList<GradientUpdater>(toCopy.B_updaters);
            for(GradientUpdater item : toCopy.B_updaters)
                    this.B_updaters.add(item.clone());
        }
        this.layersActivation = new ArrayList<ActivationLayer>(toCopy.layersActivation.size());
        for(ActivationLayer activation : toCopy.layersActivation)
            this.layersActivation.add(activation.clone());
    }
    
    /**
     * Sets the probability of dropping a value from the input layer
     * @param p the probability in [0, 1) of dropping a value in the input layer
     */
    public void setDropoutInput(double p)
    {
        if(p < 0 || p >= 1 || Double.isNaN(p))
            throw new IllegalArgumentException("Dropout probability must be in [0,1) not " + p);
        p_i = p;
        p_i_intThresh = (int) (0xffffffffL*p_i+Integer.MIN_VALUE);
    }
    
    /**
     * 
     * @return the dropout probability for the input layer
     */
    public double getDropoutInput()
    {
        return p_i;
    }
    
    /**
     * Sets the probability of dropping a value from the hidden layer
     * @param p the probability in [0, 1) of dropping a value in the hidden
     * layer
     */
    public void setDropoutHidden(double p)
    {
        if(p < 0 || p >= 1 || Double.isNaN(p))
            throw new IllegalArgumentException("Dropout probability must be in [0,1) not " + p);
        p_o = p;
        p_o_intThresh = (int) (0xffffffffL*p_o+Integer.MIN_VALUE);
    }
    
    /**
     * 
     * @return the dropout probability for the hidden layers
     */
    public double getDropoutHidden()
    {
        return p_o;
    }

    /**
     * Sets the decay rate on the global learning rate over time
     * @param etaDecay the decay rate to use
     */
    public void setEtaDecay(DecayRate etaDecay)
    {
        this.etaDecay = etaDecay;
    }

    /**
     * 
     * @return the decay rate in use
     */
    public DecayRate getEtaDecay()
    {
        return etaDecay;
    }

    /**
     * Sets the base global learning rate. 
     * @param eta the learning rate to use
     */
    public void setEta(double eta)
    {
        if(eta <= 0 || Double.isNaN(eta) || Double.isInfinite(eta))
            throw new IllegalArgumentException("eta must be a positive constant, not " + eta);
        this.eta = eta;
    }

    /**
     * 
     * @return the global learning rate used
     */
    public double getEta()
    {
        return eta;
    }

    /**
     * Sets the method of regularizing the connections weights
     * @param regularizer the method of regularizing the network
     */
    public void setRegularizer(WeightRegularizer regularizer)
    {
        this.regularizer = regularizer;
    }

    /**
     * 
     * @return the regularizer for the network
     */
    public WeightRegularizer getRegularizer()
    {
        return regularizer;
    }

    /**
     * Sets the array indicating the total number of layers in the network and 
     * the sizes of each layer. The length of the array is the number of layers 
     * and the value at each index is the size of that layer. 
     * @param layerSizes the array of layer sizes
     */
    public void setLayerSizes(int... layerSizes)
    {
        this.layerSizes = layerSizes;
    }

    /**
     * 
     * @return the array of layer sizes in the network
     */
    public int[] getLayerSizes()
    {
        return layerSizes;
    }

    /**
     * Sets the list of layer activations for all layers other than the input
     * layer. 
     * @param layersActivation the list of hidden and output layer activations
     */
    public void setLayersActivation(List<ActivationLayer> layersActivation)
    {
        this.layersActivation = layersActivation;
    }

    /**
     * Sets the gradient update that will be used when updating the weight 
     * matrices and bias terms. 
     * @param updater the updater to use
     */
    public void setGradientUpdater(GradientUpdater updater)
    {
        this.updater = updater;
    }

    /**
     * 
     * @return the gradient updater used 
     */
    public GradientUpdater getGradientUpdater()
    {
        return updater;
    }

    /**
     * Sets the method used to initialize matrix connection weights
     * @param weightInit the weight initialization method
     */
    public void setWeightInit(WeightInitializer weightInit)
    {
        this.weightInit = weightInit;
    }

    /**
     * 
     * @return the weight initialization method
     */
    public WeightInitializer getWeightInit()
    {
        return weightInit;
    }

    /**
     * Sets the method to use when initializing neuron bias values
     * @param biasInit the bias initialization method
     */
    public void setBiasInit(BiastInitializer biasInit)
    {
        this.biasInit = biasInit;
    }

    /**
     * 
     * @return the bias initialization method
     */
    public BiastInitializer getBiasInit()
    {
        return biasInit;
    }
    
    
    /**
     * Prepares the network by creating all needed structure, initializing 
     * weights, and preparing it for updates
     */
    public void setup()
    {
        assert (layersActivation.size() == layerSizes.length-1);
        
        W = new ArrayList<Matrix>(layersActivation.size());
        B = new ArrayList<Vec>(layersActivation.size());
        
        
        Random rand = RandomUtil.getRandom();
        
        for(int l = 1; l < layerSizes.length; l++)
        {
            W.add(new DenseMatrix(layerSizes[l], layerSizes[l-1]));
            weightInit.init(W.get(W.size()-1), rand);
            
            B.add(new DenseVector(layerSizes[l]));
            biasInit.init(B.get(B.size()-1), layerSizes[l-1], rand);
            
        }
        
        time = 0;
        
        prepareForUpdating();
    }
    
    /**
     * This method assumes that the neural network structure is already in 
     * place, and prepares only the structure needed to perform updates. <br>
     * Any gradient related information that was being used before (such as 
     * momentum when performing updates) will be lost
     */
    private void prepareForUpdating()
    {
        W_deltas = new ArrayList<Matrix>(layersActivation.size());
        W_updaters = new ArrayList<List<GradientUpdater>>(layersActivation.size());
        B_deltas = new ArrayList<Vec>(layersActivation.size());
        B_updaters = new ArrayList<GradientUpdater>(layersActivation.size());
        
        for(int l = 1; l < layerSizes.length; l++)
        {
            W_deltas.add(new DenseMatrix(layerSizes[l], layerSizes[l-1]));
            B_deltas.add(new DenseVector(layerSizes[l]));
            //updaters
            List<GradientUpdater> W_updaters_l = new ArrayList<GradientUpdater>(layerSizes[l]);
            for(int i = 0; i < layerSizes[l]; i++)
            {
                GradientUpdater W_updater = updater.clone();
                W_updater.setup(layerSizes[l-1]);
                W_updaters_l.add(W_updater);
            }
            W_updaters.add(W_updaters_l);
            B_updaters.add(updater.clone());
            B_updaters.get(B_updaters.size()-1).setup(layerSizes[l]);
        }
        
        activations = new Matrix[layersActivation.size()];
        unactivated = new Matrix[layersActivation.size()];
        deltas = new Matrix[layersActivation.size()];
        
    }
    
    /**
     * Calling this method indicates that the user has no intentions of updating
     * the network again and is ready to use it for prediction. This will remove
     * objects not needed for prediction and do cleanup. 
     */
    public void finishUpdating()
    {
        W_deltas = null;
        W_updaters = null;
        B_deltas = null;
        B_updaters = null;
        activations = unactivated = deltas = null;
        W.get(0).mutableMultiply(1.0-p_i);
        B.get(0).mutableMultiply(1.0-p_i);
        for(int i = 1; i < W.size(); i++)
        {
            W.get(i).mutableMultiply(1.0-p_o);
            B.get(i).mutableMultiply(1.0-p_o);
        }
    }
    
    /**
     * Performs a mini-batch update of the network using the given input and 
     * output pairs
     * @param x the list of input values
     * @param y the list of output values
     * @return the error incurred on the given mini batch
     */
    public double updateMiniBatch(List<Vec> x, List<Vec> y)
    {
        return updateMiniBatch(x, y, null);
    }
    
    /**
     * Performs a mini-batch update of the network using the given input and 
     * output pairs
     * @param x the list of input values
     * @param y the list of output values
     * @param ex the source of threads for parallel computation, may be 
     * {@code null}
     * @return the error incurred on the given mini batch
     */
    public double updateMiniBatch(List<Vec> x, List<Vec> y, ExecutorService ex)
    {
        Random rand = RandomUtil.getRandom();
        for(Matrix w : W_deltas)
            w.zeroOut();
        for(Vec b : B_deltas)
            b.zeroOut();
        
        for(int i = 0; i < layersActivation.size(); i++)
        {
            //TODO isntead of making a whole new matrix every time, use a submatrix when bigger and enlarge when too small
            if(activations[i] == null || activations[i].cols() != x.size())
                activations[i] = new DenseMatrix(layerSizes[i+1], x.size());
            if(unactivated[i] == null || unactivated[i].cols() != x.size())
                unactivated[i] = new DenseMatrix(layerSizes[i+1], x.size());
            if(deltas[i] == null || deltas[i].cols() != x.size())
                deltas[i] = new DenseMatrix(layerSizes[i+1], x.size());
        }
        
        Matrix X = new DenseMatrix(layerSizes[0], x.size());
        for (int j = 0; j < x.size(); j++)
            x.get(j).copyTo(X.getColumnView(j));
        
        if(p_i > 0)
            applyDropout(X, p_i_intThresh, rand, ex);
        
        
        double errorMade = 0;
        
        feedforward(X, activations, unactivated, ex, rand);
        
        errorMade = backpropagateError(deltas, activations, x, y, errorMade, ex, unactivated);
        
        accumulateUpdates(X, activations, deltas, ex, x);

        double eta_cur = etaDecay.rate(time++, eta);
        if(ex == null)
            applyGradient(eta_cur);
        else
            applyGradient(eta_cur, ex);
        
        return errorMade;
    }

    private void feedforward(Matrix X, Matrix[] activationsM, Matrix[] unactivatedM, ExecutorService ex, Random rand)
    {
        //feed forward
        for (int l = 0; l < layersActivation.size(); l++)
        {
            final Matrix a_lprev = (l == 0 ? X : activationsM[l - 1]);
            final Matrix a_l = activationsM[l];
            final Matrix z_l = unactivatedM[(l)];
            z_l.zeroOut();
            if(ex == null)
                W.get(l).multiply(a_lprev, z_l);
            else
                W.get(l).multiply(a_lprev, z_l, ex);

            //add the bias term back in
            final Vec B_l = B.get(l);
            if (ex == null)
            {
                for (int i = 0; i < z_l.rows(); i++)
                {
                    final double B_li = B_l.get(i);
                    for (int j = 0; j < z_l.cols(); j++)
                        z_l.increment(i, j, B_li);
                }
            }
            else
            {
                final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
                for (int id = 0; id < SystemInfo.LogicalCores; id++)
                {
                    final int ID = id;
                    ex.submit(new Runnable()
                    {

                        @Override
                        public void run()
                        {
                            for (int i = ID; i < z_l.rows(); i += SystemInfo.LogicalCores)
                            {
                                final double B_li = B_l.get(i);
                                for (int j = 0; j < z_l.cols(); j++)
                                    z_l.increment(i, j, B_li);
                            }
                            latch.countDown();
                        }
                    });

                }

                try
                {
                    latch.await();
                }
                catch (InterruptedException ex1)
                {
                    Logger.getLogger(SGDNetworkTrainer.class.getName()).log(Level.SEVERE, null, ex1);
                }
            }

            if (p_o > 0 && l != layersActivation.size() - 1)
                applyDropout(z_l, p_o_intThresh, rand, ex);
            
            layersActivation.get(l).activate(z_l, a_l, false);
        }
    }

    /**
     * Feeds the given singular pattern through the network and computes its 
     * activations
     * @param x the input vector to feed forward through the network
     * @return the final activation for this network
     */
    public Vec feedfoward(Vec x)
    {
        Vec a_lprev = x;
        for (int l = 0; l < layersActivation.size(); l++)
        {
            Vec z_l = new DenseVector(layerSizes[l+1]);
            z_l.zeroOut();
            W.get(l).multiply(a_lprev, 1.0, z_l);

            //add the bias term back in
            final Vec B_l = B.get(l);
            z_l.mutableAdd(B_l);

            layersActivation.get(l).activate(z_l, z_l);
            a_lprev = z_l;
        }
        
        return a_lprev;
    }
    
    private double backpropagateError(Matrix[] deltasM, Matrix[] activationsM, List<Vec> x, List<Vec> y, double errorMade, ExecutorService ex, Matrix[] unactivatedM)
    {
        //backpropagate the error
        for (int l = layersActivation.size() - 1; l >= 0; l--)
        {
            Matrix delta_l = deltasM[l];

            if (l == layersActivation.size() - 1)//output layer
            {
                activationsM[(l)].copyTo(delta_l);
                for(int r = 0; r < x.size(); r++)
                {
                    delta_l.getColumnView(r).mutableSubtract(y.get(r));
                    errorMade += delta_l.getColumnView(r).pNorm(2);
                }
            }
            else//any other layer
            {
                delta_l.zeroOut();
                if(ex == null)
                    W.get(l+1).transposeMultiply(deltasM[l+1], delta_l);
                else
                    W.get(l+1).transposeMultiply(deltasM[l+1], delta_l, ex);
                
                layersActivation.get(l).backprop(unactivatedM[l], activationsM[l], delta_l, delta_l, false);
            }
        }
        return errorMade;
    }

    private void accumulateUpdates(Matrix X, Matrix[] activationsM, Matrix[] deltasM, ExecutorService ex, final List<Vec> x)
    {
        final double invXsize = 1.0/x.size();
        //accumulate updates
        for (int l = 0; l < layersActivation.size(); l++)
        {
            final Matrix a_lprev = (l == 0 ? X : activationsM[(l - 1)]);
            final Matrix delta_l = deltasM[l];
            if(ex == null)
                delta_l.multiplyTranspose(a_lprev, W_deltas.get(l));
            else
                delta_l.multiplyTranspose(a_lprev, W_deltas.get(l), ex);
            W_deltas.get(l).mutableMultiply(invXsize);
            
            final Vec B_delta_l = B_deltas.get(l);
            if(ex == null)
                for(int i = 0; i < delta_l.rows(); i++)
                {
                    double change = 0;
                    for(int j = 0; j < delta_l.cols(); j++)
                        change += delta_l.get(i, j);
                    B_delta_l.increment(i, change*invXsize);
                }
            else
            {
                final CountDownLatch latch = new CountDownLatch(Math.min(SystemInfo.LogicalCores, delta_l.rows()));
                for(int id = 0; id < SystemInfo.LogicalCores; id++)
                {
                    final int ID = id;
                    ex.submit(new Runnable()
                    {
                        @Override
                        public void run()
                        {
                            for(int i = ID; i < delta_l.rows(); i+=SystemInfo.LogicalCores)
                            {
                                double change = 0;
                                for(int j = 0; j < delta_l.cols(); j++)
                                    change += delta_l.get(i, j);
                                B_delta_l.increment(i, change*invXsize);
                            }
                            latch.countDown();
                        }
                    });
                }
                
                try
                {
                    latch.await();
                }
                catch (InterruptedException ex1)
                {
                    Logger.getLogger(SGDNetworkTrainer.class.getName()).log(Level.SEVERE, null, ex1);
                }
            }
        }
    }

    private void applyGradient(double eta_cur)
    {
        //apply gradient
        for(int l = 0; l < layersActivation.size(); l++)
        {
            B_updaters.get(l).update(B.get(l), B_deltas.get(l), eta_cur);
            final Matrix W_l = W.get(l);
            final Matrix W_dl = W_deltas.get(l);
            for(int i = 0; i < W_l.rows(); i++)
            {
                Vec W_li = W_l.getRowView(i);
                W_updaters.get(l).get(i).update(W_li, W_dl.getRowView(i), eta_cur);
            }
            regularizer.applyRegularization(W_l, B.get(l));
        }
    }
    
    private void applyGradient(final double eta_cur, ExecutorService ex)
    {
        List<Future<?>> futures = new ArrayList<Future<?>>();
        //apply gradient
        for(int l = 0; l < layersActivation.size(); l++)
        {
            B_updaters.get(l).update(B.get(l), B_deltas.get(l), eta_cur);
            final Matrix W_l = W.get(l);
            final Matrix W_dl = W_deltas.get(l);
            final int L = l;
            for(int indx = 0; indx < W_l.rows(); indx++)
            {
                final int i = indx;
                futures.add(ex.submit(new Runnable()
                {

                    @Override
                    public void run()
                    {
                        Vec W_li = W_l.getRowView(i);
                        W_updaters.get(L).get(i).update(W_li, W_dl.getRowView(i), eta_cur);
                        B.get(L).set(i, regularizer.applyRegularizationToRow(W_li, B.get(L).get(i)));
                    }
                }));
            }
        }
        
        try
        {
            for(Future<?> future : futures)
                future.get();
        }
        catch (InterruptedException e)
        {
        }
        catch (ExecutionException e)
        {
        }
    }
    
    /**
     * Applies dropout to the given matrix
     * @param X the matrix to dropout values from
     * @param randThresh the threshold that a random integer must be less than to get dropped out
     * @param rand the source of randomness
     * @param ex the source of threads for parlallel computation, or {@code null} 
     */
    private static void applyDropout(final Matrix X, final int randThresh, final Random rand, ExecutorService ex)
    {
        if (ex == null)
        {
            for (int i = 0; i < X.rows(); i++)
                for (int j = 0; j < X.cols(); j++)
                    if (rand.nextInt() < randThresh)
                        X.set(i, j, 0.0);
        }
        else
        {
            final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
            for(int id = 0; id < SystemInfo.LogicalCores; id++)
            {
                final int ID = id;
                ex.submit(new Runnable()
                {

                    @Override
                    public void run()
                    {
                        for (int i = ID; i < X.rows(); i+=SystemInfo.LogicalCores)
                            for (int j = 0; j < X.cols(); j++)
                                if (rand.nextInt() < randThresh)
                                    X.set(i, j, 0.0);
                        latch.countDown();
                    }
                });
            }

            try
            {
                latch.await();
            }
            catch (InterruptedException ex1)
            {
                Logger.getLogger(SGDNetworkTrainer.class.getName()).log(Level.SEVERE, null, ex1);
            }
        }
    }

    @Override
    protected SGDNetworkTrainer clone()
    {
        return new SGDNetworkTrainer(this);
    }
}
