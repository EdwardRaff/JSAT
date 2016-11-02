
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
 * This class provides a neural network based on Geoffrey Hinton's 
 * <b>D</b>eep <b>Re</b>ctified <b>D</b>ropout <b>N</b>ets. It is parameterized 
 * to be "simpler" in that the default batch size and gradient updating method
 * should require no tuning to get decent results<br>
 * <br>
 * NOTE: Training neural networks is computationally expensive, you may want to 
 * consider a GPU implementation from another source. 
 * 
 * @author Edward Raff
 */
public class DReDNetSimple implements Classifier, Parameterized
{

    private static final long serialVersionUID = -342281027279571332L;
    private SGDNetworkTrainer network;
    private int[] hiddenSizes;
    private int batchSize = 256;
    private int epochs = 100;

    /**
     * Creates a new DRedNet that uses two hidden layers with 1024 neurons each. 
     * A batch size of 256 and 100 epochs will be used. 
     */
    public DReDNetSimple()
    {
        this(1024, 1024);
    }

    /**
     * Create a new DReDNet that uses the specified number of hidden layers. A
     * batch size of 256 and 100 epochs will be used. 
     * @param hiddenLayerSizes the length indicates the number of hidden layers,
     * and the value in each index is the number of neurons in that layer
     */
    public DReDNetSimple(int... hiddenLayerSizes)
    {
        setHiddenSizes(hiddenLayerSizes);
    }

    /**
     * Sets the hidden layer sizes for this network. The size of the array is 
     * the number of hidden layers and the value in each index denotes the size
     * of that layer. 
     * @param hiddenSizes 
     */
    public void setHiddenSizes(int[] hiddenSizes)
    {
        for(int i = 0; i < hiddenSizes.length; i++)
            if(hiddenSizes[i] <= 0)
                throw new IllegalArgumentException("Hidden layer " + i + " must contain a positive number of neurons, not " + hiddenSizes[i]);
        this.hiddenSizes = Arrays.copyOf(hiddenSizes, hiddenSizes.length);
    }

    /**
     * 
     * @return the array of hidden layer sizes
     */
    public int[] getHiddenSizes()
    {
        return hiddenSizes;
    }

    /**
     * Sets the batch size for updates
     * @param batchSize the number of items to compute the gradient from
     */
    public void setBatchSize(int batchSize)
    {
        this.batchSize = batchSize;
    }

    /**
     * 
     * @return the number of data points to use for one gradient computation
     */
    public int getBatchSize()
    {
        return batchSize;
    }

    /**
     * Sets the number of epochs to perform
     * @param epochs the number of training iterations through the whole data 
     * set
     */
    public void setEpochs(int epochs)
    {
        if(epochs <= 0)
            throw new IllegalArgumentException("Number of epochs must be positive");
        this.epochs = epochs;
    }

    /**
     * 
     * @return the number of training iterations through the data set
     */
    public int getEpochs()
    {
        return epochs;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        Vec x = data.getNumericalValues();
        Vec y = network.feedfoward(x);
        return new CategoricalResults(y.arrayCopy());
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        setup(dataSet);
        
        List<Vec> X = dataSet.getDataVectors();
        List<Vec> Y = new ArrayList<Vec>(dataSet.getSampleSize());
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            SparseVector sv = new SparseVector(dataSet.getClassSize(), 1);
            sv.set(dataSet.getDataPointCategory(i), 1.0);
            Y.add(sv);
        }
        IntList randOrder = new IntList(X.size());
        ListUtils.addRange(randOrder, 0, X.size(), 1);
        List<Vec> Xmini = new ArrayList<Vec>(batchSize);
        List<Vec> Ymini = new ArrayList<Vec>(batchSize);
        
        for(int epoch = 0; epoch < epochs; epoch++)
        {
            long start = System.currentTimeMillis();
            double epochError = 0;
            Collections.shuffle(randOrder);
            for(int i = 0; i < X.size(); i+=batchSize)
            {
                int to = Math.min(i+batchSize, X.size());
                Xmini.clear();
                Ymini.clear();
                for(int j = i; j < to; j++)
                {
                    Xmini.add(X.get(j));
                    Ymini.add(Y.get(j));
                }
                
                double localErr;
                if(threadPool != null)
                    localErr = network.updateMiniBatch(Xmini, Ymini, threadPool);
                else
                    localErr = network.updateMiniBatch(Xmini, Ymini);
                epochError += localErr;
            }
            long end = System.currentTimeMillis();
//            System.out.println("Epoch " + epoch + " had error " + epochError + " took " + (end-start)/1000.0 + " seconds");
        }
        
        network.finishUpdating();
    }

    private void setup(ClassificationDataSet dataSet)
    {
        network = new SGDNetworkTrainer();
        int[] sizes = new int[hiddenSizes.length+2];
        sizes[0] = dataSet.getNumNumericalVars();
        for(int i = 0; i < hiddenSizes.length; i++)
            sizes[i+1] = hiddenSizes[i];
        sizes[sizes.length-1] = dataSet.getClassSize();
        network.setLayerSizes(sizes);
        
        List<ActivationLayer> activations = new ArrayList<ActivationLayer>(hiddenSizes.length+2);
        for(int size : hiddenSizes)
            activations.add(new ReLU());
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
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public DReDNetSimple clone()
    {
        DReDNetSimple clone = new DReDNetSimple(hiddenSizes);
        if(this.network != null)
            clone.network = this.network.clone();
        clone.batchSize = this.batchSize;
        clone.epochs = this.epochs;
        return clone;
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
