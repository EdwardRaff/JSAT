
package jsat.classifiers.neuralnetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import static jsat.utils.SystemInfo.*;

/**
 * An implementation of a Back Propagation Neural Network (NN). NNs are powerful 
 * classifiers and regressors, but can suffer from slow training time and overfitting. <br>
 * <br>
 * Online methods often provide faster convergence for a NN, but can not be made parallel. 
 * Batch processing can be made parallel, but does not converge as quickly. Both these 
 * training methods are implemented. Calling 
 * {@link #train(jsat.regression.RegressionDataSet, java.util.concurrent.ExecutorService) } or 
 * {@link #trainC(jsat.classifiers.ClassificationDataSet, java.util.concurrent.ExecutorService) } 
 * will result in the batch mode being used. 
 * 
 * @author Edward Raff
 */
public class BackPropagationNet implements Classifier, Regressor
{   
    static public interface StepFunction
    {
        public double activation(double in);
        public double derivative(double val);
    }
    
    static public class SigmoidStep implements StepFunction
    {

        public double activation(double in)
        {
            return 1 / (1 + Math.exp(-in*4));
        }

        public double derivative(double val)
        {
            return val*(1-val);
        }
        
    }
        
    private int[] neuronsPerLayer;
    private int iterationLimit;
    private double learningRate = 0.1;
    /**
     * The number of inputs to the first layer of the network
     */
    private int numInputs;
    /**
     * The number of outputs from the final layer 
     */
    private int numOutputs;
    
    
    /**
     * The layers of the network. The last element in the list is the output layer, all other are hidden layers. 
     * Each row in a matrix is a different neuron, and each value in a colum is the weight given to an input neuron.
     */
    private List<Matrix> layers;
    private final StepFunction stepFunc;

    public BackPropagationNet(int[] neuronsPerLayer)
    {
        this(neuronsPerLayer, new SigmoidStep(), 400);
    }
    
    
    public BackPropagationNet(int[] neuronsPerLayer, StepFunction stepFunc, int iterationLimit)
    {
        this.neuronsPerLayer = neuronsPerLayer;
        layers = new ArrayList<Matrix>(neuronsPerLayer.length+1);
        this.iterationLimit = iterationLimit;
        this.stepFunc = stepFunc;
    }

    /**
     * Sets the maximal number of training iterations that the network may perform. 
     * @param iterationLimit the maximum number of iterations
     * @throws ArithmeticException if a non positive iteration limit is given
     */
    public void setIterationLimit(int iterationLimit)
    {
        if(iterationLimit <= 0)
            throw new ArithmeticException("A positive iteration count must be given, not " + iterationLimit);
        this.iterationLimit = iterationLimit;
    }

    /**
     * Returns the maximal number of iterations that the network may perform
     * @return the maximum number of iterations
     */
    public int getIterationLimit()
    {
        return iterationLimit;
    }

    /**
     * Sets the learning rate used during training. High learning rates can lead to model
     * fluctuation, and low learning rates can prevent convergence. 
     * 
     * @param learningRate the rate at which errors will be incorporated into the model
     * @throws ArithmeticException if a non positive learning rate is given 
     */
    public void setLearningRate(double learningRate)
    {
        if(Double.isInfinite(learningRate) || Double.isNaN(learningRate) || learningRate <= 0)
            throw new ArithmeticException("Invalid learning rate given, value must be in the range (0, Inf), not " + learningRate);
        this.learningRate = learningRate;
    }

    /**
     * Returns the learning rate used during training, which specified how much of the error is incorporated at each step. 
     * @return the learning rate
     */
    public double getLearningRate()
    {
        return learningRate;
    }
    
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(numOutputs);
        
        Vec outVec = output(addBiasTerm(data.getNumericalValues()));
        
        for(int i = 0; i < outVec.length(); i++)
            cr.setProb(i, outVec.get(i));
        cr.normalize();
        return cr;
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        layers.clear();
        numInputs = dataSet.getNumNumericalVars();
        numOutputs = dataSet.getClassSize();
        //Output layer
        fillRandomLayers();
        
        
        //Out data set
        final List<DataPointPair<Integer>> dataPoints = dataSet.getAsDPPList();
        
        //We create zeroed out matrices for each core, which will get batch updates that will then be collected after each itteration 
        List<List<Matrix>> networkUpdates = new ArrayList<List<Matrix>>(LogicalCores);
        for(int i = 0; i < LogicalCores; i++)
        {
            List<Matrix> networkUpdate = new ArrayList<Matrix>(layers.size());
            for(Matrix m : layers)
                networkUpdate.add(new DenseMatrix(m.rows(), m.cols()));
            networkUpdates.add(networkUpdate);
        }

        for(int i = 0; i < dataPoints.size(); i++)
        {
            DataPoint dp = dataPoints.get(i).getDataPoint();
            DataPoint newDP = new DataPoint(addBiasTerm(dp.getNumericalValues()), dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
            dataPoints.get(i).setDataPoint(newDP);
        }
        
        List<Future<Double>> futures = new ArrayList<Future<Double>>(LogicalCores);
        
        int iteartions = 0;
        double lastError;
        double error = 0;

        do
        {
            lastError = error;
            error = 0;
            
            
            
            //Que up jobs
            for(int id = 0; id < LogicalCores; id++)
            {
                final int threadID = id;
                final List<Matrix> myUpdateLayer = networkUpdates.get(id);
                final List<Vec> errorVecs = new ArrayList<Vec> (layers.size());
                final Vec expected = new DenseVector(numOutputs);
                
                Future<Double> future = threadPool.submit(new Callable<Double>() {

                    public Double call()
                    {
                        double error = 0;
                        for(int i  = threadID; i < dataPoints.size(); i+=LogicalCores)
                        {
                            DataPointPair<Integer> dpp = dataPoints.get(i);
                            Vec inputVec = dpp.getVector();

                            expected.zeroOut();
                            expected.set(dpp.getPair(), 1.0);
                            error += learnExample(inputVec, expected, errorVecs, myUpdateLayer);
                        }
                        
                        return error;
                    }
                });
                
                futures.add(future);
            }
            
            //Collect the resutls and perform batch updates
            try
            {
                for (Future<Double> future : futures)
                    error += future.get();
                
                //Once all the futures have been grabbed, all the networkUpdates have been filled
                for(List<Matrix> networkUpdate : networkUpdates)
                {
                    for(int i = 0; i < networkUpdate.size(); i++)
                    {
                        layers.get(i).mutableAdd(1.0/(dataPoints.size()/LogicalCores), networkUpdate.get(i));
                        networkUpdate.get(i).zeroOut();//Zero out so it can be filled up again
                    }
                }
                
            }
            catch (InterruptedException interruptedException)
            {
            }
            catch (ExecutionException executionException)
            {
            }
            
            iteartions++;
        }
        while(iteartions < iterationLimit);
    }
    
    public void trainC(ClassificationDataSet dataSet)
    {
        layers.clear();
        numInputs = dataSet.getNumNumericalVars();
        numOutputs = dataSet.getClassSize();
        //Output layer
        fillRandomLayers();
        
        
        //Out data set
        List<DataPointPair<Integer>> dataPoints = dataSet.getAsDPPList();

        for(int i = 0; i < dataPoints.size(); i++)
        {
            DataPoint dp = dataPoints.get(i).getDataPoint();
            DataPoint newDP = new DataPoint(addBiasTerm(dp.getNumericalValues()), dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
            dataPoints.get(i).setDataPoint(newDP);
        }
        
        int iteartions = 0;
        double lastError;
        double error = 0;

        do
        {
            //We do not want to learn the order of the data set, randomize it
            Collections.shuffle(dataPoints);
            lastError = error;
            error = 0;
            
            Vec expected = new DenseVector(numOutputs);
            List<Vec> errorVecs = new ArrayList<Vec> (layers.size());
            for(int i  = 0; i < dataPoints.size(); i++)
            {
                DataPointPair<Integer> dpp = dataPoints.get(i);
                Vec inputVec = dpp.getVector();

                expected.zeroOut();
                expected.set(dpp.getPair(), 1.0);
                error += learnExample(inputVec, expected, errorVecs, null);
                
            }
            
            iteartions++;
        }
        while(iteartions < iterationLimit);
    }
    
    public double regress(DataPoint data)
    {
        return output(addBiasTerm(data.getNumericalValues())).get(0);
    }

    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        layers.clear();
        numInputs = dataSet.getNumNumericalVars();
        numOutputs = 1;
        //Output layer
        fillRandomLayers();
        
        
        //Out data set
        final List<DataPointPair<Double>> dataPoints = dataSet.getAsDPPList();
        
        //We create zeroed out matrices for each core, which will get batch updates that will then be collected after each itteration 
        List<List<Matrix>> networkUpdates = new ArrayList<List<Matrix>>(LogicalCores);
        for(int i = 0; i < LogicalCores; i++)
        {
            List<Matrix> networkUpdate = new ArrayList<Matrix>(layers.size());
            for(Matrix m : layers)
                networkUpdate.add(new DenseMatrix(m.rows(), m.cols()));
            networkUpdates.add(networkUpdate);
        }

        for(int i = 0; i < dataPoints.size(); i++)
        {
            DataPoint dp = dataPoints.get(i).getDataPoint();
            DataPoint newDP = new DataPoint(addBiasTerm(dp.getNumericalValues()), dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
            dataPoints.get(i).setDataPoint(newDP);
        }
        
        List<Future<Double>> futures = new ArrayList<Future<Double>>(LogicalCores);
        
        int iteartions = 0;
        double lastError;
        double error = 0;

        do
        {
            lastError = error;
            error = 0;
            
            
            
            //Que up jobs
            for(int id = 0; id < LogicalCores; id++)
            {
                final int threadID = id;
                final List<Matrix> myUpdateLayer = networkUpdates.get(id);
                final List<Vec> errorVecs = new ArrayList<Vec> (layers.size());
                final Vec expected = new DenseVector(numOutputs);
                
                Future<Double> future = threadPool.submit(new Callable<Double>() {

                    public Double call()
                    {
                        double error = 0;
                        for(int i  = threadID; i < dataPoints.size(); i+=LogicalCores)
                        {
                            DataPointPair<Double> dpp = dataPoints.get(i);
                            Vec inputVec = dpp.getVector();

                            expected.zeroOut();
                            expected.set(0, dpp.getPair());
                            error += learnExample(inputVec, expected, errorVecs, myUpdateLayer);
                        }
                        
                        return error;
                    }
                });
                
                futures.add(future);
            }
            
            //Collect the resutls and perform batch updates
            try
            {
                for (Future<Double> future : futures)
                    error += future.get();
                
                //Once all the futures have been grabbed, all the networkUpdates have been filled
                for(List<Matrix> networkUpdate : networkUpdates)
                {
                    for(int i = 0; i < networkUpdate.size(); i++)
                    {
                        layers.get(i).mutableAdd(1.0/(dataPoints.size()/LogicalCores), networkUpdate.get(i));
                        networkUpdate.get(i).zeroOut();//Zero out so it can be filled up again
                    }
                }
                
            }
            catch (InterruptedException interruptedException)
            {
            }
            catch (ExecutionException executionException)
            {
            }
            
            iteartions++;
        }
        while(iteartions < iterationLimit);
    }

    public void train(RegressionDataSet dataSet)
    {
        layers.clear();
        numInputs = dataSet.getNumNumericalVars();
        numOutputs = 1;
        
        fillRandomLayers();
                
        //Out data set
        List<DataPointPair<Double>> dataPoints = dataSet.getAsDPPList();

        for(int i = 0; i < dataPoints.size(); i++)
        {
            DataPoint dp = dataPoints.get(i).getDataPoint();
            DataPoint newDP = new DataPoint(addBiasTerm(dp.getNumericalValues()), dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
            dataPoints.get(i).setDataPoint(newDP);
        }
        
        int iteartions = 0;
        double lastError;
        double error = 0;
        
        do
        {
            //We do not want to learn the order of the data set, randomize it
            Collections.shuffle(dataPoints);
            lastError = error;
            error = 0;
            
            Vec expected = new DenseVector(numOutputs);
            List<Vec> errorVecs = new ArrayList<Vec> (layers.size());
            for(int i  = 0; i < dataPoints.size(); i++)
            {
                DataPointPair<Double> dpp = dataPoints.get(i);
                Vec inputVec = dpp.getVector();

                expected.zeroOut();
                expected.set(0, dpp.getPair());//Only one value, the regression target
                error += learnExample(inputVec, expected, errorVecs, null);
                
            }
            
            iteartions++;
        }
        while(iteartions < iterationLimit);
    }

    /**
     * Back computes the error vectors for the back propagation
     * @param outputs the output of each layer
     * @param lastErrorVec the error of the final layer
     * @param errorVecs the location to store each back propagated error vector 
     */
    private void backPropagateErrors(List<Vec> outputs, Vec lastErrorVec, List<Vec> errorVecs)
    {
        //now we backpropigate these errors
        for(int k = outputs.size()-2; k >= 0; k--)
        {
            //Each error vector needs the error vector previously computed, the matching output, and the Matrix that produced the output 
            Matrix Wl = layers.get(k+1);
            Vec mathingOutput = outputs.get(k);
            
            Vec errorVec = Wl.transposeMultiply(1.0, lastErrorVec);
            errorVec.pairwiseMultiply(derivative(mathingOutput));
            errorVecs.add(errorVec);
            
            lastErrorVec = errorVec;
        }
    }

    /** 
     * Fills the layers of the NN with the right matrices full of random values 
     */
    private void fillRandomLayers()
    {
        Random rand = new Random();
        //The +1 to the column length is the constant '1.0' we will add for the bias term
        //First hidden layer
        layers.add(randomMatrix(neuronsPerLayer[0], numInputs+1, rand));
        //All other hidden layers
        for(int i = 1; i < neuronsPerLayer.length; i++)
            layers.add(randomMatrix(neuronsPerLayer[i], neuronsPerLayer[i-1], rand));
        //Output layer
        layers.add(randomMatrix(numOutputs, neuronsPerLayer[neuronsPerLayer.length-1], rand));
    }

    /**
     * Performs the work to learn one example, propagating the information back through the network. 
     * @param inputVec the input vector to learn
     * @param expected the expected output for the input
     * @param errorVecs a storage place to hold error vectors from each layer. This list will be cleared before use
     * @param the list of matrices to store the Layer updates in. Only used if non null
     * @return the Network's error for thsi input example 
     */
    private double learnExample(Vec inputVec, Vec expected, List<Vec> errorVecs, List<Matrix> updateStore)
    {
        List<Vec> outputs = outputs(inputVec);
        Vec lastOutput = outputs.get(outputs.size()-1);
        Vec delta = expected.subtract(lastOutput);
        double error = delta.dot(delta);//sum of the squares
        //We now create the error vectors, they are created in reverse order
        errorVecs.clear();
        //First one (last output) is special
        Vec lastErrorVec = delta.clone();
        lastErrorVec.pairwiseMultiply(derivative(lastOutput));
        errorVecs.add(lastErrorVec);
        backPropagateErrors(outputs, lastErrorVec, errorVecs);
        //Now reverse the errorVecs array so they are in the same order as the matrix array
        Collections.reverse(errorVecs);
        //Now we adjust the weight Matrices
        //W_l = W_l + learningRate * (errorVec_l * output_(l-1)^T )
        /* We alter the Neuron matrix by adding to it the error Vectors,
        /* which tell us how much error we are blaiming to each input neuron,
        /* times the value of the output of the neurons that informed out deicious.
         */
        //We add the input to the front of this list, as it is the inital "output"
        outputs.add(0, inputVec);
        for(int k = 0; k < layers.size(); k++)
        {
            Vec errorVec = errorVecs.get(k);
            Vec matrixInput = outputs.get(k);
            
            errorVec.mutableMultiply(learningRate);
            if(updateStore == null)
                Matrix.OuterProductUpdate(layers.get(k), errorVec, matrixInput, 1.0);
            else
                Matrix.OuterProductUpdate(updateStore.get(k), errorVec, matrixInput, 1.0);
        }
        return error;
    }
    
    /**
     * 
     * @param input the input to feed into the network
     * @return a list of the Vectors representing the output from each layer
     */
    private List<Vec> outputs(Vec input)
    {
        List<Vec> ouputs = new ArrayList<Vec> (layers.size());
        //The output of layer n, represented by matrix W_n, is W_n * out_(n-1)
        for(Matrix W : layers)
        {
            input = W.multiply(input);
            
            for(int i = 0; i < input.length(); i++)
                input.set(i, stepFunc.activation(input.get(i)));
            ouputs.add(input);
        }
        
        return ouputs;
    }
    
    private Vec output(Vec input)
    {
        List<Vec> outputs = outputs(input);
        
        return outputs.get(outputs.size()-1);
    }

    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public BackPropagationNet clone()
    {
        BackPropagationNet copy = new BackPropagationNet(neuronsPerLayer, stepFunc, iterationLimit);
        copy.layers.clear();
        for(Matrix m : this.layers)
            copy.layers.add(m.clone());
        
        return copy;
    }
    
    private static DenseMatrix randomMatrix(int rows, int cols, Random rand)
    {
        DenseMatrix newMatrix = new DenseMatrix(rows, cols);
        
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols-1; j++)
                newMatrix.set(i, j, (rand.nextDouble()*2-1)*0.5 /* [-0.5, 0.5]*/ );
            //The last column is the bias, which we default to 0
            newMatrix.set(i, cols-1, 0 );
        }
        
        return newMatrix;
    }

    /**
     * Computes the vector that contains the derivatives of the values given
     * @param v the input vector
     * @return a new derivative vector
     */
    private Vec derivative(Vec v)
    {
        Vec der = new DenseVector(v.length());
        for(int i = 0; i < v.length(); i++)
            der.set(i, stepFunc.derivative(v.get(i)));
        return der;
    }
    
    /**
     * @param input
     * @return a new vector that has all the same values, but is 1 long and 
     * contains a 1.0 for the bias term
     */
    private static Vec addBiasTerm(Vec input)
    {
        Vec toReturn;
        if(input.isSparse())
            toReturn = new SparseVector(input.length()+1);
        else
            toReturn = new DenseVector(input.length()+1);
        
        for(int i = 0; i < input.length(); i++)
            toReturn.set(i, input.get(i));
        toReturn.set(input.length(), 1.0);
        
        return toReturn;
    }
}
