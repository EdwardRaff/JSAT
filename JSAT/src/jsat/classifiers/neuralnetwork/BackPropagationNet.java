
package jsat.classifiers.neuralnetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.SparceVector;
import jsat.linear.Vec;
import jsat.utils.FakeExecutor;

/**
 *
 * @author Edward Raff
 */
public class BackPropagationNet implements Classifier
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
        
    int[] neuronsPerLayer;
    int iterationLimit;
    double learningRate = 0.1;
    
    /**
     * The layers of the network. The last element in the list is the output layer, all other are hidden layers. Each row in a matrix is a diffrent neuron, and each value in a colum is the weight given to an input neuron.
     */
    List<Matrix> layers;
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

    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(layers.get(layers.size()-1).rows());
        
        Vec outVec = output(addBiasTerm(data.getNumericalValues()));
        
        for(int i = 0; i < outVec.length(); i++)
            cr.setProb(i, outVec.get(i));
        cr.normalize();
        return cr;
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        layers.clear();
        int numInputs = dataSet.getNumNumericalVars();
        int numOutputs = dataSet.getClassSize();
        Random rand = new Random();
        
        
        //The +1 to the column length is the constant '1.0' we will add for the bias term
        //First hidden layer
        layers.add(randomMatrix(neuronsPerLayer[0], numInputs+1, rand));
        //All other hidden layers
        for(int i = 1; i < neuronsPerLayer.length; i++)
            layers.add(randomMatrix(neuronsPerLayer[i], neuronsPerLayer[i-1], rand));
        //Output layer
        layers.add(randomMatrix(numOutputs, neuronsPerLayer[neuronsPerLayer.length-1], rand));
        
        
        //Out data set
        List<DataPointPair<Integer>> dataPoints = dataSet.getAsDPPList();

        for(int i = 0; i < dataPoints.size(); i++)
        {
            DataPoint dp = dataPoints.get(i).getDataPoint();
            DataPoint newDP = new DataPoint(addBiasTerm(dp.getNumericalValues()), dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
            dataPoints.get(i).setDataPoint(newDP);
        }
        
        int iteartions = 0;
        int lec;
        double lastError;
        double error = 0;

        do
        {
            //We do not want to learn the order of the data set, randomize it
            Collections.shuffle(dataPoints);
            lastError = error;
            error = 0;
            
            for(int i  = 0; i < dataPoints.size(); i++)
            {
                DataPointPair<Integer> dpp = dataPoints.get(i);
                Vec inputVec = dpp.getVector();

                Vec expected = new DenseVector(numOutputs);
                expected.set(dpp.getPair(), 1.0);

                
                List<Vec> outputs = outputs(inputVec);
                Vec lastOutput = outputs.get(outputs.size()-1);
                
                Vec delta = expected.subtract(lastOutput);

                error += delta.dot(delta);//sum of the squares
                
                //We now create the error vectors, they are created in reverse order
                List<Vec> errorVecs = new ArrayList<Vec> (layers.size());
                //First one (last output) is special
                Vec lastErrorVec = delta.clone();
                lastErrorVec.pairwiseMultiply(derivative(lastOutput));
                errorVecs.add(lastErrorVec);
                
                //now we backpropigate these errors
                for(int k = outputs.size()-2; k >= 0; k--)
                {
                    //Each error vector needs the error vector previously computed, the matching output, and the Matrix that produced the output 
                    Matrix Wl = layers.get(k+1);
                    Vec mathingOutput = outputs.get(k);
                    
                    Vec errorVec = Wl.transpose().multiply(lastErrorVec);
                    errorVec.pairwiseMultiply(derivative(mathingOutput));
                    errorVecs.add(errorVec);
                    
                    lastErrorVec = errorVec;
                }
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
                    DenseMatrix updateMatrix = new DenseMatrix(errorVec, matrixInput);
                    layers.get(k).mutableAdd(updateMatrix);
                }
                
            }
            
            iteartions++;
        }
        while(iteartions < iterationLimit);
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

    public Classifier clone()
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
        if(input instanceof SparceVector)
            toReturn = new SparceVector(input.length()+1);
        else
            toReturn = new DenseVector(input.length()+1);
        
        for(int i = 0; i < input.length(); i++)
            toReturn.set(i, input.get(i));
        toReturn.set(input.length(), 1.0);
        
        return toReturn;
    }
}
