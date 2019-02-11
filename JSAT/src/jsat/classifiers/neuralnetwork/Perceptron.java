
package jsat.classifiers.neuralnetwork;

import jsat.SingleWeightVectorModel;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * The perceptron is a simple algorithm that attempts to find a hyperplane that 
 * separates two classes. It may find any possible separating plane, and there 
 * are no guarantees when the data is not linearly separable. 
 * <br>
 * It is equivalent to a single node Neural Network, and is related to SVMs
 * 
 * 
 * @author Edward Raff
 */
public class Perceptron extends BaseUpdateableClassifier implements BinaryScoreClassifier, SingleWeightVectorModel
{

    private static final long serialVersionUID = -3605237847981632020L;
    private double learningRate;
    private double bias;
    private Vec weights;

    /**
     * Creates a new Perceptron learner
     */
    public Perceptron()
    {
        this(0.1, 20);
    }
    
    /**
     * Creates a new Perceptron learner
     * 
     * @param learningRate the rate at which to incorporate the change of errors
     * into the model
     * @param iteratinLimit the maximum number of iterations to perform when converging
     */
    public Perceptron(double learningRate, int iteratinLimit)
    {
        if(learningRate <= 0 || learningRate > 1)
            throw new RuntimeException("Preceptron learning rate must be in the range (0,1]");
        this.learningRate = learningRate;
	setEpochs(epochs);
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(2);
        cr.setProb(output(data), 1);
        
        return cr;
    }

    @Override
    public double getScore(DataPoint dp)
    {
        return weights.dot(dp.getNumericalValues()) + bias;
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
	if(predicting.getNumOfCategories() != 2)
	    throw new FailedToFitException("Perceptrion is for binary problems only");
	weights = new DenseVector(numericAttributes);
	bias = 0;
    }

    @Override
    public void update(DataPoint dataPoint, double weight, int targetClass)
    {
	if(classify(dataPoint).mostLikely() == targetClass)
	    return;//nothing to do
	//else, error
	double c = (targetClass*2-1)*learningRate;
	weights.mutableAdd(c, dataPoint.getNumericalValues());
	bias += c;
    }
    
    private int output(DataPoint input)
    {
        double dot = getScore(input);
        
        return (dot >= 0) ? 1 : 0;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }

    @Override
    public Vec getRawWeight()
    {
        return weights;
    }

    @Override
    public double getBias()
    {
        return bias;
    }
    
    @Override
    public Vec getRawWeight(int index)
    {
        if(index < 1)
            return getRawWeight();
        else
            throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }

    @Override
    public double getBias(int index)
    {
        if (index < 1)
            return getBias();
        else
            throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }

    @Override
    public int numWeightsVecs()
    {
        return 1;
    }
    
    @Override
    public Perceptron clone()
    {
        Perceptron copy = new  Perceptron(learningRate, epochs);
        if(this.weights != null)
            copy.weights = this.weights.clone();
        copy.bias = this.bias;
        
        return copy;
    }    
}
