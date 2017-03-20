
package jsat.classifiers.neuralnetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import jsat.SingleWeightVectorModel;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.utils.PairedReturn;
import jsat.utils.random.RandomUtil;

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
public class Perceptron implements BinaryScoreClassifier, SingleWeightVectorModel
{


	private static final long serialVersionUID = -3605237847981632021L;
	private double learningRate;
    private double bias;
    private Vec weights;
    private int iteratinLimit;

    /**
     * Creates a new Perceptron learner
     */
    public Perceptron()
    {
        this(0.1, 400);
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
        this.iteratinLimit = iteratinLimit;
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

    /**
     * We use the probability match object to return both the vector and the bias term.
     * The first index in the double will contain the change in bias, the 2nd 
     * will contain the change in global error
     */
    private class BatchTrainingUnit implements Callable<PairedReturn<Vec, Double[]>>
    {
        //this will be updated incrementally
        private Vec tmpSummedErrors;
        private double biasChange;
        private double globalError;

        
        List<DataPointPair<Integer>> dataPoints;

        public BatchTrainingUnit(List<DataPointPair<Integer>> toOperateOn)
        {
            this.tmpSummedErrors = new DenseVector(weights.length());
            this.dataPoints = toOperateOn;
            this.globalError = 0;
            this.biasChange = 0;
        }

        public PairedReturn<Vec, Double[]> call() throws Exception
        {
            for(DataPointPair<Integer> dpp : dataPoints)
            {
                
                int output = output(dpp.getDataPoint());
                double localError = dpp.getPair() - output;
                
                
                if(localError != 0)
                {//Update the weight vecotrs
                    //The weight of this sample, take it into account!
                    double extraWeight = dpp.getDataPoint().getWeight();
                    
                    double magnitude = learningRate*localError*extraWeight;
                
                    tmpSummedErrors.mutableAdd(magnitude, dpp.getVector());
                    biasChange += magnitude;
                    globalError += Math.abs(localError)*extraWeight;
                } 
            }
            
            return new PairedReturn<Vec, Double[]>(tmpSummedErrors, new Double[] {biasChange, globalError} );
        }
    }
    
    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        if(dataSet.getClassSize() != 2)
            throw new FailedToFitException("Preceptron only supports binary calssification");
        else if(dataSet.getNumCategoricalVars() != 0)
            throw new FailedToFitException("Preceptron only supports vector classification");

        List<DataPointPair<Integer>> dataPoints = dataSet.getAsDPPList();
        Collections.shuffle(dataPoints);
        
        int partions = Runtime.getRuntime().availableProcessors();
        
        Random r = RandomUtil.getRandom();
        int numerVars = dataSet.getNumNumericalVars();
        
        weights = new DenseVector(numerVars);
        for(int i = 0; i < weights.length(); i++)//give all variables a random weight in the range [0,1]
            weights.set(i, r.nextDouble());
        
        
        Vec bestWeightsSoFar = null;
        double lowestErrorSoFar = Double.MAX_VALUE;
        int iterations = 0;
        bias = 0;
        double globalError;
        do
        {
            globalError = 0;
            final Vec sumedErrors = new DenseVector(weights.length());
            double biasChange = 0;
            
            
            //Where our intermediate partial results will be stored
            List<Future<PairedReturn<Vec, Double[]>>> futures = 
                    new ArrayList<Future<PairedReturn<Vec, Double[]>>> (partions);
            //create a task for each thing being submitied
            int blockSize = dataPoints.size() / partions;
            for(int i = 0; i < partions; i++)
            {
                List<DataPointPair<Integer>> subList;
                if(i == partions -1)
                    subList = dataPoints.subList(i*blockSize, dataPoints.size());
                else
                    subList = dataPoints.subList(i*blockSize, (i+1)*blockSize);
                
                futures.add(threadPool.submit(new BatchTrainingUnit(subList))); 
            }
            
            //Now collect the results
            for(Future<PairedReturn<Vec, Double[]>> future : futures)
            {
                try
                {
                    PairedReturn<Vec, Double[]> partialResult = future.get();
                    sumedErrors.mutableAdd(partialResult.getFirstItem());
                    biasChange += partialResult.getSecondItem()[0];
                    globalError += partialResult.getSecondItem()[1];
                }
                catch (InterruptedException ex)
                {
                    
                }
                catch (ExecutionException ex)
                {
                    
                }
            }
            
            if(globalError < lowestErrorSoFar)
            {
                bestWeightsSoFar = weights;
                lowestErrorSoFar = globalError;
            }
            
            bias += biasChange;
            weights.mutableAdd(sumedErrors);
            
            iterations++;
        }
        while(globalError > 0 && iterations < iteratinLimit);
        
        weights = bestWeightsSoFar;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainCOnline(dataSet);
    }
    
    //Uses the online training algorithm instead of the batch one. 
    public void trainCOnline(ClassificationDataSet dataSet)
    {
        if(dataSet.getClassSize() != 2)
            throw new FailedToFitException("Preceptron only supports binary calssification");
        else if(dataSet.getNumCategoricalVars() != 0)
            throw new FailedToFitException("Preceptron only supports vector classification");

        List<DataPointPair<Integer>> dataPoints = dataSet.getAsDPPList();
        Collections.shuffle(dataPoints);
        
        Random r = RandomUtil.getRandom();
        int numerVars = dataSet.getNumNumericalVars();
        
        weights = new DenseVector(numerVars);
        for(int i = 0; i < weights.length(); i++)//give all variables a random weight in the range [0,1]
            weights.set(i, r.nextDouble());
        
        Vec bestWeightsSoFar = null;
        double lowestErrorSoFar = Double.MAX_VALUE;
        int iterations = 0;
        
        double globalError;
        do
        {
            globalError = 0;
            //For each data point
            for(DataPointPair<Integer> dpp : dataPoints)
            {
                int output = output(dpp.getDataPoint());
                double localError = dpp.getPair() - output;
                
                
                if(localError != 0)
                {//Update the weight vecotrs
                    //The weight of this sample, take it into account!
                    double extraWeight = dpp.getDataPoint().getWeight();
                    
                    double magnitude = learningRate*localError*extraWeight;
                
                    weights.mutableAdd(magnitude, dpp.getVector());
                    bias += magnitude;
                    globalError += Math.abs(localError)*extraWeight;
                }
            }
            
            if(globalError < lowestErrorSoFar)
            {
                bestWeightsSoFar = weights;
                lowestErrorSoFar = globalError;
            }
            iterations++;
        }
        while(globalError > 0 && iterations < iteratinLimit);
        
        weights = bestWeightsSoFar;
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
        Perceptron copy = new  Perceptron(learningRate, iteratinLimit);
        if(this.weights != null)
            copy.weights = this.weights.clone();
        copy.bias = this.bias;
        
        return copy;
    }    
}
