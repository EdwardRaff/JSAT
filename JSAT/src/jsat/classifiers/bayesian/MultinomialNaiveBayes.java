
package jsat.classifiers.bayesian;

import static java.lang.Math.exp;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.*;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.math.MathTricks;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;

/**
 * An implementation of the Multinomial Naive Bayes model (MNB). In this model,
 * vectors are implicitly assumed to be sparse and that zero values can be 
 * skipped. This model requires that all numeric features be non negative, any
 * negative value will be treated as a zero. <br>
 * <br>Note: the is no reason to ever use more than one 
 * {@link #setEpochs(int) epoch} for MNB<br>
 * <br>MNB requires taking the log probabilities to perform predictions, which
 * created a trade off. Updating the classifier requires the non log form, but
 * updates require the log form, making classification take considerably longer 
 * to take the logs of the probabilities. This can be reduced by 
 * {@link #finalizeModel() finalizing} the model. This prevents the model from 
 * being updated further, but reduces classification time. By default, this will
 * be done after a call to 
 * {@link #trainC(jsat.classifiers.ClassificationDataSet) } but not after 
 * {@link #update(jsat.classifiers.DataPoint, int) }
 * 
 * @author Edward Raff
 */
public class MultinomialNaiveBayes extends BaseUpdateableClassifier implements Parameterized
{

	private static final long serialVersionUID = -469977945722725478L;
	private double[][][] apriori;
    private double[][] wordCounts;
    private double[] totalWords;
    
    private double priorSum = 0;
    private double[] priors;
    /**
     * Smoothing correction.
     * Added in classification instead of addition
     */
    private double smoothing;
    
    private boolean finalizeAfterTraining = true;
    /**
     * No more training
     */
    private boolean finalized;

    /**
     * Creates a new Multinomial model with laplace smoothing
     */
    public MultinomialNaiveBayes()
    {
        this(1.0);
    }
    
    /**
     * Creates a new Multinomial model with the given amount of smoothing
     * @param smoothing the amount of smoothing to apply
     */
    public MultinomialNaiveBayes(double smoothing)
    {
        setSmoothing(smoothing);
        setEpochs(1);
    }

    /**
     * Copy constructor
     * @param other the one to copy
     */
    protected MultinomialNaiveBayes(MultinomialNaiveBayes other)
    {
        this(other.smoothing);
        if(other.apriori != null)
        {
            this.apriori = new double[other.apriori.length][][];
            this.wordCounts = new double[other.wordCounts.length][];
            this.totalWords = Arrays.copyOf(other.totalWords, other.totalWords.length);
            this.priors = Arrays.copyOf(other.priors, other.priors.length);
            this.priorSum = other.priorSum;

            
            for(int c = 0; c < other.apriori.length; c++)
            {
                this.apriori[c] = new double[other.apriori[c].length][];
                for(int j = 0; j < other.apriori[c].length; j++)
                    this.apriori[c][j] = Arrays.copyOf(other.apriori[c][j], 
                            other.apriori[c][j].length);
                this.wordCounts[c] = Arrays.copyOf(other.wordCounts[c], other.wordCounts[c].length);
            }
            
            this.priorSum = other.priorSum;
            this.priors = Arrays.copyOf(other.priors, other.priors.length);
        }
        this.finalizeAfterTraining = other.finalizeAfterTraining;
        this.finalized = other.finalized;
    }

    /**
     * Sets the amount of smoothing applied to the model. <br>
     * Using a value of 1.0 is equivalent to laplace smoothing
     * <br><br>
     * The smoothing can be changed after the model has already been trained 
     * without needed to re-train the model for the change to take effect. 
     * 
     * @param smoothing the positive smoothing constant
     */
    public void setSmoothing(double smoothing)
    {
        if(Double.isNaN(smoothing) || Double.isInfinite(smoothing) || smoothing <= 0)
            throw new IllegalArgumentException("Smoothing constant must be in range (0,Inf), not " + smoothing);
        this.smoothing = smoothing;
    }

    /**
     * 
     * @return the smoothing applied to categorical counts
     */
    public double getSmoothing()
    {
        return smoothing;
    }

    /**
     * If set {@code true}, the model will be finalized after a call to 
     * {@link #trainC(jsat.classifiers.ClassificationDataSet) }. This prevents 
     * the model from being updated in an online fashion for an reduction in 
     * classification time. 
     * 
     * @param finalizeAfterTraining {@code true} to finalize after a call to 
     * train, {@code false} to keep the model updatable. 
     */
    public void setFinalizeAfterTraining(boolean finalizeAfterTraining)
    {
        this.finalizeAfterTraining = finalizeAfterTraining;
    }

    /**
     * Returns {@code true} if the model will be finalized after batch training. 
     * {@code false} if it will be left in an updatable state. 
     * @return {@code true} if the model will be finalized after batch training. 
     */
    public boolean isFinalizeAfterTraining()
    {
        return finalizeAfterTraining;
    }
    
    @Override
    public MultinomialNaiveBayes clone()
    {
        return new MultinomialNaiveBayes(this);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        super.trainC(dataSet, threadPool);
        if(finalizeAfterTraining)
            finalizeModel();
    }
    
    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        super.trainC(dataSet);
        if(finalizeAfterTraining)
            finalizeModel();
    }

    /**
     * Finalizes the current model. This prevents the model from being updated 
     * further, causing {@link #update(jsat.classifiers.DataPoint, int) } to 
     * throw an exception. This finalization reduces the cost of calling 
     * {@link #classify(jsat.classifiers.DataPoint) } 
     */
    public void finalizeModel()
    {
        if(finalized)
            return;
        final double priorSumSmooth = priorSum + priors.length * smoothing;

        for (int c = 0; c < priors.length; c++)
        {
            double logProb = Math.log((priors[c] + smoothing) / priorSumSmooth);
            priors[c] = logProb;

            double[] counts = wordCounts[c];
            double logTotalCounts = Math.log(totalWords[c] + smoothing * counts.length);

            for(int i = 0; i < counts.length; i++)
            {
                //(n/N)^obv
                counts[i] = Math.log(counts[i] + smoothing) - logTotalCounts;
            }

            for (int j = 0; j < apriori[c].length; j++)
            {
                double sum = 0;
                for (int z = 0; z < apriori[c][j].length; z++)
                    sum += apriori[c][j][z] + smoothing;
                for (int z = 0; z < apriori[c][j].length; z++)
                    apriori[c][j][z] = Math.log( (apriori[c][j][z]+smoothing)/sum);
            }
        }
        finalized = true;
    }
    
    
    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        final int nCat = predicting.getNumOfCategories();
        apriori = new double[nCat][categoricalAttributes.length][];
        wordCounts = new double[nCat][numericAttributes];
        totalWords = new double[nCat];
        priors = new double[nCat];
        priorSum = 0.0;
        
        for (int i = 0; i < nCat; i++)
            for (int j = 0; j < categoricalAttributes.length; j++)
                apriori[i][j] = new double[categoricalAttributes[j].getNumOfCategories()];
        finalized = false;
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        if(finalized)
            throw new FailedToFitException("Model has already been finalized, and can no longer be updated");
        final double weight = dataPoint.getWeight();
        final Vec x = dataPoint.getNumericalValues();
        
        //Categorical value updates
        int[] catValues = dataPoint.getCategoricalValues();
        for(int j = 0; j < apriori[targetClass].length; j++)
            apriori[targetClass][j][catValues[j]]+=weight;
        double localCountsAdded = 0;
        for(IndexValue iv : x)
        {
            final double v = iv.getValue();
            if(v < 0)
                continue;
            wordCounts[targetClass][iv.getIndex()] += v*weight;
            localCountsAdded += v*weight;
        }
        totalWords[targetClass] += localCountsAdded;
        priors[targetClass] += weight;
        priorSum += weight;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(apriori == null)
            throw new UntrainedModelException("Model has not been intialized");
        CategoricalResults results = new CategoricalResults(apriori.length);
        double[] logProbs = new double[apriori.length];
        double maxLogProg = Double.NEGATIVE_INFINITY;
        Vec numVals = data.getNumericalValues();
        if(finalized)
        {
            for(int c = 0; c < priors.length; c++)
            {
                double logProb = priors[c];

                double[] counts = wordCounts[c];

                for (IndexValue iv : numVals)
                {
                    //(n/N)^obv
                    logProb += iv.getValue() * counts[iv.getIndex()];
                }

                for (int j = 0; j < apriori[c].length; j++)
                {
                    logProb += apriori[c][j][data.getCategoricalValue(j)];
                }

                logProbs[c] = logProb;
                maxLogProg = Math.max(maxLogProg, logProb);
            }
        }
        else
        {
            final double priorSumSmooth = priorSum+logProbs.length*smoothing;
            for(int c = 0; c < priors.length; c++)
            {
                double logProb = Math.log((priors[c]+smoothing)/priorSumSmooth);

                double[] counts = wordCounts[c];
                double logTotalCounts = Math.log(totalWords[c]+smoothing*counts.length);

                for (IndexValue iv : numVals)
                {
                    //(n/N)^obv
                    logProb += iv.getValue() * (Math.log(counts[iv.getIndex()]+smoothing) - logTotalCounts);
                }

                for (int j = 0; j < apriori[c].length; j++)
                {
                    double sum = 0;
                    for (int z = 0; z < apriori[c][j].length; z++)
                        sum += apriori[c][j][z]+smoothing;
                    double p = apriori[c][j][data.getCategoricalValue(j)]+smoothing;
                    logProb += Math.log(p / sum);
                }

                logProbs[c] = logProb;
                maxLogProg = Math.max(maxLogProg, logProb);
            }
        }
        double denom = MathTricks.logSumExp(logProbs, maxLogProg);

        for (int i = 0; i < results.size(); i++)
            results.setProb(i, exp(logProbs[i] - denom));
        results.normalize();
        return results;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
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
