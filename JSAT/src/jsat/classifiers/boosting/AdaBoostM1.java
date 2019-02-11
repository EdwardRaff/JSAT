
package jsat.classifiers.boosting;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.OneVSAll;
import jsat.exceptions.FailedToFitException;
import jsat.linear.Vec;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;

/**
 * Implementation of Experiments with a New Boosting Algorithm, by Yoav Freund&amp;Robert E. Schapire.
 * <br>
 * This is the first AdaBoost algorithm presented in the paper, and the first boosting algorithm. 
 * Though not often mentioned, AdaBoost does support non binary classification tasks. However, 
 * for any <i>k</i> labels, the weak learner's error still needs to be better then 1/2, which 
 * is not an easy requirement to satisfy. For this reason, many use AdaBoostM1 by reducing 
 * <i>k</i> class classification problems to several 2 class problems. 
 * <br><br>
 * Many Boosting methods, when given a binary classification task, reduce to having the same results as this class. 
 * <br> <br>
 * AdaBoost is often combined with {@link OneVSAll} to obtain better classification accuracy. 
 * 
 * 
 * @author Edward Raff
 */
public class AdaBoostM1 implements Classifier, Parameterized
{

    private static final long serialVersionUID = 4205232097748332861L;
    private Classifier weakLearner;
    private int maxIterations;
    /**
     * The list of weak hypothesis
     */
    protected List<Classifier> hypoths;
    /**
     * The weights for each weak learner
     */
    protected List<Double> hypWeights;
    protected CategoricalData predicting;
    
    public AdaBoostM1(Classifier weakLearner, int maxIterations)
    {
        setWeakLearner(weakLearner);
        this.maxIterations = maxIterations;
    }
    
    public AdaBoostM1(AdaBoostM1 toCopy)
    {
        this(toCopy.weakLearner.clone(), toCopy.maxIterations);
        if(toCopy.hypWeights != null)
            this.hypWeights = new DoubleList(toCopy.hypWeights);
        if(toCopy.hypoths != null)
        {
            this.hypoths = new ArrayList<>(toCopy.hypoths.size());
            for(int i = 0; i < toCopy.hypoths.size(); i++)
                this.hypoths.add(toCopy.hypoths.get(i).clone());
        }
        if(toCopy.predicting != null)
            this.predicting = toCopy.predicting.clone();
    }

    /**
     * Returns the maximum number of iterations used
     * @return the maximum number of iterations used
     */
    public int getMaxIterations()
    {
        return maxIterations;
    }

    /**
     * 
     * @return a list of the models that are in this ensemble. 
     */
    public List<Classifier> getModels()
    {
        return Collections.unmodifiableList(hypoths);
    }
    
    /**
     * 
     * @return a list of the models weights that are in this ensemble. 
     */
    public List<Double> getModelWeights()
    {
        return Collections.unmodifiableList(hypWeights);
    }

    /**
     * Sets the maximal number of boosting iterations that may be performed 
     * @param maxIterations the maximum number of iterations
     */
    public void setMaxIterations(int maxIterations)
    {
        if(maxIterations < 1)
            throw new IllegalArgumentException("Number of iterations must be a positive value, no " + maxIterations);
        this.maxIterations = maxIterations;
    }

    /**
     * Returns the weak learner currently being used by this method. 
     * @return the weak learner currently being used by this method. 
     */
    public Classifier getWeakLearner()
    {
        return weakLearner;
    }

    /**
     * Sets the weak learner used during training. 
     * @param weakLearner the weak learner to use
     */
    public void setWeakLearner(Classifier weakLearner)
    {
        if(!weakLearner.supportsWeightedData())
            throw new FailedToFitException("WeakLearner must support weighted data to be boosted");
        this.weakLearner = weakLearner;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(predicting == null)
            throw new RuntimeException("Classifier has not been trained yet");
        
        CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());
        
        for(int i=0; i < hypoths.size(); i++)
            cr.incProb(hypoths.get(i).classify(data).mostLikely(), hypWeights.get(i));
        
        cr.normalize();
        return cr;
    }

    @Override
    public void train(ClassificationDataSet dataSet, boolean parallel)
    {
        /*
         * Implementation note: We want all weights to be >= 1, so we will scale all weight values by the smallest weight value 
         */
        predicting = dataSet.getPredicting();
        hypWeights = new DoubleList(maxIterations);
        hypoths = new ArrayList<>(maxIterations);
        
	
	Vec origWeights = dataSet.getDataWeights();
	
        //Initialization step, set up the weights  so they are all 1 / size of dataset
        for(int i = 0; i < dataSet.size(); i++)
	    dataSet.setWeight(i, 1.0);
        double scaledBy = dataSet.size();
        
        
        //Rather then reclasify points, we just save this list
        boolean[] wasCorrect = new boolean[dataSet.size()];
        
        for(int t = 0; t < maxIterations; t++)
        {
            weakLearner.train(dataSet, parallel);

            double error = 0.0;
            for(int i = 0; i < dataSet.size(); i++)
                if( !(wasCorrect[i] = weakLearner.classify(dataSet.getDataPoint(i)).mostLikely() == dataSet.getDataPointCategory(i)) )
                    error += dataSet.getWeight(i);
            error /= scaledBy;
            if(error > 0.5 || error == 0.0)
                return;
            
            double bt = error /( 1.0 - error );
            
            //Update Distribution weights 
            double Zt = 0.0;
            double newScale = scaledBy;//Not scaled
            for(int i = 0; i < wasCorrect.length; i++)
            {
                DataPoint dp = dataSet.getDataPoint(i);
                if(wasCorrect[i])//Put less weight on the points we got correct
                {
                    double w = dataSet.getWeight(i)*bt;
                    dataSet.setWeight(i, w);
                }
                double trueWeight = dataSet.getWeight(i)/scaledBy;
                if(1.0/trueWeight > newScale)
                    newScale = 1.0/trueWeight;
                Zt += dataSet.getWeight(i)/scaledBy;//Sum the values
            }
            
            for(int i = 0; i < dataSet.size(); i++)//Normalize so the weights make a distribution
                dataSet.setWeight(i, dataSet.getWeight(i)/scaledBy*newScale/Zt);
            scaledBy = newScale;
            
            hypoths.add(weakLearner.clone());
            hypWeights.add(Math.log(1/bt));
        }
	
	for(int i = 0; i < dataSet.size(); i++)
	    dataSet.setWeight(i, origWeights.get(i));
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public AdaBoostM1 clone()
    {	
        return new AdaBoostM1(this);
    }
}
