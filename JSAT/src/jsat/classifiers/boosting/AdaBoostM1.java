
package jsat.classifiers.boosting;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.OneVSAll;
import jsat.exceptions.FailedToFitException;
import jsat.parameters.Parameter;
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

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        /*
         * Implementation note: We want all weights to be >= 1, so we will scale all weight values by the smallest weight value 
         */
        predicting = dataSet.getPredicting();
        hypWeights = new DoubleList(maxIterations);
        hypoths = new ArrayList<Classifier>(maxIterations);
        
        List<DataPointPair<Integer>> dataPoints = dataSet.getAsDPPList();
        //Initialization step, set up the weights  so they are all 1 / size of dataset
        for(DataPointPair<Integer> dpp : dataPoints)
            dpp.getDataPoint().setWeight(1.0);//Scaled, they are all 1 
        double scaledBy = dataPoints.size();
        
        
        //Rather then reclasify points, we just save this list
        boolean[] wasCorrect = new boolean[dataPoints.size()];
        
        for(int t = 0; t < maxIterations; t++)
        {
            if(threadPool != null)
                weakLearner.trainC(new ClassificationDataSet(dataPoints, predicting), threadPool);
            else
                weakLearner.trainC(new ClassificationDataSet(dataPoints, predicting));

            double error = 0.0;
            for(int i = 0; i < dataPoints.size(); i++)
                if( !(wasCorrect[i] = weakLearner.classify(dataPoints.get(i).getDataPoint()).mostLikely() == dataPoints.get(i).getPair()) )
                    error += dataPoints.get(i).getDataPoint().getWeight();
            error /= scaledBy;
            if(error > 0.5 || error == 0.0)
                return;
            
            double bt = error /( 1.0 - error );
            
            //Update Distribution weights 
            double Zt = 0.0;
            double newScale = scaledBy;//Not scaled
            for(int i = 0; i < wasCorrect.length; i++)
            {
                DataPoint dp = dataPoints.get(i).getDataPoint();
                if(wasCorrect[i])//Put less weight on the points we got correct
                {
                    double w = dp.getWeight()*bt;
                    dp.setWeight(w);
                }
                double trueWeight = dp.getWeight()/scaledBy;
                if(1.0/trueWeight > newScale)
                    newScale = 1.0/trueWeight;
                Zt += dp.getWeight()/scaledBy;//Sum the values
            }
            
            for(DataPointPair dpp : dataPoints)//Normalize so the weights make a distribution
                dpp.getDataPoint().setWeight(dpp.getDataPoint().getWeight()/scaledBy*newScale/Zt);
            scaledBy = newScale;
            
            hypoths.add(weakLearner.clone());
            hypWeights.add(Math.log(1/bt));
        }
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
    }

    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public AdaBoostM1 clone()
    {	
        AdaBoostM1 copy = new AdaBoostM1( weakLearner.clone(), maxIterations);
        if(hypWeights != null)
            copy.hypWeights = new DoubleList(this.hypWeights);
        if(this.hypoths != null)
        {
            copy.hypoths = new ArrayList<Classifier>(this.hypoths.size());
            for(int i = 0; i < this.hypoths.size(); i++)
                copy.hypoths.add(this.hypoths.get(i).clone());
        }
        if(this.predicting != null)
            copy.predicting = this.predicting.clone();
        return copy;
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
