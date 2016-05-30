
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
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;

/**
 * This is an implementation of the Multi-Class AdaBoost method SAMME (Stagewise Additive Modeling using
 * a Multi-Class Exponential loss function), presented in <i>Multi-class AdaBoost</i> by Ji Zhu, 
 * Saharon Rosset, Hui Zou,&amp;Trevor Hasstie <br>
 * <br>
 * This algorithm reduces to {@link AdaBoostM1 } for binary classification problems. Its often performs 
 * better for <i>k</i> class classification problems, and has a weaker requirement of besting 1/<i>k</i>
 * accuracy for any k instead of 1/2. 
 * 
 * @author Edward Raff
 */
public class SAMME implements Classifier, Parameterized
{

    private static final long serialVersionUID = -3584203799253810599L;
    private Classifier weakLearner;
    private int maxIterations;
    /**
     * The list of weak hypothesis
     */
    private List<Classifier> hypoths;
    /**
     * The weights for each weak learner
     */
    private List<Double> hypWeights;
    private CategoricalData predicting;

    public SAMME(Classifier weakLearner, int maxIterations)
    {
        if(!weakLearner.supportsWeightedData())
            throw new RuntimeException("WeakLearner must support weighted data to be boosted");
        this.weakLearner = weakLearner;
        this.maxIterations = maxIterations;
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

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        predicting = dataSet.getPredicting();
        hypWeights = new DoubleList(maxIterations);
        hypoths = new ArrayList<Classifier>();
        /**
         * The number of classes we are predicting
         */
        int K = predicting.getNumOfCategories();
        double logK = Math.log(K-1.0)/Math.log(2);
        
        List<DataPointPair<Integer>> dataPoints = dataSet.getAsDPPList();
        //Initialization step, set up the weights  so they are all 1 / size of dataset
        for(DataPointPair<Integer> dpp : dataPoints)
            dpp.getDataPoint().setWeight(1.0);//Scaled, they are all 1 
        double sumOfWeights = dataPoints.size();
        
        
        //Rather then reclasify points, we just save this list
        boolean[] wasCorrect = new boolean[dataPoints.size()];
        
        for(int t = 0; t < maxIterations; t++)
        {
            if(threadPool == null || threadPool instanceof FakeExecutor)
                weakLearner.trainC(new ClassificationDataSet(dataPoints, predicting));
            else
                weakLearner.trainC(new ClassificationDataSet(dataPoints, predicting), threadPool);

            //Error is the same as in AdaBoost.M1
            double error = 0.0;
            for(int i = 0; i < dataPoints.size(); i++)
                if( !(wasCorrect[i] = weakLearner.classify(dataPoints.get(i).getDataPoint()).mostLikely() == dataPoints.get(i).getPair()) )
                    error += dataPoints.get(i).getDataPoint().getWeight();
            error /= sumOfWeights;
            if(error >= (1.0-1.0/K) || error == 0.0)///Diference, we only need to be better then random guessing classes 
                return;
            //The main difference - a different error term
            double am = Math.log((1.0-error)/error)/Math.log(2) +logK;
            
            //Update Distribution weights 
            for(int i = 0; i < wasCorrect.length; i++)
            {
                DataPoint dp = dataPoints.get(i).getDataPoint();
                if(!wasCorrect[i])
                {
                    double w = dp.getWeight();
                    double newW = w*Math.exp(am);
                    sumOfWeights += (newW-w);
                    dp.setWeight(newW);
                }
            }
            
            hypoths.add(weakLearner.clone());
            hypWeights.add(am);
        }
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
    public SAMME clone()
    {
        SAMME clone = new SAMME(weakLearner.clone(), maxIterations);
        if(this.hypWeights != null)
            clone.hypWeights = new DoubleList(this.hypWeights);
        if(this.hypoths != null)
        {
            clone.hypoths = new ArrayList<Classifier>(this.hypoths.size());
            for(int i = 0; i < this.hypoths.size(); i++)
                clone.hypoths.add(this.hypoths.get(i).clone());
        }
        if(this.predicting != null)
            clone.predicting = this.predicting.clone();
        
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
