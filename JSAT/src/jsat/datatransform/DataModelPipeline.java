package jsat.datatransform;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;

/**
 * A Data Model Pipeline combines several data transforms and a base Classifier 
 * or Regressor into a unified object for performing classification and 
 * Regression with. This is useful for certain transforms for which their 
 * behavior is more tightly coupled with the model being used. In addition this 
 * allows a way for easily turning the parameters for a transform along with 
 * those of the predictor. <br>
 * When using the Data Model Pipeline, the transforms that are apart of the 
 * pipeline should not be added to the model evaluators - as this will cause the
 * transforms to be applied multiple times. 
 * 
 * @author Edward Raff
 */
public class DataModelPipeline implements Classifier, Regressor, Parameterized
{

    private static final long serialVersionUID = -2300996837897094414L;
    @ParameterHolder(skipSelfNamePrefix = true)
    private DataTransformProcess baseDtp;
    private Classifier baseClassifier;
    private Regressor baseRegressor;
    
    private DataTransformProcess learnedDtp;
    private Classifier learnedClassifier;
    private Regressor learnedRegressor;

    /**
     * Creates a new Data Model Pipeline from the given transform process and 
     * base classifier
     * @param dtp the data transforms to apply
     * @param baseClassifier the classifier to learn with
     */
    public DataModelPipeline(Classifier baseClassifier, DataTransformProcess dtp)
    {
        this.baseDtp = dtp;
        this.baseClassifier = baseClassifier;
        if(baseClassifier instanceof Regressor)
            this.baseRegressor = (Regressor) baseClassifier;
    }
    
    /**
     * Creates a new Data Model Pipeline from the given transform factories and 
     * base classifier
     * @param transforms the data transforms to apply
     * @param baseClassifier the classifier to learn with
     */
    public DataModelPipeline(Classifier baseClassifier, DataTransform... transforms)
    {
        this(baseClassifier, new DataTransformProcess(transforms));
    }
    
    /**
     * Creates a new Data Model Pipeline from the given transform process and
     * base regressor
     * @param dtp the data transforms to apply
     * @param baseRegressor the regressor to learn with
     */
    public DataModelPipeline(Regressor baseRegressor, DataTransformProcess dtp)
    {
        this.baseDtp = dtp;
        this.baseRegressor = baseRegressor;
        if(baseRegressor instanceof Classifier)
            this.baseClassifier = (Classifier) baseRegressor;
    }
    
    /**
     * Creates a new Data Model Pipeline from the given transform factories and 
     * base classifier
     * @param transforms the data transforms to apply
     * @param baseRegressor the regressor to learn with
     */
    public DataModelPipeline(Regressor baseRegressor, DataTransform... transforms)
    {
        this(baseRegressor, new DataTransformProcess(transforms));
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public DataModelPipeline(DataModelPipeline toCopy)
    {
        this.baseDtp = toCopy.baseDtp.clone();
        if(toCopy.baseClassifier != null && toCopy.baseClassifier == toCopy.baseRegressor)//only possible if both a classifier and regressor
        {
            this.baseClassifier = toCopy.baseClassifier.clone();
            this.baseRegressor = (Regressor) this.baseClassifier;
        }
        else if(toCopy.baseClassifier != null)
            this.baseClassifier = toCopy.baseClassifier.clone();
        else if(toCopy.baseRegressor != null)
            this.baseRegressor = toCopy.baseRegressor.clone();
        else
            throw new RuntimeException("BUG: Report Me!");
                    
        
        if(toCopy.learnedDtp != null)
            this.learnedDtp = toCopy.learnedDtp.clone();
        if(toCopy.learnedClassifier != null)
            this.learnedClassifier = toCopy.learnedClassifier.clone();
        if(toCopy.learnedRegressor != null)
            this.learnedRegressor = toCopy.learnedRegressor.clone();
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        return learnedClassifier.classify(learnedDtp.transform(data));
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        learnedDtp = baseDtp.clone();
        dataSet = dataSet.shallowClone();//dont want to actually edit the data set they gave us
        learnedDtp.learnApplyTransforms(dataSet);
        
        learnedClassifier = baseClassifier.clone();
        if(threadPool == null)
            learnedClassifier.trainC(dataSet);
        else
            learnedClassifier.trainC(dataSet, threadPool);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
    }

    @Override
    public boolean supportsWeightedData()
    {
        if(baseClassifier != null)
            return baseClassifier.supportsWeightedData();
        else if(baseRegressor != null)
            return baseRegressor.supportsWeightedData();
        else
            throw new RuntimeException("BUG: Report Me! This should not have happened");
    }

    @Override
    public double regress(DataPoint data)
    {
        return learnedRegressor.regress(learnedDtp.transform(data));
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        learnedDtp = baseDtp.clone();
        dataSet = dataSet.shallowClone();//dont want to actually edit the data set they gave us
        learnedDtp.learnApplyTransforms(dataSet);
        
        learnedRegressor = baseRegressor.clone();
        if(threadPool == null)
            learnedRegressor.train(dataSet);
        else
            learnedRegressor.train(dataSet, threadPool);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, null);
    }

    @Override
    public DataModelPipeline clone()
    {
        return new DataModelPipeline(this);
    }

    @Override
    public List<Parameter> getParameters()
    {
        List<Parameter> params = Parameter.getParamsFromMethods(this);
        if(baseClassifier != null && baseClassifier instanceof Parameterized)
            params.addAll(((Parameterized)baseClassifier).getParameters());
        else if(baseRegressor != null && baseRegressor instanceof Parameterized)
            params.addAll(((Parameterized)baseRegressor).getParameters());
        return params;
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
    
}
