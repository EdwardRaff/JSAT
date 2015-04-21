package jsat.classifiers;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;

/**
 * This meta algorithm wraps a {@link Regressor} to perform binary 
 * classification. This is done my labeling class 0 data points as "-1" and 
 * class 1 points as "1". The sign of the outputs then determines the class. Not
 * all regression algorithms will work well in this setting, and standard 
 * parameter values need to change. <br>
 * The parameter values returned are exactly those provided by the given 
 * regressor, or an empty list if the regressor does not implement 
 * {@link Parameterized}
 * 
 * @author Edward Raff
 */
public class RegressorToClassifier implements BinaryScoreClassifier, Parameterized
{

	private static final long serialVersionUID = -2607433019826385335L;
	private Regressor regressor;

    /**
     * Creates a new Binary Classifier by using the given regressor 
     * @param regressor the regressor to wrap as a binary classifier 
     */
    public RegressorToClassifier(Regressor regressor)
    {
        this.regressor = regressor;
    }

    @Override
    public double getScore(DataPoint dp)
    {
        return regressor.regress(dp);
    }

    @Override
    public RegressorToClassifier clone()
    {
        return new RegressorToClassifier(regressor.clone());
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(2);
        if(getScore(data) > 0)
            cr.setProb(1, 1.0);
        else
            cr.setProb(0, 1.0);
            
        return cr;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        RegressionDataSet rds = getRegressionDataSet(dataSet);
        regressor.train(rds, threadPool);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        RegressionDataSet rds = getRegressionDataSet(dataSet);
        regressor.train(rds);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return regressor.supportsWeightedData();
    }

    private RegressionDataSet getRegressionDataSet(ClassificationDataSet dataSet)
    {
        RegressionDataSet rds = new RegressionDataSet(dataSet.getNumNumericalVars(), dataSet.getCategories());
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            rds.addDataPoint(dataSet.getDataPoint(i), dataSet.getDataPointCategory(i)*2-1);
        return rds;
    }

    @Override
    public List<Parameter> getParameters()
    {
        if(regressor instanceof Parameterized)
            return ((Parameterized)regressor).getParameters();
        else
            return Collections.EMPTY_LIST;
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        if(regressor instanceof Parameterized)
            return ((Parameterized)regressor).getParameter(paramName);
        else
            return null;
    }
    
}
