package jsat.regression.evaluation;

import jsat.utils.DoubleList;

/**
 * This abstract class provides the work for maintaining the history of 
 * predictions and their true values. 
 * 
 * @author Edward Raff
 */
public abstract class TotalHistoryRegressionScore implements RegressionScore
{

	private static final long serialVersionUID = -5262934560490160236L;
	/**
     * List of the true target values
     */
    protected DoubleList truths;
    /**
     * List of the predict values for each target
     */
    protected DoubleList predictions;
    /**
     * The weight of importance for each point
     */
    protected DoubleList weights;

    public TotalHistoryRegressionScore()
    {
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public TotalHistoryRegressionScore(TotalHistoryRegressionScore toCopy)
    {
        if(toCopy.truths != null)
        {
            this.truths = new DoubleList(toCopy.truths);
            this.predictions = new DoubleList(toCopy.predictions);
            this.weights = new DoubleList(toCopy.weights);
        }
    }
    
    @Override
    public void prepare()
    {
        truths = new DoubleList();
        predictions = new DoubleList();
        weights = new DoubleList();
    }

    @Override
    public void addResult(double prediction, double trueValue, double weight)
    {
        truths.add(trueValue);
        predictions.add(prediction);
        weights.add(weight);
    }

    @Override
    public void addResults(RegressionScore other)
    {
        TotalHistoryRegressionScore otherObj = (TotalHistoryRegressionScore) other;
        this.truths.addAll(otherObj.truths);
        this.predictions.addAll(otherObj.predictions);
        this.weights.addAll(otherObj.weights);
    }

    @Override
    public abstract TotalHistoryRegressionScore clone();
    
}
