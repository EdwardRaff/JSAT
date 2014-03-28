package jsat.regression.evaluation;

import jsat.math.OnLineStatistics;

/**
 * Uses the Mean of Absolute Errors between the predictions and the true values. 
 * 
 * @author Edward Raff
 */
public class MeanAbsoluteError implements RegressionScore
{
    private OnLineStatistics absError;

    public MeanAbsoluteError()
    {
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public MeanAbsoluteError(MeanAbsoluteError toCopy)
    {
        if(toCopy.absError != null)
            this.absError = toCopy.absError.clone();
    }
    
    @Override
    public void prepare()
    {
        absError = new OnLineStatistics();
    }
    
    @Override
    public void addResult(double prediction, double trueValue, double weight)
    {
        if(absError == null)
            throw new RuntimeException("regression score has not been initialized");
        absError.add(Math.abs(prediction-trueValue), weight);
    }

    @Override
    public void addResults(RegressionScore other)
    {
        MeanAbsoluteError otherObj = (MeanAbsoluteError) other;
        if(otherObj.absError != null)
            this.absError.add(otherObj.absError);
    }

    @Override
    public double getScore()
    {
        return absError.getMean();
    }

    @Override
    public boolean lowerIsBetter()
    {
        return true;
    }

    @Override
    public MeanAbsoluteError clone()
    {
        return new MeanAbsoluteError(this);
    }

    @Override
    public int hashCode()
    {
        return getName().hashCode();
    }
    
    @Override
    public boolean equals(Object obj)
    {
        if(this.getClass().isAssignableFrom(obj.getClass()) && obj.getClass().isAssignableFrom(this.getClass()))
        {
            return true;
        }
        return false;
    }
    
    @Override
    public String getName()
    {
        return "Mean Absolute Error";
    }
    
}
