package jsat.regression.evaluation;

import jsat.math.OnLineStatistics;

/**
 * Uses the Mean of the Squared Errors between the predictions and the true 
 * values.
 * 
 * @author Edward Raff
 */
public class MeanSquaredError implements RegressionScore
{

	private static final long serialVersionUID = 3655567184376550126L;
	private OnLineStatistics meanError;
    private boolean rmse;

    public MeanSquaredError()
    {
        this(false);
    }

    public MeanSquaredError(boolean rmse)
    {
        setRMSE(rmse);
    }

    public void setRMSE(boolean rmse)
    {
        this.rmse = rmse;
    }

    public boolean isRMSE()
    {
        return rmse;
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public MeanSquaredError(MeanSquaredError toCopy)
    {
        if(toCopy.meanError != null)
            this.meanError = toCopy.meanError.clone();
        this.rmse = toCopy.rmse;
    }
    
    @Override
    public void prepare()
    {
        meanError = new OnLineStatistics();
    }
    
    @Override
    public void addResult(double prediction, double trueValue, double weight)
    {
        if(meanError == null)
            throw new RuntimeException("regression score has not been initialized");
        meanError.add(Math.pow(prediction-trueValue, 2), weight);
    }

    @Override
    public void addResults(RegressionScore other)
    {
        MeanSquaredError otherObj = (MeanSquaredError) other;
        if(otherObj.meanError != null)
            this.meanError.add(otherObj.meanError);
    }

    @Override
    public double getScore()
    {
        if(rmse)
            return Math.sqrt(meanError.getMean());
        else
            return meanError.getMean();
    }

    @Override
    public boolean lowerIsBetter()
    {
        return true;
    }
    
    @Override
    public int hashCode()
    {//XXX this is a strange hashcode method
        return getName().hashCode();
    }
    
    @Override
    public boolean equals(Object obj)
    {//XXX check for equality of fields and obj == null
        if(this.getClass().isAssignableFrom(obj.getClass()) && obj.getClass().isAssignableFrom(this.getClass()))
        {
            return this.rmse == ((MeanSquaredError)obj).rmse;
        }
        return false;
    }

    @Override
    public MeanSquaredError clone()
    {
        return new MeanSquaredError(this);
    }

    @Override
    public String getName()
    {
        String prefix = rmse ? "Root " : "";
        return prefix + "Mean Squared Error";
    }
    
}
