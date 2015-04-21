package jsat.regression.evaluation;

/**
 * Uses the Sum of Absolute Errors divided by the sum of the absolute value of 
 * the true values subtracted from their mean. This produces an error metric 
 * that has no units. 
 * 
 * @author Edward Raff
 */
public class RelativeAbsoluteError extends TotalHistoryRegressionScore
{

	private static final long serialVersionUID = -6152988968756871647L;

	/**
     * Creates a new Relative Absolute Error evaluator
     */
    public RelativeAbsoluteError()
    {
        super();
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public RelativeAbsoluteError(RelativeAbsoluteError toCopy)
    {
        super(toCopy);
    }
    
    @Override
    public double getScore()
    {
        double trueMean = truths.getVecView().mean();
        double numer = 0, denom = 0;
        for(int i = 0; i < truths.size(); i++)
        {
            numer += Math.abs(predictions.getD(i)-truths.getD(i));
            denom += Math.abs(trueMean-truths.getD(i));
        }
        return numer/denom;
    }

    @Override
    public boolean lowerIsBetter()
    {
        return true;
    }

    @Override
    public RelativeAbsoluteError clone()
    {
        return new RelativeAbsoluteError(this);
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
            return true;
        }
        return false;
    }
    
    @Override
    public String getName()
    {
        return "Relative Absolute Error";
    }
    
}
