package jsat.regression.evaluation;

/**
 * Uses the Coefficient of Determination, also known as R<sup>2</sup>, is an 
 * evaluation score in [0,1]. 
 * 
 * @author Edward Raff
 */
public class CoefficientOfDetermination extends TotalHistoryRegressionScore
{

	private static final long serialVersionUID = 1215708502913888821L;

	/**
     * Creates a new Coefficient of Determination object
     */
    public CoefficientOfDetermination()
    {
        super();
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public CoefficientOfDetermination(CoefficientOfDetermination toCopy)
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
            numer += Math.pow(predictions.getD(i)-truths.getD(i), 2);
            denom += Math.pow(trueMean-truths.getD(i), 2);
        }
        return 1-numer/denom;
    }

    @Override
    public boolean lowerIsBetter()
    {
        return true;
    }

    @Override
    public CoefficientOfDetermination clone()
    {
        return new CoefficientOfDetermination(this);
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
        return "Coefficient of Determination";
    }
    
}
