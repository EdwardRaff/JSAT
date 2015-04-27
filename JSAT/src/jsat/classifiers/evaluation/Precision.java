package jsat.classifiers.evaluation;

/**
 * Evaluates a classifier based on the Precision, where the class of index 0 
 * is considered the positive class. This score is only valid for binary 
 * classification problems. 
 * 
 * @author Edward Raff
 */
public class Precision extends SimpleBinaryClassMetric
{


	private static final long serialVersionUID = 7046590252900909918L;

	public Precision()
    {
        super();
    }
    
    public Precision(Precision toClone)
    {
        super(toClone);
    }

    @Override
    public double getScore()
    {
        return tp/(tp+fp);
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
    public int hashCode()
    {
        return getName().hashCode();
    }

    @Override
    public Precision clone()
    {
        return new Precision(this);
    }

    @Override
    public String getName()
    {
        return "Precision";
    }
    
}
