package jsat.classifiers.evaluation;

/**
 * Evaluates a classifier based on Mathews Correlation Coefficient
 * 
 * @author Edward Raff
 */
public class MatthewsCorrelationCoefficient extends SimpleBinaryClassMetric
{


	private static final long serialVersionUID = 7102318546460007008L;

	public MatthewsCorrelationCoefficient()
    {
        super();
    }
    
    public MatthewsCorrelationCoefficient(MatthewsCorrelationCoefficient toClone)
    {
        super(toClone);
    }

    @Override
    public double getScore()
    {
        double denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn);
        if(denom <= 1e-16)
            return 0;
        return (tp*tn-fp*fn)/Math.sqrt(denom);
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
    public MatthewsCorrelationCoefficient clone()
    {
        return new MatthewsCorrelationCoefficient(this);
    }

    @Override
    public String getName()
    {
        return "Matthews Correlation Coefficient";
    }
    
}
