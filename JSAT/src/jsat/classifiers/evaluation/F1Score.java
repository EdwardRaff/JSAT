package jsat.classifiers.evaluation;

/**
 * The F1 score is the harmonic mean of {@link Precision} and 
 * {@link Recall}. This score is only valid for binary 
 * classification problems. 
 * 
 * @author Edward Raff
 */
public class F1Score extends SimpleBinaryClassMetric
{


	private static final long serialVersionUID = -6192302685766444921L;

	public F1Score()
    {
        super();
    }
    
    public F1Score(F1Score toClone)
    {
        super(toClone);
    }

    @Override
    public double getScore()
    {
        return 2*tp/(2*tp+fp+fn);
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
    public F1Score clone()
    {
        return new F1Score(this);
    }

    @Override
    public String getName()
    {
        return "F1 Score";
    }
    
}
