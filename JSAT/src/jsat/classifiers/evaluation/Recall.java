package jsat.classifiers.evaluation;

/**
 * Evaluates a classifier based on the Recall rate, where the class of index 0 
 * is considered the positive class. This score is only valid for binary 
 * classification problems. 
 * 
 * @author Edward Raff
 */
public class Recall extends SimpleBinaryClassMetric
{


	private static final long serialVersionUID = 4832185425203972017L;

	/**
     * Creates a new Recall evaluator
     */
    public Recall()
    {
        super();
    }
    
    /**
     * Copy constructor
     * @param toClone the object to copy
     */
    public Recall(Recall toClone)
    {
        super(toClone);
    }

    @Override
    public double getScore()
    {
        return tp/(tp+fn);
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
    public Recall clone()
    {
        return new Recall(this);
    }

    @Override
    public String getName()
    {
        return "Recall";
    }
    
}
