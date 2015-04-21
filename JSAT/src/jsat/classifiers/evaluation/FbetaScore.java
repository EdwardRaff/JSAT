package jsat.classifiers.evaluation;

/**
 * The F<sub>&beta;</sub> score is the generalization of {@link F1Score}, where 
 * &beta; indicates the level of preference for precision over recall.  
 * This score is only valid for binary classification problems. 
 * 
 * @author Edward Raff
 */
public class FbetaScore extends SimpleBinaryClassMetric
{

	private static final long serialVersionUID = -7530404462591303694L;
	private double beta;
    
    /**
     * Creates a new F<sub>&beta;</sub> score
     * @param beta the weight to apply to precision over recall, must be in (0, 
     * &infin;)
     */
    public FbetaScore(double beta)
    {
        super();
        if(beta <= 0 || Double.isInfinite(beta) || Double.isNaN(beta))
            throw new IllegalArgumentException("beta must be in (0, inf), not " + beta);
        this.beta = beta;
    }
    
    /**
     * Copy constructor
     * @param toClone the object to copy
     */
    public FbetaScore(FbetaScore toClone)
    {
        super(toClone);
        this.beta = toClone.beta;
    }

    @Override
    public double getScore()
    {
        final double betaSqrd = beta*beta;
        return (1+betaSqrd)*tp/((1+betaSqrd)*tp+fp+betaSqrd*fn);
    }
    
    @Override
    public boolean equals(Object obj)
    {
        if(this.getClass().isAssignableFrom(obj.getClass()) && obj.getClass().isAssignableFrom(this.getClass()))
        {
            return this.beta == ((FbetaScore)obj).beta;
        }
        return false;
    }

    @Override
    public int hashCode()
    {
        return new Double(beta).hashCode();
    }

    @Override
    public FbetaScore clone()
    {
        return new FbetaScore(this);
    }

    @Override
    public String getName()
    {
        return "F beta(" + beta + ") Score";
    }
    
}
