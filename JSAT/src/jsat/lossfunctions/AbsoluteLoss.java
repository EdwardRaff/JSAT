package jsat.lossfunctions;

/**
 * The AbsoluteLoss loss function for regression <i>L(x, y) = |x-y|</i>. 
 * <br>
 * This function is only one differentiable.
 *
 * @author Edward Raff
 */
public class AbsoluteLoss implements LossR
{


	private static final long serialVersionUID = -3398199227407867808L;

	/**
     * Computes the absolute loss
     *
     * @param pred the predicted value
     * @param y the target value
     * @return the loss for the functions
     */
    public static double loss(double pred, double y)
    {
        return Math.abs(y - pred);
    }

    /**
     * Returns the derivative of the absolute loss
     *
     * @param pred the predicted value
     * @param y the target value
     * @return the derivative of the loss function
     */
    public static double deriv(double pred, double y)
    {
        return Math.signum(pred - y);
    }
    
    public static double regress(double score)
    {
        return score;
    }

    @Override
    public double getLoss(double pred, double y)
    {
        return loss(pred, y);
    }

    @Override
    public double getDeriv(double pred, double y)
    {
        return deriv(pred, y);
    }

    @Override
    public double getDeriv2(double pred, double y)
    {
        return 0;
    }

    @Override
    public double getDeriv2Max()
    {
        return 0;
    }

    @Override
    public AbsoluteLoss clone()
    {
        return this;
    }

    @Override
    public double getRegression(double score)
    {
        return score;
    }
}
