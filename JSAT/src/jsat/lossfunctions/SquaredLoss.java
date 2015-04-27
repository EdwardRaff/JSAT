package jsat.lossfunctions;

/**
 * The SquaredLoss loss function for regression <i>L(x, y) = (x-y)<sup>2</sup></i>.
 * <br>
 * This function is twice differentiable.
 *
 * @author Edward Raff
 */
public class SquaredLoss implements LossR
{


	private static final long serialVersionUID = 130786305325167077L;

	/**
     * Computes the SquaredLoss loss
     *
     * @param pred the predicted value
     * @param y the true value
     * @return the squared loss
     */
    public static double loss(double pred, double y)
    {
        final double x = y - pred;
        return x * x * 0.5;
    }

    /**
     * Computes the first derivative of the squared loss
     *
     * @param pred the predicted value
     * @param y the true value
     * @return the first derivative of the squared loss
     */
    public static double deriv(double pred, double y)
    {
        return (pred - y);
    }

    /**
     * Computes the second derivative of the squared loss, which is always
     * {@code 1}
     *
     * @param pred the predicted value
     * @param y the true value
     * @return the second derivative of the squared loss
     */
    public static double deriv2(double pred, double y)
    {
        return 1;
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
        return deriv2(pred, y);
    }

    @Override
    public double getDeriv2Max()
    {
        return 1;
    }

    @Override
    public SquaredLoss clone()
    {
        return this;
    }

    @Override
    public double getRegression(double score)
    {
        return score;
    }
}
