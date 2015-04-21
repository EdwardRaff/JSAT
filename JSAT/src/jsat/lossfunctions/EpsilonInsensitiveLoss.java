package jsat.lossfunctions;

/**
 * The &epsilon;-insensitive loss for regression <i>L(x, y) = 
 * max(0, |x-y|-&epsilon;)</i> is the common loss function used for Support 
 * Vector Regression. <br>
 * When &epsilon; = 0, the loss becomes equivalent to the {@link AbsoluteLoss}.
 *
 * @author Edward Raff
 */
public class EpsilonInsensitiveLoss implements LossR
{


	private static final long serialVersionUID = -8735274561429676350L;

	/**
     * Computes the &epsilon;-insensitive loss
     *
     * @param pred the predicted value
     * @param y the true value
     * @param eps the epsilon tolerance
     * @return the &epsilon;-insensitive loss
     */
    public static double loss(double pred, double y, double eps)
    {
        final double x = Math.abs(pred - y);
        return Math.max(0, x-eps);
    }

    /**
     * Computes the first derivative of the &epsilon;-insensitive loss
     *
     * @param pred the predicted value
     * @param y the true value
     * @param eps the epsilon tolerance
     * @return the first derivative of the &epsilon;-insensitive loss
     */
    public static double deriv(double pred, double y, double eps)
    {
        final double x = pred - y;
        if(eps < Math.abs(x))
            return Math.signum(x);
        else
            return 0;
    }
    
    private double eps;

    /**
     * Creates a new Epsilon Insensitive loss
     * @param eps the epsilon tolerance on error
     */
    public EpsilonInsensitiveLoss(double eps)
    {
        if(eps < 0 || Double.isNaN(eps) || Double.isInfinite(eps))
            throw new IllegalArgumentException("Epsilon must be non-negative, not " + eps);
        this.eps = eps;
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public EpsilonInsensitiveLoss(EpsilonInsensitiveLoss toCopy)
    {
        this.eps = toCopy.eps;
    }


    @Override
    public double getLoss(double pred, double y)
    {
        return loss(pred, y, eps);
    }

    @Override
    public double getDeriv(double pred, double y)
    {
        return deriv(pred, y, eps);
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
    public EpsilonInsensitiveLoss clone()
    {
        return new EpsilonInsensitiveLoss(this);
    }

    @Override
    public double getRegression(double score)
    {
        return score;
    }
}
