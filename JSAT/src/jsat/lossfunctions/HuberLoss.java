package jsat.lossfunctions;

/**
 * The HuberLoss loss function for regression. The HuberLoss loss switches between
 * {@link SquaredLoss} and {@link AbsoluteLoss} loss based on a threshold value.
 * <br>
 * This function is only partially twice differentiable.
 *
 * @author Edward Raff
 */
public class HuberLoss implements LossR
{


	private static final long serialVersionUID = -4463269746356262940L;
	private double c;

    /**
     * Creates a new HuberLoss loss
     *
     * @param c the threshold to switch between the squared and logistic loss at
     */
    public HuberLoss(double c)
    {
        this.c = c;
    }

    /**
     * Creates a new HuberLoss loss thresholded at 1
     */
    public HuberLoss()
    {
        this(1);
    }

    /**
     * Computes the HuberLoss loss
     *
     * @param pred the predicted value
     * @param y the true value
     * @param c the threshold value
     * @return the HuberLoss loss
     */
    public static double loss(double pred, double y, double c)
    {
        final double x = y - pred;
        if (Math.abs(x) <= c)
            return x * x * 0.5;
        else
            return c * (Math.abs(x) - c / 2);
    }

    /**
     * Computes the first derivative of the HuberLoss loss
     *
     * @param pred the predicted value
     * @param y the true value
     * @param c the threshold value
     * @return the first derivative of the HuberLoss loss
     */
    public static double deriv(double pred, double y, double c)
    {
        double x = pred-y;

        if (Math.abs(x) <= c)
            return x;
        else
            return c * Math.signum(x);
    }

    /**
     * Computes the second derivative of the HuberLoss loss, which only exists for
     * values &lt; {@code c}
     *
     * @param pred the predicted value
     * @param y the target value
     * @param c the threshold value
     * @return the second derivative of the HuberLoss loss
     */
    public static double deriv2(double pred, double y, double c)
    {
        if (Math.abs(pred-y) < c)
            return 1;
        else
            return 0;
    }

    public static double regress(double score)
    {
        return score;
    }
    
    @Override
    public double getLoss(double pred, double y)
    {
        return loss(pred, y, c);
    }

    @Override
    public double getDeriv(double pred, double y)
    {
        return deriv(pred, y, c);
    }

    @Override
    public double getDeriv2(double pred, double y)
    {
        return deriv2(pred, y, c);
    }

    @Override
    public double getDeriv2Max()
    {
        return 1;
    }

    @Override
    public HuberLoss clone()
    {
        return new HuberLoss(c);
    }

    @Override
    public double getRegression(double score)
    {
        return score;
    }
}
