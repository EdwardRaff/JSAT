package jsat.lossfunctions;

/**
 * Specifies a getLoss function for regression problems.
 *
 * @author Edward Raff
 */
public interface LossR extends LossFunc
{

    /**
     * Computes the getLoss for a regression problem.
     *
     * @param pred the predicted value in (-Infinity, Infinity)
     * @param y the true target value in (-Infinity, Infinity)
     * @return the getLoss in [0, Inf)
     */
    @Override
    public double getLoss(double pred, double y);

    /**
     * Computes the first derivative of the getLoss function.
     *
     * @param pred the predicted value in (-Infinity, Infinity)
     * @param y the true target value in (-Infinity, Infinity)
     * @return the first derivative of the getLoss
     */
    @Override
    public double getDeriv(double pred, double y);

    /**
     * Computes the second derivative of the getLoss function.
     *
     * @param pred the predicted value in (-Infinity, Infinity)
     * @param y the true target value in (-Infinity, Infinity)
     * @return the second derivative of the getLoss function
     */
    @Override
    public double getDeriv2(double pred, double y);
}
