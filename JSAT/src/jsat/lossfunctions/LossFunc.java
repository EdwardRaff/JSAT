package jsat.lossfunctions;

import java.io.Serializable;

/**
 * Provides a generic interface for some loss function on some problem that can
 * be described with a single real prediction value and a single real expected
 * value.
 * <br><br>
 * A loss function must be non-negative and should be convex.
 *
 * @author Edward Raff
 */
public interface LossFunc extends Serializable
{

    /**
     * Computes the loss for some problem.
     *
     * @param pred the predicted value in (-Infinity, Infinity)
     * @param y the true value in (-Infinity, Infinity)
     * @return the loss in [0, Inf)
     */
    public double getLoss(double pred, double y);

    /**
     * Computes the first derivative of the loss function.
     *
     * @param pred the predicted score in (-Infinity, Infinity)
     * @param y the true value in (-Infinity, Infinity)
     * @return the first derivative of the getLoss
     */
    public double getDeriv(double pred, double y);

    /**
     * Computes the second derivative of the getLoss function.
     *
     * @param pred the predicted score in (-Infinity, Infinity)
     * @param y the true value in (-Infinity, Infinity)
     * @return the second derivative of the getLoss function
     */
    public double getDeriv2(double pred, double y);

    /**
     * Returns an upper bound on the maximum value of the second derivative. If
     * the second derivative does not exist, {@link Double#NaN} is a valid
     * result. It is also possible for {@code 0} and
     * {@link Double#POSITIVE_INFINITY} to be valid results, and must be checked
     * for.
     *
     * @return the max value of {@link #getDeriv2(double, double) }
     */
    public double getDeriv2Max();
    
    public LossFunc clone();
}
