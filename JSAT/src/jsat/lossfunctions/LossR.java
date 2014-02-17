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
    
    /**
     * Given the score value of a data point, this returns the correct numeric 
     * result. For most regression problems this simply returns the score value.
     * 
     * @param score the score for a data point
     * @return the correct numeric regression value for this loss function
     */
    public double getRegression(double score);
    
    @Override
    public LossR clone();
}
