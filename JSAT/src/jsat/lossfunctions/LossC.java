package jsat.lossfunctions;

import jsat.classifiers.CategoricalResults;

/**
 * Specifies a loss function for binary classification problems.
 *
 * @author Edward Raff
 */
public interface LossC extends LossFunc
{

    /**
     * Computes the getLoss for a classification problem.
     *
     * @param pred the predicted score in (-Infinity, Infinity)
     * @param y the true class label in {-1, 1}
     * @return the getLoss in [0, Inf)
     */
    @Override
    public double getLoss(double pred, double y);

    /**
     * Computes the first derivative of the getLoss function.
     *
     * @param pred the predicted score in (-Infinity, Infinity)
     * @param y the true class label in {-1, 1}
     * @return the first derivative of the getLoss
     */
    @Override
    public double getDeriv(double pred, double y);

    /**
     * Computes the second derivative of the getLoss function.
     *
     * @param pred the predicted score in (-Infinity, Infinity)
     * @param y the true class label in {-1, 1}
     * @return the second derivative of the getLoss function
     */
    @Override
    public double getDeriv2(double pred, double y);
    
    /**
     * Given the score value of a data point, this returns the classification 
     * results. 
     * 
     * @param score the score for a data point
     * @return the categorical results with the correct probability values for 
     * this loss function. 
     */
    public CategoricalResults getClassification(double score);
    
    /**
     * Computes the result of the conjugate function of this loss. This function
     * is generally optional, and should return {@link Double#NaN} if not
     * properly implemented. Some optimization algorithms do require a working
     * implementation though.
     * @param b the primary input to the function
     * @param pred the predicted score in (-Infinity, Infinity)
     * @param y the true class label in {-1, 1}
     * @return the result of the conjugate function of this loss
     */
    public double getConjugate(double b, double pred, double y);
    
    /**
     * If this loss is L-Lipschitz (1/L Lipschitz smooth), this method will return the value of L. If it is not L-Lipschitz, a value of 0 will be returned.  
     * @return the L-Lipschitz  constant, or 0 if this loss is not L-Lipschitz;
     */
    public double lipschitz();
    
    @Override
    public LossC clone();
}
