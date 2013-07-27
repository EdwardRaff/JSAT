package jsat.lossfunctions;

import jsat.classifiers.CategoricalResults;

/**
 * The HingeLoss loss function for classification <i>L(x, y) = max(0, 1-y*x)</i>.
 * <br>
 * This function is only once differentiable.
 *
 * @author Edward Raff
 */
public class HingeLoss implements LossC
{

    /**
     * Computes the HingeLoss loss
     *
     * @param pred the predicted value
     * @param y the target value
     * @return the HingeLoss loss
     */
    public static double loss(double pred, double y)
    {
        return Math.max(0, 1 - y * pred);
    }

    /**
     * Computes the first derivative of the HingeLoss loss
     *
     * @param pred the predicted value
     * @param y the target value
     * @return the first derivative of the HingeLoss loss
     */
    public static double deriv(double pred, double y)
    {
        if (pred * y > 1)
            return 0;
        else
            return -y;
    }
    
    public static CategoricalResults classify(double score)
    {
        CategoricalResults cr = new CategoricalResults(2);
        if(score > 0)
            cr.setProb(1, 1.0);
        else
            cr.setProb(0, 1.0);
        return cr;
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
    public HingeLoss clone()
    {
        return this;
    }

    @Override
    public CategoricalResults getClassification(double score)
    {
        return classify(score);
    }
}
