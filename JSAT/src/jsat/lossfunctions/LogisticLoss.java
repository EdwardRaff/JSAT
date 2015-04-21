package jsat.lossfunctions;

import static java.lang.Math.*;
import jsat.classifiers.CategoricalResults;

/**
 * The LogisticLoss loss function for classification <i>L(x, y) =
 * log(1+exp(-y*x))</i>. 
 * <br>
 * This function is twice differentiable.
 *
 * @author Edward Raff
 */
public class LogisticLoss implements LossC
{
    /*
     * NOTE: 30 used as a threshold b/c at the small values exp(-30) stradles 
     * the edge of numerical double precision
     */


	private static final long serialVersionUID = -3929171604513497068L;

	/**
     * Computes the logistic loss
     *
     * @param pred the predicted value
     * @param y the true value
     * @return the logistic loss
     */
    public static double loss(double pred, double y)
    {
        final double x = -y * pred;
        if (x >= 30)//as x -> inf, L(x) -> x. At 30 exp(x) is O(10^13), getting unstable. L(x)-x at this value is O(10^-14), also avoids exp and log ops
            return x;
        else if (x <= -30)
            return 0;
        return log(1 + exp(x));
    }

    /**
     * Computes the first derivative of the logistic loss
     *
     * @param pred the predicted value
     * @param y the true value
     * @return the first derivative of the logistic loss
     */
    public static double deriv(double pred, double y)
    {
        final double x = y * pred;
        if (x >= 30)
            return 0;
        else if (x <= -30)
            return y;

        return -y / (1 + exp(y * pred));
    }

    /**
     * Computes the second derivative of the logistic loss
     *
     * @param pred the predicted value
     * @param y the true value
     * @return the second derivative of the logistic loss
     */
    public static double deriv2(double pred, double y)
    {
        final double x = y * pred;
        if (x >= 30)
            return 0;
        else if (x <= -30)
            return 0;

        final double p = 1 / (1 + exp(y * pred));

        return p * (1 - p);
    }
    
    public static CategoricalResults classify(double score)
    {
        CategoricalResults cr = new CategoricalResults(2);
        final double p;
        if (score > 30)
            p = 1.0;
        else if (score < -30)
            p = 0.0;
        else
            p = 1 / (1 + Math.exp(-score));
        cr.setProb(0, 1 - p);
        cr.setProb(1, p);
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
        return deriv2(pred, y);
    }

    @Override
    public double getDeriv2Max()
    {
        return 1.0 / 4.0;
    }

    @Override
    public LogisticLoss clone()
    {
        return this;
    }

    @Override
    public CategoricalResults getClassification(double score)
    {
        return classify(score);
    }
}
