
package jsat.classifiers.calibration;

import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import static jsat.math.FastMath.*;

/**
 * Platt Calibration essentially performs logistic regression on the output 
 * scores of a model against their class labels. While first described for SVMs, 
 * Platt's method can be used for any scoring algorithm in general. 
 * <br><br>
 * See:<br>
 * <ul>
 * <li>Platt, J. C. (1999). <i>Probabilistic Outputs for Support Vector
 * Machines and Comparisons to Regularized Likelihood Methods</i>. Advances in 
 * Large Margin Classifiers (pp. 61–74). MIT Press. Retrieved from 
 * <a href="http://www.tu-harburg.de/ti6/lehre/seminarCI/slides/ws0506/SVMprob.pdf">
 * here </a></li>
 * <li>Lin, H.-T., Lin, C.-J.,&amp;Weng, R. C. (2007). <i>A note on Platt’s 
 * probabilistic outputs for support vector machines</i>. Machine learning, 
 * 68(3), 267–276. Retrieved from 
 * <a href="http://www.springerlink.com/index/8417V9235M561471.pdf">here</a></li>
 * <li>Niculescu-Mizil, A.,&amp;Caruana, R. (2005). <i>Predicting Good 
 * Probabilities with Supervised Learning</i>. International Conference on 
 * Machine Learning (pp. 625–632). Retrieved from 
 * <a href="http://dl.acm.org/citation.cfm?id=1102430">here</a></li>
 * </ul>
 * @author Edward Raff
 */
public class PlattCalibration extends BinaryCalibration
{

	private static final long serialVersionUID = 1099230240231262536L;
	private double A,B;
    private double maxIter = 100;
    private double minStep = 1e-10;
    private double sigma = 1e-12;
    

    /**
     * Creates a new Platt Calibration object
     * @param base the base model to calibrate the outputs of
     * @param mode the calibration mode to use
     */
    public PlattCalibration(BinaryScoreClassifier base, CalibrationMode mode)
    {
        super(base, mode);
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(2);
        double p_1 = 1/(1+exp(A*base.getScore(data)+B));
        cr.setProb(0, 1-p_1);
        cr.setProb(1, p_1);
        return cr;
    }
    
    @Override
    protected void calibrate(boolean[] label, double[] deci, final int len)
    {
        int prior1 = 0;//number of positive examples
        for (boolean positive : label)
            if (positive)
                prior1++;
        final int prior0 = label.length - prior1;//number of negative examples

        final double hiTarget = (prior1 + 1.0) / (prior1 + 2.0);
        final double loTarget = 1 / (prior0 + 2.0);

        double[] t = new double[len];
        for (int i = 0; i < len; i++)
            if (label[i])
                t[i] = hiTarget;
            else
                t[i] = loTarget;

        A = 0.0;
        B = log((prior0 + 1.0) / (prior1 + 1.0));
        double fval = 0.0;
        for(int i = 0; i < len; i++)
        {
            double fApB=deci[i]*A+B;
            if(fApB >= 0)
                fval += t[i]*fApB+log(1+exp(-fApB));
            else
                fval += (t[i]-1)*fApB+log(1+exp(-fApB));
        }

        for (int it = 0; it < maxIter; it++)
        {
            //Update Gradient and Hessian (use H’ = H + sigma I)
            double h11 = sigma, h22 = sigma;
            double h21 = 0, g1 = 0, g2 = 0.0;
            
            for(int i = 0; i < len; i++)
            {
                double fApB = deci[i] * A + B, p, q;
                if (fApB >= 0)
                {
                    p = exp(-fApB) / (1.0 + exp(-fApB));
                    q = 1.0 / (1.0 + exp(-fApB));
                }
                else
                {
                    p = 1.0 / (1.0 + exp(fApB));
                    q = exp(fApB) / (1.0 + exp(fApB));
                }
                
                double d2 = p * q;
                h11 += deci[i] * deci[i] * d2;
                h22 += d2;
                h21 += deci[i] * d2;
                double d1 = t[i] - p;
                g1 += deci[i] * d1;
                g2 += d1;
            }
            
            if (Math.abs(g1)<1e-5 && Math.abs(g2)<1e-5) //Stopping criteria
                break;
            //Compute modified Newton directions
            double det = h11 * h22 - h21 * h21;
            double dA = -(h22 * g1 - h21 * g2) / det;
            double dB = -(-h21 * g1 + h11 * g2) / det;
            double gd = g1 * dA + g2 * dB;
            double stepsize = 1;
            
            while (stepsize >= minStep)//Line search
            {
                double newA = A + stepsize * dA, newB = B + stepsize * dB, newf = 0.0;
                for (int i = 0; i < len; i++)
                {
                    double fApB = deci[i] * newA + newB;
                    if (fApB >= 0)
                        newf += t[i] * fApB + log(1 + exp(-fApB));
                    else
                        newf += (t[i] - 1) * fApB + log(1 + exp(fApB));
                }
                
                if (newf < fval + 0.0001 * stepsize * gd)
                {
                    A = newA;
                    B = newB;
                    fval = newf;
                    break; //Sufficient decrease satisfied
                }
                else
                    stepsize /= 2.0;
            }
            
            if (stepsize < minStep)
                break;
        }
    }
    
    @Override
    public boolean supportsWeightedData()
    {
        return base.supportsWeightedData();
    }

    @Override
    public PlattCalibration clone()
    {
        PlattCalibration clone = new PlattCalibration(base.clone(), this.mode);
        
        clone.A = this.A;
        clone.B = this.B;
        
        clone.folds = this.folds;
        clone.holdOut = this.holdOut;
        clone.sigma = this.sigma;
        clone.minStep = this.minStep;
        clone.maxIter = this.maxIter;
        
        return clone;
    }

}
