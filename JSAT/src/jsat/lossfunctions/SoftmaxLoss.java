package jsat.lossfunctions;

import jsat.classifiers.CategoricalResults;
import jsat.linear.Vec;
import jsat.math.MathTricks;

/**
 * The Softmax loss function is a multi-class generalization of the 
 * {@link LogisticLoss Logistic loss}. 
 * 
 * @author Edward Raff
 */
public class SoftmaxLoss extends LogisticLoss implements LossMC
{

	private static final long serialVersionUID = 3936898932252996024L;

	@Override
    public double getLoss(Vec processed, int y)
    {
        return -Math.log(processed.get(y));
    }

    @Override
    public void process(Vec pred, Vec processed)
    {
        if(pred != processed)
            pred.copyTo(processed);
        MathTricks.softmax(processed, false);
    }

    @Override
    public void deriv(Vec processed, Vec derivs, int y)
    {
        for(int i = 0; i < processed.length(); i++)
            if(i == y)
                derivs.set(i, processed.get(i)-1);//-(1-p)
            else
                derivs.set(i, processed.get(i));//-(0-p)
    }

    @Override
    public CategoricalResults getClassification(Vec processed)
    {
        return new CategoricalResults(processed.arrayCopy());
    }
}
