
package jsat.classifiers;

import java.util.concurrent.ExecutorService;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.Vec;
import jsat.regression.LogisticRegression;
import jsat.regression.RegressionDataSet;
import jsat.utils.FakeExecutor;

/**
 * Multinomial Logistic Regression is an extension of {@link LogisticRegression} for classification when
 * there are more then two target classes. The results of this extension can differ greatly from applying
 * {@link OneVSAll} to Logistic Regression. 
 * 
 * @author Edward Raff
 */
public class MultinomialLogisticRegression implements Classifier
{

	private static final long serialVersionUID = -9168502043850569017L;
	private Vec[] classCoefficents;
    public CategoricalResults classify(DataPoint data)
    {
        if(classCoefficents == null)
            throw new UntrainedModelException("Model has not yet been trained");
        /**
         * The probabilities for the MLR for a class k != 0 are
         * 
         *                  exp/X  B \
         *                     \ i  k/
         * P/y  = k\ = --------------------
         *  \ i    /         K
         *                 =====
         *                 \
         *             1 +  >    exp/X  B \
         *                 /        \ i  j/
         *                 =====
         *                 j = 1
         * 
         * and for class = 0
         * 
         * 
         *                       1
         * P/y  = 0\ = --------------------
         *  \ i    /         K
         *                 =====
         *                 \
         *             1 +  >    exp/X  B \
         *                 /        \ i  j/
         *                 =====
         *                 j = 1
         */
        CategoricalResults results = new CategoricalResults(classCoefficents.length+1);
        double sum = 0.0;
        results.setProb(0, 1.0);
        Vec b = data.getNumericalValues();
        for(int i = 0; i < classCoefficents.length; i++)
        {
            Vec coefs = classCoefficents[i];
            double exp = coefs.get(0);
            for(int j = 1; j < coefs.length(); j++)
                exp += b.get(j-1)*coefs.get(j);
            exp = Math.exp(exp);
            sum += exp;
            results.setProb(i+1, exp);
        }
        
        results.divideConst(1.0+sum);
        
        return results;
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        LogisticRegression logit = new LogisticRegression();
        
        classCoefficents = new Vec[dataSet.getClassSize()-1];
        
        
        for(int k = 1; k < dataSet.getClassSize(); k++)
        {
            RegressionDataSet rds = new RegressionDataSet(dataSet.getNumNumericalVars(), dataSet.getCategories());
            for(int i = 0; i < dataSet.getSampleSize(); i++)
                rds.addDataPoint(dataSet.getDataPoint(i), (dataSet.getDataPointCategory(i) == k ? 1.0 : 0.0 ) );

            logit.train(rds, threadPool);
            classCoefficents[k-1] = logit.getCoefficents();
        }
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public MultinomialLogisticRegression clone()
    {
        MultinomialLogisticRegression clone = new MultinomialLogisticRegression();
        if(this.classCoefficents != null)
        {
            clone.classCoefficents = new Vec[this.classCoefficents.length];
            for(int i = 0; i < this.classCoefficents.length; i++)
                clone.classCoefficents[i] = this.classCoefficents[i].clone();
        }
        
        return clone;
    }
    
}
