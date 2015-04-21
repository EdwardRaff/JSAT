package jsat.classifiers.evaluation;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;

/**
 * Evaluates a classifier based on the Kappa statistic. 
 * 
 * @author Edward Raff
 */
public class Kappa implements ClassificationScore
{

	private static final long serialVersionUID = -1684937057234736715L;
	private Matrix errorMatrix;

    public Kappa()
    {
    }

    public Kappa(Kappa toClone)
    {
        if(toClone.errorMatrix != null)
            this.errorMatrix = toClone.errorMatrix.clone();
    }
    
    @Override
    public void addResult(CategoricalResults prediction, int trueLabel, double weight)
    {
        errorMatrix.increment(prediction.mostLikely(), trueLabel, weight);
    }

    @Override
    public void addResults(ClassificationScore other)
    {
        Kappa otherObj = (Kappa) other;
        if(otherObj.errorMatrix == null)
            return;
        if(this.errorMatrix == null)
            throw new RuntimeException("KappaScore has not been prepared");
        this.errorMatrix.mutableAdd(otherObj.errorMatrix);
    }

    @Override
    public void prepare(CategoricalData toPredict)
    {
        int N = toPredict.getNumOfCategories();
        errorMatrix = new DenseMatrix(N, N);
    }

    @Override
    public double getScore()
    {
        double[] rowTotals = new double[errorMatrix.rows()];
        double[] colTotals = new double[errorMatrix.rows()];
        for(int i = 0; i < errorMatrix.rows(); i++)
        {
            rowTotals[i] = errorMatrix.getRowView(i).sum();
            colTotals[i] = errorMatrix.getColumnView(i).sum();
        }
        
        double chanceAgreement = 0;
        double accuracy = 0;
        double totalCount = 0;
        for(int i = 0; i < rowTotals.length; i++)
        {
            chanceAgreement += rowTotals[i]*colTotals[i];
            totalCount += rowTotals[i];
            accuracy += errorMatrix.get(i, i);
        }
        chanceAgreement /= totalCount*totalCount;
        accuracy /= totalCount;
        
        return (accuracy-chanceAgreement)/(1-chanceAgreement);        
    }
    
    @Override
    public boolean equals(Object obj)
    {
        if(this.getClass().isAssignableFrom(obj.getClass()) && obj.getClass().isAssignableFrom(this.getClass()))
        {
            return true;
        }
        return false;
    }

    @Override
    public int hashCode()
    {
        return getName().hashCode();
    }

    @Override
    public boolean lowerIsBetter()
    {
        return false;
    }

    @Override
    public Kappa clone()
    {
        return new Kappa(this);
    }

    @Override
    public String getName()
    {
        return "Kappa";
    }
    
}
