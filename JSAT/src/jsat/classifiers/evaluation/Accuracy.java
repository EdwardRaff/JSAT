package jsat.classifiers.evaluation;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;

/**
 * Evaluates a classifier based on its accuracy in predicting the correct class. 
 * 
 * @author Edward Raff
 */
public class Accuracy implements ClassificationScore
{

	private static final long serialVersionUID = 397690693205481128L;
	private double correct, total;

    public Accuracy()
    {
    }

    public Accuracy(Accuracy toClone)
    {
        this.correct = toClone.correct;
        this.total = toClone.total;
    }
    
    @Override
    public void addResult(CategoricalResults prediction, int trueLabel, double weight)
    {
        if(prediction.mostLikely() == trueLabel)
            correct += weight;
        total += weight;
    }

    @Override
    public void addResults(ClassificationScore other)
    {
        Accuracy otherObj = (Accuracy) other;
        this.correct += otherObj.correct;
        this.total += otherObj.total;
    }
    
    @Override
    public void prepare(CategoricalData toPredict)
    {
        correct = 0;
        total = 0;
    }

    @Override
    public double getScore()
    {
        return correct/total;
    }

    @Override
    public boolean lowerIsBetter()
    {
        return false;
    }

    @Override
    public boolean equals(Object obj)
    {
        if(obj instanceof Accuracy)
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
    public Accuracy clone()
    {
        return new Accuracy(this);
    }

    @Override
    public String getName()
    {
        return "Accuracy";
    }
    
}
