package jsat.classifiers.evaluation;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;

/**
 * This is a base class for scores that can be computed from simple counts of 
 * the true positives, true negatives, false positives, and false negatives. The
 * class with index zero will be considered the positive class and the class 
 * with the first index will be the negative class. <br>
 * <br>
 * By default this class assumes higher scores are better
 * 
 * @author Edward Raff
 */
public abstract class SimpleBinaryClassMetric implements ClassificationScore
{

	private static final long serialVersionUID = -84479984342547212L;
	/**
     * true positives
     */
    protected double tp;
    /**
     * true negatives
     */
    protected double tn;
    /**
     * false positives
     */
    protected double fp;
    /**
     * false negatives
     */
    protected double fn;

    public SimpleBinaryClassMetric()
    {
    }

    public SimpleBinaryClassMetric(SimpleBinaryClassMetric toClone)
    {
        this.tp = toClone.tp;
        this.tn = toClone.tn;
        this.fp = toClone.fp;
        this.fn = toClone.fn;
    }
    
    @Override
    public void addResult(CategoricalResults prediction, int trueLabel, double weight)
    {
        int pred = prediction.mostLikely();
        if(pred == trueLabel)
            if(pred == 0)
                tp += weight;
            else
                tn += weight;
        else
        {
            if(pred == 0)
                fp += weight;
            else
                fn += weight;
        }
        
    }

    @Override
    public void prepare(CategoricalData toPredict)
    {
        tp = tn = fp = fn = 0;
    }

    @Override
    public void addResults(ClassificationScore other)
    {
        SimpleBinaryClassMetric otherObj = (SimpleBinaryClassMetric) other;
        this.tp += otherObj.tp;
        this.tn += otherObj.tn;
        this.fp += otherObj.fp;
        this.fn += otherObj.fn;
    }

    @Override
    abstract public double getScore();

    @Override
    public boolean lowerIsBetter()
    {
        return false;
    }

    @Override
    abstract public SimpleBinaryClassMetric clone();
    
}
