package jsat.classifiers.evaluation;

import java.util.Arrays;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;

/**
 * This computes the multi-class Log Loss<br>
 * - 1/N <big>&Sigma;</big><sub>&forall; i &isin; N</sub> log(p<sub>i, y</sub>)
 * <br>
 * <br>
 * Where <i>N</i> is the number of data points and <i>p<sub>i, y</sub></i> is 
 * the estimated probability of the true class label. The lower the loss score, 
 * the better. 
 * <br><br>
 * When <i>p<sub>i, y</sub></i> = 0 the log loss is uninformatively forced to 
 * &infin;, even if all other data points are perfectly correct. To avoid this a 
 * small nudge factor is added. 
 * 
 * @author Edward Raff
 */
public class LogLoss implements ClassificationScore
{

	private static final long serialVersionUID = 3123851772991293430L;
	private double loss;
    private double weightSum;
    private double nudge;

    /**
     * Creates a new Log Loss evaluation score
     */
    public LogLoss()
    {
        this(1e-15);
    }

    /**
     * Creates a new Log Loss evaluation score
     * @param nudge the nudge value to avoid zero probabilities, must be non 
     * negative and less than 0.1
     */
    public LogLoss(double nudge)
    {
        if(nudge < 0 || nudge >= 0.1)
            throw new IllegalArgumentException("nudge must be a small non-negative value in [0, 0.1) not " + nudge);
        this.nudge = nudge;
    }

    public LogLoss(LogLoss toClone)
    {
        this.loss = toClone.loss;
        this.weightSum = toClone.weightSum;
        this.nudge = toClone.nudge;
    }
    
    @Override
    public void addResult(CategoricalResults prediction, int trueLabel, double weight)
    {
        loss += weight * Math.log(Math.max(prediction.getProb(trueLabel), nudge));
        weightSum += weight;
    }

    @Override
    public void addResults(ClassificationScore other)
    {
        LogLoss otherObj = (LogLoss) other;
        this.loss += otherObj.loss;
        this.weightSum += otherObj.weightSum;
    }
    
    @Override
    public void prepare(CategoricalData toPredict)
    {
        loss = 0;
        weightSum = 0;
    }

    @Override
    public double getScore()
    {
        return -loss/weightSum;
    }
    
    @Override
    public boolean equals(Object obj)
    {
        if(this.getClass().isAssignableFrom(obj.getClass()) && obj.getClass().isAssignableFrom(this.getClass()))
        {
            return ((LogLoss)obj).nudge == this.nudge;
        }
        return false;
    }

    @Override
    public int hashCode()
    {
        return Arrays.hashCode(new double[]{nudge});
    }

    @Override
    public boolean lowerIsBetter()
    {
        return true;
    }

    @Override
    public LogLoss clone()
    {
        return new LogLoss(this);
    }

    @Override
    public String getName()
    {
        return "Log Loss";
    }
    
}
