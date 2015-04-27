package jsat.classifiers.evaluation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;

/**
 * Computes the Area Under the ROC Curve as an evaluation of classification 
 * scores. The AUC takes <i>O(n log n)</i> time for <i>n</i> predictions and is 
 * only valid for binary classification problems. 
 * 
 * @author Edward Raff
 */
public class AUC implements ClassificationScore
{

	private static final long serialVersionUID = 6882234590870560718L;

	private static class Tuple implements Comparable<Tuple>
    {
        /**
         * larger means positive class, smaller means negative class
         */
        public double score;
        /**
         * Does this point truly belong to the positive class
         */
        public boolean positiveClass;
        
        public double weight;

        public Tuple(double score, boolean positiveClass, double weight)
        {
            this.score = score;
            this.positiveClass = positiveClass;
            this.weight = weight;
        }
        

        @Override
        public int compareTo(Tuple o)
        {
            return Double.compare(this.score, o.score);
        }
        
    }
    private List<Tuple> scores;

    /**
     * Creates a new AUC object
     */
    public AUC()
    {
    }

    /**
     * Copy constructor
     * @param toClone the object to copy
     */
    public AUC(AUC toClone)
    {
        if(toClone.scores != null)
        {
            this.scores = new ArrayList<Tuple>(toClone.scores);
            for(int i = 0; i < this.scores.size(); i++)
                this.scores.set(i, new Tuple(this.scores.get(i).score, this.scores.get(i).positiveClass, this.scores.get(i).weight));
        }
    }
    
    @Override
    public void addResult(CategoricalResults prediction, int trueLabel, double weight)
    {
        scores.add(new Tuple(prediction.getProb(0), trueLabel == 0, weight));
    }

    @Override
    public void addResults(ClassificationScore other)
    {
        AUC otherObj = (AUC) other;
        this.scores.addAll(otherObj.scores);
    }
    
    @Override
    public void prepare(CategoricalData toPredict)
    {
        if(toPredict.getNumOfCategories() != 2)
            throw new IllegalArgumentException("AUC is only defined for binary classification problems");
        scores = new ArrayList<Tuple>();
    }

    @Override
    public double getScore()
    {
        Collections.sort(scores);

        double pos = 0, neg = 0, sum = 0;
        for (Tuple i : scores)
            if (i.positiveClass)
                pos += i.weight;
            else
                neg += i.weight;
        double posLeft = pos;
        for (Tuple i : scores)
            if (i.positiveClass)//oh no, saw the wrong thing
                posLeft -= i.weight;
            else//posLeft instances of the positive class were correctly above the negative class
                sum += posLeft;

        return sum / (double) (pos * neg);
    }

    @Override
    public boolean lowerIsBetter()
    {
        return false;
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
    public AUC clone()
    {
        return new AUC(this);
    }

    @Override
    public String getName()
    {
        return "AUC";
    }
    
}
