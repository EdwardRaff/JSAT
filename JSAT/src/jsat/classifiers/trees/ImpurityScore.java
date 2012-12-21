package jsat.classifiers.trees;

import static java.lang.Math.*;
import java.util.Arrays;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;

/**
 * ImpurityScore provides a measure of the impurity of a set of data points 
 * respective to their class labels. The impurity score is maximized when the 
 * classes are evenly distributed, and minimized when all points belong to one 
 * class. <br>
 * The gain in purity can be computed using the static <i>gain</i> methods of
 * the class. However, not all impurity measures can be used for arbitrary data 
 * and splits. Some may only support binary splits, and some may only support 
 * binary target classes.
 * 
 * @author Edward Raff
 */
public class ImpurityScore implements Cloneable
{
    /**
     * Different methods of measuring the impurity in a set of data points 
     * based on nominal class labels
     */
    public enum ImpurityMeasure
    {
        INFORMATION_GAIN, 
        INFORMATION_GAIN_RATIO,
        GINI,
        CLASSIFICATION_ERROR
    }
    
    private double sumOfWeights;
    private double[] counts;
    private ImpurityMeasure impurityMeasure;

    /**
     * Creates a new impurity score that can be updated
     * 
     * @param classCount the number of target class values
     * @param impurityMeasure 
     */
    public ImpurityScore(int classCount, ImpurityMeasure impurityMeasure)
    {
        sumOfWeights = 0.0;
        counts = new double[classCount];
        this.impurityMeasure = impurityMeasure;
    }
    
    /**
     * Copy constructor
     * @param toClone 
     */
    private ImpurityScore(ImpurityScore toClone)
    {
        this.sumOfWeights = toClone.sumOfWeights;
        this.counts = Arrays.copyOf(toClone.counts, toClone.counts.length);
        this.impurityMeasure = toClone.impurityMeasure;
    }
    
    /**
     * Removes one point from the impurity score
     * @param dp the data point to add
     * @param targetClass the class of the point to add
     */
    public void removePoint(DataPoint dp, int targetClass)
    {
        removePoint(dp.getWeight(), targetClass);
    }
    
    /**
     * Removes one point from the impurity score
     * @param weight the weight of the point to add
     * @param targetClass the class of the point to add
     */
    public void removePoint(double weight, int targetClass)
    {
        counts[targetClass] -= weight;
        sumOfWeights -= weight;
    }

    /**
     * Adds one more point to the impurity score
     * @param dp the data point to add
     * @param targetClass the class of the point to add
     */
    public void addPoint(DataPoint dp, int targetClass)
    {
        addPoint(dp.getWeight(), targetClass);
    }
    
    /**
     * Adds one more point to the impurity score
     * @param weight the weight of the point to add
     * @param targetClass the class of the point to add
     */
    public void addPoint(double weight, int targetClass)
    {
        counts[targetClass] += weight;
        sumOfWeights += weight;
    }

    /**
     * Computes the current impurity score for the points that have been added.
     * A higher score is worse, a score of zero indicates a perfectly pure set 
     * of points (all one class). 
     * @return the impurity score
     */
    public double getScore()
    {
        double score = 0.0;

        if (impurityMeasure == ImpurityMeasure.INFORMATION_GAIN_RATIO
                || impurityMeasure == ImpurityMeasure.INFORMATION_GAIN)
        {
            for (Double count : counts)
            {
                double p = count / sumOfWeights;
                if (p > 0)
                    score += p * log(p) / log(2);
            }
        }
        else if (impurityMeasure == ImpurityMeasure.GINI)
        {
            score = 1;
            for (double count : counts)
            {
                double p = count / sumOfWeights;
                score -= p * p;
            }
        }
        else if (impurityMeasure == ImpurityMeasure.CLASSIFICATION_ERROR)
        {
            double maxClass = 0;
            for (double count : counts)
                maxClass = Math.max(maxClass, count / sumOfWeights);
            score = 1.0 - maxClass;
        }

        return abs(score);
    }

    /**
     * Returns the sum of the weights for all points currently in the impurity 
     * score
     * @return the sum of weights
     */
    public double getSumOfWeights()
    {
        return sumOfWeights;
    }
    
    /**
     * Returns the impurity measure being used 
     * @return the impurity measure being used 
     */
    public ImpurityMeasure getImpurityMeasure()
    {
        return impurityMeasure;
    }
    
    /**
     * Obtains the current categorical results by prior probability 
     * 
     * @return the categorical results for the current score
     */
    public CategoricalResults getResults()
    {
        CategoricalResults cr = new CategoricalResults(counts.length);
        for(int i = 0; i < counts.length; i++)
            cr.setProb(i, counts[i]/sumOfWeights);
        return cr;
    }
    
    /*
     * NOTE: for calulating the entropy in a split, if S is the current set of
     * all data points, and S_i denotes one of the subsets gained from splitting
     * The Gain for a split is
     *
     *                       n
     *                     ===== |S |
     *                     \     | i|
     * Gain = Entropy(S) -  >    ---- Entropy/S \
     *                     /      |S|        \ i/
     *                     =====
     *                     i = 1
     *
     *                   Gain
     * GainRatio = ----------------
     *             SplitInformation
     *
     *                        n
     *                      ===== |S |    /|S |\
     *                      \     | i|    || i||
     * SplitInformation = -  >    ---- log|----|
     *                      /      |S|    \ |S|/
     *                      =====
     *                      i = 1
     */
    
    /**
     * Computes the gain in score from a splitting of the data set
     * 
     * @param wholeData the score for the whole data set
     * @param splits the scores for each of the splits
     * @return the gain for the values given
     */
    public static double gain(ImpurityScore wholeData, ImpurityScore... splits)
    {
        double splitScore = 0.0;
        
        boolean useSplitInfo = splits[0].impurityMeasure == ImpurityMeasure.INFORMATION_GAIN_RATIO;
        
        double splitInfo = useSplitInfo ? 1.0 : 0.0;
        
        for(ImpurityScore split : splits)
        {
            double p = split.getSumOfWeights()/wholeData.getSumOfWeights();
            if(p <= 0)//log(0) is -Inft, so skip and treat as zero
                continue;
            splitScore += p*split.getScore();
            
            if(useSplitInfo)
                splitInfo += p * log(p);
        }
        
        return (wholeData.getScore()-splitScore)/abs(splitInfo);
    }

    @Override
    protected ImpurityScore clone()
    {
        return new ImpurityScore(this);
    }
}
