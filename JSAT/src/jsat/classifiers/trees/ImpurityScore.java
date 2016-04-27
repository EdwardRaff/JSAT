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
        /**
         * Normalized Mutual Information. The {@link #getScore() } value will be
         * the same as {@link #INFORMATION_GAIN}, however - the gain returned 
         * is considerably different - and is a normalization of the mutual 
         * information between the split and the class label by the class and 
         * split entropy. 
         */
        NMI,
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
        if(sumOfWeights <= 0)
            return 0;
        double score = 0.0;

        if (impurityMeasure == ImpurityMeasure.INFORMATION_GAIN_RATIO
                || impurityMeasure == ImpurityMeasure.INFORMATION_GAIN
                || impurityMeasure == ImpurityMeasure.NMI)
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
        return gain(wholeData, 1.0, splits);
    }
    
    /**
     * Computes the gain in score from a splitting of the data set
     * 
     * @param wholeData the score for the whole data set
     * @param wholeScale a constant to scale the wholeData counts and sums by, useful for handling missing value cases
     * @param splits the scores for each of the splits
     * @return the gain for the values given
     */
    public static double gain(ImpurityScore wholeData, double wholeScale, ImpurityScore... splits)
    {
        double sumOfAllSums = wholeScale*wholeData.sumOfWeights;
        
        if(splits[0].impurityMeasure == ImpurityMeasure.NMI)
        {
            double mi = 0, splitEntropy = 0.0, classEntropy = 0.0;
            
            for(int c = 0; c < wholeData.counts.length; c++)//c: class
            {
                final double p_c = wholeScale*wholeData.counts[c]/sumOfAllSums;
                if(p_c <= 0.0)
                    continue;
                
                double logP_c = log(p_c);
                
                classEntropy += p_c*logP_c;
                        
                for(int s = 0; s < splits.length; s++)//s: split
                {
                    final double p_s = splits[s].sumOfWeights/sumOfAllSums;
                    if(p_s <= 0)
                        continue;
                    final double p_cs = splits[s].counts[c]/sumOfAllSums;
                    if(p_cs <= 0)
                        continue;
                    
                    mi += p_cs * (log(p_cs) - logP_c - log(p_s));
                    
                    if(c == 0)
                        splitEntropy += p_s * log(p_s);
                }
            }
            
            splitEntropy = abs(splitEntropy);
            classEntropy = abs(classEntropy);
            
            return 2*mi/(splitEntropy+classEntropy);
            
        }
        //Else, normal cases
        double splitScore = 0.0;
        
        boolean useSplitInfo = splits[0].impurityMeasure == ImpurityMeasure.INFORMATION_GAIN_RATIO;
        
        if(useSplitInfo)
        {
            /*
             * TODO should actualy be 0, but performance bug is consistently 
             * occuring if I use another value. Needs serious investigation. 
             * I was testing on (Oracle) 1.7u51 & u20 smoething and both had the
             * issue, on OSX and Windows. 
             * 
             * I was unable to replicate the issue with a smaller self contained
             * program. So I suspect I might be at some threshold / corner case 
             * of the optimizer
             * 
             * Adding a -1 at the final results causes the performance 
             * degredation agian. Occures with both client and server JVM
             * 
             * Using the same code with an if stament seperating the 2 (see old revision) was originally backwards. Changing the correct way revealed the behavior. I'm leaving them seperated to ease investiation later. 
             */
            double splitInfo = 1.0;
            for(ImpurityScore split : splits)
            {
                double p = split.getSumOfWeights()/sumOfAllSums;
                if(p <= 0)//log(0) is -Inft, so skip and treat as zero
                    continue;
                splitScore += p * split.getScore();
                splitInfo += p * -log(p);
            }

            return (wholeData.getScore()-splitScore)/splitInfo;
        }
        else
        {
            for(ImpurityScore split : splits)
            {
                double p = split.getSumOfWeights()/sumOfAllSums;
                if(p <= 0)//log(0) is -Inft, so skip and treat as zero
                    continue;
                splitScore += p*split.getScore();
            }

            return wholeData.getScore()-splitScore;
        }
        
    }
    
    @Override
    protected ImpurityScore clone()
    {
        return new ImpurityScore(this);
    }
}
