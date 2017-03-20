package jsat.datatransform.featureselection;

import java.util.*;

import jsat.DataSet;
import jsat.classifiers.*;
import jsat.datatransform.*;
import static jsat.datatransform.featureselection.SFS.addFeature;
import static jsat.datatransform.featureselection.SFS.removeFeature;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.IntList;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;
import jsat.utils.random.RandomUtil;

/**
 * Sequential Backward Selection (SBS) is a greedy method of selecting a subset 
 * of features to use for prediction. It starts from the set of all features and 
 * attempts to remove the least informative feature from the set at each 
 * iteration
 * 
 * @author Edward Raff
 */
public class SBS extends RemoveAttributeTransform
{

    private static final long serialVersionUID = -2516121100148559742L;
    private double maxDecrease;
    private int folds;
    private int minFeatures, maxFeatures;
    private Object evaluator;

    
    /**
     * Copy constructor
     * @param toClone the version to copy
     */
    private SBS(SBS toClone)
    {
        super(toClone);
        this.maxDecrease = toClone.maxDecrease;
        this.folds = toClone.folds;
        this.minFeatures = toClone.minFeatures;
        this.maxFeatures = toClone.maxFeatures;
        this.evaluator = toClone.evaluator;
        
    }
    
    /**
     * Performs SBS feature selection for a classification problem
     *
     * @param minFeatures the minimum number of features to find
     * @param maxFeatures the maximum number of features to find
     * @param evaluater the classifier to use in determining accuracy given a
     * feature subset
     * @param folds the number of cross validation folds to use in selection
     * @param maxDecrease the maximum tolerable decrease in accuracy in accuracy
     * when a feature is removed
     */
    public SBS(int minFeatures, int maxFeatures, Classifier evaluater, double maxDecrease)
    {
        this(minFeatures, maxFeatures, evaluater, 3, maxDecrease);
    }

    private SBS(int minFeatures, int maxFeatures, Object evaluater, int folds, double maxDecrease)
    {
        super();
        setMaxDecrease(maxDecrease);
        setMinFeatures(minFeatures);
        setMaxFeatures(maxFeatures);
        setEvaluator(evaluater);
        setFolds(folds);
    }
    
    /**
     * Performs SBS feature selection for a classification problem
     *
     * @param minFeatures the minimum number of features to find
     * @param maxFeatures the maximum number of features to find
     * @param cds the data set to perform feature selection on 
     * @param evaluater the classifier to use in determining accuracy given a
     * feature subset
     * @param folds the number of cross validation folds to use in selection
     * @param maxDecrease the maximum tolerable decrease in accuracy in accuracy
     * when a feature is removed
     */
    public SBS(int minFeatures, int maxFeatures, ClassificationDataSet cds, Classifier evaluater, int folds, double maxDecrease)
    {
        this(minFeatures, maxFeatures, evaluater, folds, maxDecrease);
        search(cds, evaluater, minFeatures, maxFeatures, folds);
    }
    
    /**
     * Performs SBS feature selection for a regression problem
     *
     * @param minFeatures the minimum number of features to find
     * @param maxFeatures the maximum number of features to find
     * @param evaluater the regressor to use in determining accuracy given a
     * feature subset
     * @param maxDecrease the maximum tolerable increase in the error rate when
     * a feature is removed
     */
    public SBS(int minFeatures, int maxFeatures, Regressor evaluater, double maxDecrease)
    {
        this(minFeatures, maxFeatures, evaluater, 3, maxDecrease);
    }
    /**
     * Performs SBS feature selection for a regression problem
     *
     * @param minFeatures the minimum number of features to find
     * @param maxFeatures the maximum number of features to find
     * @param rds the data set to perform feature selection on 
     * @param evaluater the regressor to use in determining accuracy given a
     * feature subset
     * @param folds the number of cross validation folds to use in selection
     * @param maxDecrease the maximum tolerable increase in the error rate when
     * a feature is removed
     */
    public SBS(int minFeatures, int maxFeatures, RegressionDataSet rds, Regressor evaluater, int folds, double maxDecrease)
    {
        this(minFeatures, maxFeatures, evaluater, folds, maxDecrease);
        search(rds, evaluater, minFeatures, maxFeatures, folds);
    }

    @Override
    public void fit(DataSet data)
    {
        super.fit(data); 
        search(data, evaluator, minFeatures, maxFeatures, folds);
    }
    
    private void search(DataSet dataSet, Object learner, int minFeatures, int maxFeatures, int folds)
    {
        Random rand = RandomUtil.getRandom();
        int nF = dataSet.getNumFeatures();
        int nCat = dataSet.getNumCategoricalVars();
        
        Set<Integer> available = new IntSet();
        ListUtils.addRange(available, 0, nF, 1);
        Set<Integer> catSelected = new IntSet(dataSet.getNumCategoricalVars());
        Set<Integer> numSelected = new IntSet(dataSet.getNumNumericalVars());
        
        Set<Integer> catToRemove = new IntSet(dataSet.getNumCategoricalVars());
        Set<Integer> numToRemove = new IntSet(dataSet.getNumNumericalVars());

        //Start will all selected, and prune them out
        ListUtils.addRange(catSelected, 0, nCat, 1);
        ListUtils.addRange(numSelected, 0, nF-nCat, 1);
        
        double[] bestScore = new double[]{Double.POSITIVE_INFINITY};

        while(catSelected.size() + numSelected.size() > minFeatures)
        {
            
            if(SBSRemoveFeature(available, dataSet, catToRemove, numToRemove, 
                    catSelected, numSelected, learner, folds, rand, 
                    maxFeatures, bestScore, maxDecrease) < 0)
                break;

        }
        
        int pos = 0;
        catIndexMap = new int[catSelected.size()];
        for(int i : catSelected)
            catIndexMap[pos++] = i;
        Arrays.sort(catIndexMap);
        
        pos = 0;
        numIndexMap = new int[numSelected.size()];
        for(int i : numSelected)
            numIndexMap[pos++] = i;
        Arrays.sort(numIndexMap);
    }
    
    @Override
    public SBS clone()
    {
        return new SBS(this);
    }
    
    /**
     * Returns a copy of the set of categorical features selected by the search 
     * algorithm
     * 
     * @return the set of categorical features to use
     */
    public Set<Integer> getSelectedCategorical()
    {
        return new IntSet(IntList.view(catIndexMap, catIndexMap.length));
    }
    
    /**
     * Returns a copy of the set of numerical features selected by the search 
     * algorithm. 
     * 
     * @return the set of numeric features to use
     */
    public Set<Integer> getSelectedNumerical()
    {
        return new IntSet(IntList.view(numIndexMap, numIndexMap.length));
    }

    /**
     * Attempts to remove one feature from the list while maintaining its
     * accuracy
     *
     * @param available the set of available features from [0, n) to consider 
     * for removal
     * @param dataSet the original data set to perform feature selection from
     * @param catToRemove the current set of categorical features to remove
     * @param numToRemove the current set of numerical features to remove
     * @param catSelecteed the current set of categorical features we are keeping
     * @param numSelected the current set of numerical features we are keeping
     * @param evaluater the classifier or regressor to perform evaluations with
     * @param folds the number of cross validation folds to determine performance
     * @param rand the source of randomness
     * @param maxFeatures the maximum allowable number of features
     * @param PbestScore an array to behave as a pointer to the best score seen 
     * so far
     * @param maxDecrease the maximum allowable decrease in accuracy from the 
     * best accuracy we see
     * @return the feature that was selected to be removed, or -1 if none were 
     * removed
     */
    protected static int SBSRemoveFeature(Set<Integer> available, DataSet dataSet,
            Set<Integer> catToRemove, Set<Integer> numToRemove, 
            Set<Integer> catSelecteed, Set<Integer> numSelected, 
            Object evaluater, int folds, Random rand, int maxFeatures, 
            double[] PbestScore, double maxDecrease)
    {
        int curBest = -1;
        int nCat = dataSet.getNumCategoricalVars();
        double curBestScore = Double.POSITIVE_INFINITY;
        for(int feature : available)
        {
            DataSet workOn = dataSet.shallowClone();
            addFeature(feature, nCat, catToRemove, numToRemove);
            
            RemoveAttributeTransform remove = new RemoveAttributeTransform(workOn, catToRemove, numToRemove);
            workOn.applyTransform(remove);
            
            double score = SFS.getScore(workOn, evaluater, folds, rand);
            
            if(score < curBestScore)
            {
                curBestScore = score;
                curBest = feature;
            }
            removeFeature(feature, nCat, catToRemove, numToRemove);
        }
        if (catSelecteed.size() + numSelected.size() > maxFeatures
                 || PbestScore[0] - curBestScore > -maxDecrease)
        {
            PbestScore[0] = curBestScore;
            removeFeature(curBest, nCat, catSelecteed, numSelected);
            addFeature(curBest, nCat, catToRemove, numToRemove);
            available.remove(curBest);
            return curBest;
        }
        else
            return  -1; //No possible improvment & weve got enough
    }

    /**
     * Sets the maximum allowable decrease in accuracy (increase in error) from
     * the previous set of features to the new current set.
     *
     * @param maxDecrease the maximum allowable decrease in the accuracy from
     * removing a feature
     */
    public void setMaxDecrease(double maxDecrease)
    {
        if (maxDecrease < 0)
            throw new IllegalArgumentException("Decarese must be a positive value, not " + maxDecrease);
        this.maxDecrease = maxDecrease;
    }

    /**
     * Returns the maximum allowable decrease in accuracy from one set of
     * features to the next
     *
     * @return the maximum allowable decrease in accuracy from one set of
     * features to the next
     */
    public double getMaxDecrease()
    {
        return maxDecrease;
    }

    /**
     * Sets the minimum number of features that must be selected
     *
     * @param minFeatures the minimum number of features to learn
     */
    public void setMinFeatures(int minFeatures)
    {
        this.minFeatures = minFeatures;
    }

    /**
     * Returns the minimum number of features to find
     *
     * @return the minimum number of features to find
     */
    public int getMinFeatures()
    {
        return minFeatures;
    }

    /**
     * Sets the maximum number of features that must be selected
     *
     * @param maxFeatures the maximum number of features to find
     */
    public void setMaxFeatures(int maxFeatures)
    {
        this.maxFeatures = maxFeatures;
    }

    /**
     * Returns the maximum number of features to find
     *
     * @return the maximum number of features to find
     */
    public int getMaxFeatures()
    {
        return maxFeatures;
    }
    
    /**
     * Sets the number of folds to use for cross validation when estimating the error rate
     * @param folds the number of folds to use for cross validation when estimating the error rate
     */
    public void setFolds(int folds)
    {
        if(folds <= 0 )
            throw new IllegalArgumentException("Number of CV folds must be positive, not " + folds);
        this.folds = folds;
    }

    /**
     * 
     * @return the number of folds to use for cross validation when estimating the error rate
     */
    public int getFolds()
    {
        return folds;
    }
    
    private void setEvaluator(Object evaluator)
    {
        this.evaluator = evaluator;
    }
}
