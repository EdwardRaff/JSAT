package jsat.datatransform.featureselection;

import java.util.*;

import jsat.DataSet;
import jsat.classifiers.*;
import jsat.datatransform.*;
import jsat.regression.*;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;
import jsat.utils.random.RandomUtil;

/**
 * Sequential Forward Selection (SFS) is a greedy method of selecting a subset 
 * of features to use for prediction. It starts from the set of no features and 
 * attempts to add the next best feature to the set at each iteration. 
 * 
 * @author Edward Raff
 */
public class SFS implements DataTransform
{

    private static final long serialVersionUID = 140187978708131002L;
    private RemoveAttributeTransform finalTransform;
    private Set<Integer> catSelected;
    private Set<Integer> numSelected;
    private double maxIncrease;
    private Classifier classifier;
    private Regressor regressor;
    private int minFeatures, maxFeatures;
    private int folds;
    private Object evaluator;

    /**
     * Copy constructor 
     * @param toClone the SFS to copy
     */
    private SFS(SFS toClone)
    {
        if(toClone.catSelected != null)
        {
            this.finalTransform = toClone.finalTransform.clone();
            this.catSelected = new IntSet(toClone.catSelected);
            this.numSelected = new IntSet(toClone.numSelected);
        }

        this.maxIncrease = toClone.maxIncrease;
        this.folds = toClone.folds;
        this.minFeatures = toClone.minFeatures;
        this.maxFeatures = toClone.maxFeatures;
        this.evaluator = toClone.evaluator;
        if (toClone.classifier != null)
            this.classifier = toClone.classifier.clone();
        if (toClone.regressor != null)
            this.regressor = toClone.regressor.clone();
    }
    
    /**
     * Performs SFS feature selection for a classification problem
     *
     * @param minFeatures the minimum number of features to find
     * @param maxFeatures the maximum number of features to find
     * @param evaluater the classifier to use in determining accuracy given a
     * feature subset
     * @param maxIncrease the maximum tolerable increase in error when a feature
     * is added
     */
    public SFS(int minFeatures, int maxFeatures, Classifier evaluater, double maxIncrease)
    {
        this(minFeatures, maxFeatures, evaluater.clone(), 3, maxIncrease);
    }
    
    /**
     * Performs SFS feature selection for a classification problem
     *
     * @param minFeatures the minimum number of features to find
     * @param maxFeatures the maximum number of features to find
     * @param dataSet the data set to perform feature selection on
     * @param evaluater the classifier to use in determining accuracy given a
     * feature subset
     * @param folds the number of cross validation folds to use in selection
     * @param maxIncrease the maximum tolerable increase in error when a feature
     * is added
     */
    public SFS(int minFeatures, int maxFeatures, ClassificationDataSet dataSet, Classifier evaluater, int folds, double maxIncrease)
    {
        this(minFeatures, maxFeatures, evaluater.clone(), folds, maxIncrease);
        search(minFeatures, maxFeatures, dataSet, folds);
    }
    
    /**
     * Creates SFS feature selection for a regression problem
     *
     * @param minFeatures the minimum number of features to find
     * @param maxFeatures the maximum number of features to find
     * @param regressor the regressor to use in determining accuracy given a
     * feature subset
     * @param maxIncrease the maximum tolerable increase in error when a feature
     * is added
     */
    public SFS(int minFeatures, int maxFeatures, Regressor regressor, double maxIncrease)
    {
        this(minFeatures, maxFeatures, regressor.clone(), 3, maxIncrease);
    }
    
    /**
     * Performs SFS feature selection for a regression problem
     *
     * @param minFeatures the minimum number of features to find
     * @param maxFeatures the maximum number of features to find
     * @param dataSet the data set to perform feature selection on
     * @param regressor the regressor to use in determining accuracy given a
     * feature subset
     * @param folds the number of cross validation folds to use in selection
     * @param maxIncrease the maximum tolerable increase in error when a feature
     * is added
     */
    public SFS(int minFeatures, int maxFeatures, RegressionDataSet dataSet, Regressor regressor, int folds, double maxIncrease)
    {
        this(minFeatures, maxFeatures, regressor.clone(), folds, maxIncrease);
        search(minFeatures, maxFeatures, dataSet, folds);
    }
    
    private SFS(int minFeatures, int maxFeatures, Object evaluator, int folds, double maxIncrease)
    {
        setMinFeatures(minFeatures);
        setMaxFeatures(maxFeatures);
        setFolds(folds);
        setMaxIncrease(maxIncrease);
        setEvaluator(evaluator);
    }

    @Override
    public void fit(DataSet data)
    {
        search(minFeatures, maxFeatures, data, minFeatures);
    }
    
    private void search(int minFeatures, int maxFeatures, DataSet dataSet, int folds)
    {
        Random rand = RandomUtil.getRandom();
        int nF = dataSet.getNumFeatures();
        int nCat = dataSet.getNumCategoricalVars();
        
        Set<Integer> available = new IntSet();
        ListUtils.addRange(available, 0, nF, 1);
        catSelected = new IntSet(dataSet.getNumCategoricalVars());
        numSelected = new IntSet(dataSet.getNumNumericalVars());
        
        Set<Integer> catToRemove = new IntSet(dataSet.getNumCategoricalVars());
        Set<Integer> numToRemove = new IntSet(dataSet.getNumNumericalVars());
        ListUtils.addRange(catToRemove, 0, nCat, 1);
        ListUtils.addRange(numToRemove, 0, nF-nCat, 1);
        
        double[] bestScore = new double[]{Double.POSITIVE_INFINITY};
        
        Object learner = regressor;
        if (dataSet instanceof ClassificationDataSet)
            learner = classifier;
            
        while (catSelected.size() + numSelected.size() < maxFeatures)
        {
            if (SFSSelectFeature(available, dataSet,
                    catToRemove, numToRemove, catSelected, numSelected,
                    learner, folds, rand, bestScore, minFeatures) < 0)
                break;

        }
        
        this.finalTransform = new RemoveAttributeTransform(dataSet, catToRemove, numToRemove);
    }

    /**
     *
     * @param curBest the value of curBest
     * @param nCat the value of nCat
     * @param catF the value of catF
     * @param numF the value of numF
     */
    static protected void addFeature(int curBest, int nCat, Set<Integer> catF, Set<Integer> numF)
    {
        if(curBest >= nCat)
            numF.add(curBest-nCat);
        else
            catF.add(curBest);
    }

    /**
     *
     * @param feature the value of feature
     * @param nCat the value of nCat
     * @param catF the value of catF
     * @param numF the value of numF
     */
    static protected void removeFeature(int feature, int nCat, Set<Integer> catF, Set<Integer> numF)
    {
        if(feature >= nCat)
            numF.remove(feature-nCat);
        else
            catF.remove(feature);
    }

    @Override
    public DataPoint transform(DataPoint dp)
    {
        return finalTransform.transform(dp);
    }

    @Override
    public SFS clone()
    {
        return new SFS(this);
    }
    
    /**
     * Returns a copy of the set of categorical features selected by the search 
     * algorithm
     * 
     * @return the set of categorical features to use
     */
    public Set<Integer> getSelectedCategorical()
    {
        return new IntSet(catSelected);
    }
    
    /**
     * Returns a copy of the set of numerical features selected by the search 
     * algorithm. 
     * 
     * @return the set of numeric features to use
     */
    public Set<Integer> getSelectedNumerical()
    {
        return new IntSet(numSelected);
    }

    /**
     * Attempts to add one feature to the list of features while increasing or 
     * maintaining the current accuracy
     * 
     * @param available the set of available features from [0, n) to consider 
     * for adding
     * @param dataSet the original data set to perform feature selection from
     * @param catToRemove the current set of categorical features to remove
     * @param numToRemove the current set of numerical features to remove 
     * @param catSelecteed the current set of categorical features we are keeping
     * @param numSelected the current set of numerical features we are keeping
     * @param evaluater the classifier or regressor to perform evaluations with
     * @param folds the number of cross validation folds to determine performance
     * @param rand the source of randomness
     * @param PbestScore an array to behave as a pointer to the best score seen 
     * so far
     * @param minFeatures the minimum number of features needed
     * @return the feature that was selected to add, or -1 if none were added.
     */
    static protected int SFSSelectFeature(Set<Integer> available, 
            DataSet dataSet, Set<Integer> catToRemove, Set<Integer> numToRemove,
            Set<Integer> catSelecteed, Set<Integer> numSelected, 
            Object evaluater, int folds, Random rand, double[] PbestScore, 
            int minFeatures)
    {
        int nCat = dataSet.getNumCategoricalVars();
        int curBest = -1;
        double curBestScore = Double.POSITIVE_INFINITY;
        for(int feature : available)
        {
            removeFeature(feature, nCat, catToRemove, numToRemove);
            
            DataSet workOn = dataSet.shallowClone();
            RemoveAttributeTransform remove = new RemoveAttributeTransform(workOn, catToRemove, numToRemove);
            workOn.applyTransform(remove);
            
            double score = getScore(workOn, evaluater, folds, rand);
            
            if(score < curBestScore)
            {
                curBestScore = score;
                curBest = feature;
            }
            addFeature(feature, nCat, catToRemove, numToRemove);
        }
        if(curBestScore <= 1e-14 && PbestScore[0] <= 1e-14
                && catSelecteed.size() + numSelected.size() >= minFeatures )
            return -1;
        if (curBestScore < PbestScore[0] 
                 || catSelecteed.size() + numSelected.size() < minFeatures 
                 || Math.abs(PbestScore[0]-curBestScore) < 1e-3)
        {
            PbestScore[0] = curBestScore;
            addFeature(curBest, nCat, catSelecteed, numSelected);
            removeFeature(curBest, nCat, catToRemove, numToRemove);
            available.remove(curBest);
            return curBest;
        }
        else
            return -1; //No possible improvment & weve got enough
    }
    
    /**
     * The score function for a data set and a learner by cross validation of a 
     * classifier
     *
     * @param workOn the transformed data set to test from with cross validation
     * @param evaluater the learning algorithm to use
     * @param folds the number of cross validation folds to perform
     * @param rand the source of randomness
     * @return the score value in terms of cross validated error
     */
    protected static double getScore(DataSet workOn, Object evaluater, int folds, Random rand)
    {
        if(workOn instanceof ClassificationDataSet)
        {
            ClassificationModelEvaluation cme =
                    new ClassificationModelEvaluation((Classifier)evaluater, 
                    (ClassificationDataSet)workOn);
            cme.evaluateCrossValidation(folds, rand);

            return cme.getErrorRate();
        }
        else if(workOn instanceof RegressionDataSet)
        {
            RegressionModelEvaluation rme = 
                    new RegressionModelEvaluation((Regressor)evaluater, 
                    (RegressionDataSet)workOn);
            rme.evaluateCrossValidation(folds, rand);
            
            return rme.getMeanError();
        }
        return Double.POSITIVE_INFINITY;
    }
    
    /**
     * Sets the maximum allowable the maximum tolerable increase in error when a
     * feature is added
     *
     * @param maxIncrease the maximum allowable the maximum tolerable increase
     * in error when a feature is added
     */
    public void setMaxIncrease(double maxIncrease)
    {
        if (maxIncrease < 0)
            throw new IllegalArgumentException("Decarese must be a positive value, not " + maxIncrease);
        this.maxIncrease = maxIncrease;
    }

    /**
     *
     * @return the maximum allowable the maximum tolerable increase in error
     * when a feature is added
     */
    public double getMaxIncrease()
    {
        return maxIncrease;
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
        if(evaluator instanceof Classifier)
            this.classifier = (Classifier) evaluator;
        if(evaluator instanceof Regressor)
            this.regressor = (Regressor) evaluator;
    }
}
