package jsat.datatransform.featureselection;

import java.util.*;

import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.datatransform.*;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;
import jsat.utils.random.RandomUtil;

/**
 * plus-L minus-R Selection (LRS) is a greedy method of selecting a subset 
 * of features to use for prediction. Its behavior is dependent upon whether L 
 * or R is the larger value. No mater what, L features will be greedily added to
 * the set to decrease the error rate, and R features will be greedily removed 
 * while trying to maintain the error rate. <br>
 * If L &gt; R, then L-R features will be selected, the L step running first 
 * followed by R performing pruning on the found set. <br>
 * If L &lt; R, then D-R+L features will be selected, where D is the original 
 * number of features. First R features will be removed, and then L of the 
 * removed features will be added back to the final set. <br>
 * L = R is not allowed. 
 * 
 * @author Edward Raff
 */
public class LRS implements DataTransform
{

    private static final long serialVersionUID = 3065300352046535656L;
    private RemoveAttributeTransform finalTransform;
    private Set<Integer> catSelected;
    private Set<Integer> numSelected;
    private int L;
    private int R;
    private Object evaluater;
    private int folds;

    /**
     * Copy constructor
     * @param toClone the version to copy
     */
    private LRS(LRS toClone)
    {
        this.L = toClone.L;
        this.R = toClone.R;
        this.folds = toClone.folds;
        this.evaluater = toClone.evaluater;
        if(toClone.catSelected != null)
        {
            this.finalTransform = toClone.finalTransform.clone();
            this.catSelected = new IntSet(toClone.catSelected);
            this.numSelected = new IntSet(toClone.numSelected);
        }
    }
    
    /**
     * Creates a LRS feature selection object for a classification problem
     * 
     * @param L the number of features to greedily add
     * @param R the number of features to greedily remove
     * @param evaluater the classifier to use in determining accuracy given a 
     * feature subset
     * @param folds the number of cross validation folds to use in selection
     */
    public LRS(int L, int R, Classifier evaluater, int folds)
    {
        setFeaturesToAdd(L);
        setFeaturesToRemove(R);
        setFolds(folds);
        setEvaluator(evaluater);
    }
    
    /**
     * Performs LRS feature selection for a classification problem
     * 
     * @param L the number of features to greedily add
     * @param R the number of features to greedily remove
     * @param cds the data set to perform feature selection on
     * @param evaluater the classifier to use in determining accuracy given a 
     * feature subset
     * @param folds the number of cross validation folds to use in selection
     */
    public LRS(int L, int R, ClassificationDataSet cds, Classifier evaluater, int folds)
    {
        search(cds, L, R, evaluater, folds);
    }
    
    /**
     * Creates a LRS feature selection object for a regression problem
     * 
     * @param L the number of features to greedily add
     * @param R the number of features to greedily remove
     * @param evaluater the regressor to use in determining accuracy given a 
     * feature subset
     * @param folds the number of cross validation folds to use in selection
     */
    public LRS(int L, int R, Regressor evaluater, int folds)
    {
        setFeaturesToAdd(L);
        setFeaturesToRemove(R);
        setFolds(folds);
        setEvaluator(evaluater);
    }
    
    /**
     * Performs LRS feature selection for a regression problem
     * 
     * @param L the number of features to greedily add
     * @param R the number of features to greedily remove
     * @param rds the data set to perform feature selection on
     * @param evaluater the regressor to use in determining accuracy given a 
     * feature subset
     * @param folds the number of cross validation folds to use in selection
     */
    public LRS(int L, int R, RegressionDataSet rds, Regressor evaluater, int folds)
    {
        this(L, R, evaluater, folds);
        search(rds, L, R, evaluater, folds);
    }

    @Override
    public DataPoint transform(DataPoint dp)
    {
        return finalTransform.transform(dp);
    }

    @Override
    public LRS clone()
    {
        return new LRS(this);
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

    @Override
    public void fit(DataSet data)
    {
        search(data, L, R, evaluater, folds);
    }
    
    private void search(DataSet cds, int L, int R, Object evaluater, int folds)
    {
        int nF = cds.getNumFeatures();
        int nCat = cds.getNumCategoricalVars();
        
        catSelected = new IntSet(nCat);
        numSelected = new IntSet(nF-nCat);
        Set<Integer> catToRemove = new IntSet(nCat);
        Set<Integer> numToRemove = new IntSet(nF-nCat);
        
        Set<Integer> available = new IntSet(nF);
        ListUtils.addRange(available, 0, nF, 1);
        
        Random rand = RandomUtil.getRandom();
        double[] pBestScore = new double[]{Double.POSITIVE_INFINITY};
        
        if (L > R)
        {
            ListUtils.addRange(catToRemove, 0, nCat, 1);
            ListUtils.addRange(numToRemove, 0, nF-nCat, 1);
            
            //Select L features
            for(int i = 0; i < L; i++)
                SFS.SFSSelectFeature(available, cds, catToRemove, numToRemove, 
                        catSelected, numSelected, evaluater, folds, 
                        rand, pBestScore, L);
            //We now restrict ourselves to the L features
            available.clear();
            available.addAll(catSelected);
            for(int i : numSelected)
                available.add(i+nCat);
            //Now remove R features from the L selected
            for(int i = 0; i < R; i++)
                SBS.SBSRemoveFeature(available, cds, catToRemove, numToRemove, 
                        catSelected, numSelected, evaluater, folds, rand, 
                        L-R, pBestScore, 0.0);
        }
        else if(L < R)
        {
            ListUtils.addRange(catSelected, 0, nCat, 1);
            ListUtils.addRange(numSelected, 0, nF-nCat, 1);
            
            //Remove R features
            for(int i = 0; i < R; i++)
                SBS.SBSRemoveFeature(available, cds, catToRemove, numToRemove, 
                        catSelected, numSelected, evaluater, folds, rand, 
                        nF-R, pBestScore, 0.0);
            
            //Now we restrict out selves to adding back the features that were removed
            available.clear();
            available.addAll(catToRemove);
            for(int i : numToRemove)
                available.add(i+nCat);
            
            //Now add L features back
            for(int i = 0; i < L; i++)
                SFS.SFSSelectFeature(available, cds, catToRemove, numToRemove, 
                        catSelected, numSelected, evaluater, folds, 
                        rand, pBestScore, R-L);
        }
        
        finalTransform = new RemoveAttributeTransform(cds, catToRemove, numToRemove);
    }
    
    /**
     * Sets the number of features to add (the L parameter).
     *
     * @param featuresToAdd the number of features to greedily add
     */
    public void setFeaturesToAdd(int featuresToAdd)
    {
        if (featuresToAdd < 1)
            throw new IllegalArgumentException("Number of features to add must be positive, not " + featuresToAdd);
        this.L = featuresToAdd;
    }

    /**
     * Returns the number of features to add
     *
     * @return the number of features to add
     */
    public int getFeaturesToAdd()
    {
        return L;
    }

    /**
     * Sets the number of features to remove (the R parameter).
     *
     * @param featuresToRemove the number of features to greedily remove
     */
    public void setFeaturesToRemove(int featuresToRemove)
    {
        if (featuresToRemove < 1)
            throw new IllegalArgumentException("Number of features to remove must be positive, not " + featuresToRemove);
        this.R = featuresToRemove;
    }

    /**
     * Returns the number of features to remove
     *
     * @return the number of features to remove
     */
    public int getFeaturesToRemove()
    {
        return R;
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
        this.evaluater = evaluator;
    }
}
