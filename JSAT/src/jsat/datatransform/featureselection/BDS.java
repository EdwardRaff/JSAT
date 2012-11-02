package jsat.datatransform.featureselection;

import java.util.*;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.datatransform.DataTransformFactory;
import jsat.datatransform.RemoveAttributeTransform;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.ListUtils;

/**
 * Bidirectional Search (BDS)  is a greedy method of selecting a subset 
 * of features to use for prediction. It performs both {@link SFS} and 
 * {@link SBS} search at the same time. At each step, a feature is greedily 
 * added to one set, and then a feature greedily removed from another set. 
 * Once a feature si added / removed in one set, it is unavailable for selection
 * in the other. This can be used to select up to half of the original features. 
 * 
 * @author Edward Raff
 */
public class BDS implements DataTransform
{
    private RemoveAttributeTransform finalTransform;
    private Set<Integer> catSelected;
    private Set<Integer> numSelected;

    /**
     * Copy constructor
     * 
     * @param toClone 
     */
    public BDS(BDS toClone)
    {
        if(toClone.finalTransform != null)
        {
            this.finalTransform = toClone.finalTransform.clone();
            this.catSelected = new HashSet<Integer>(toClone.catSelected);
            this.numSelected = new HashSet<Integer>(toClone.numSelected);
        }
    }

    /**
     * Performs BDS feature selection for a classification problem
     * 
     * @param featureCount the number of features to select
     * @param dataSet the data set to perform feature selection on
     * @param evaluator the classifier to use in determining accuracy given a 
     * feature subset
     * @param folds the number of cross validation folds to use in selection
     */
    public BDS(int featureCount, ClassificationDataSet dataSet, Classifier evaluator, int folds)
    {
        search(dataSet, featureCount, folds, evaluator);
    }
    
    /**
     * Performs BDS feature selection for a regression problem
     * 
     * @param featureCount the number of features to select
     * @param dataSet the data set to perform feature selection on
     * @param evaluator the regressor to use in determining accuracy given a 
     * feature subset
     * @param folds the number of cross validation folds to use in selection
     */
    public BDS(int featureCount, RegressionDataSet dataSet, Regressor evaluator, int folds)
    {
        search(dataSet, featureCount, folds, evaluator);
    }

    @Override
    public DataPoint transform(DataPoint dp)
    {
        return finalTransform.transform(dp);
    }

    @Override
    public DataTransform clone()
    {
        return new BDS(this);
    }
    
    /**
     * Returns a copy of the set of categorical features selected by the search 
     * algorithm
     * 
     * @return the set of categorical features to use
     */
    public Set<Integer> getSelectedCategorical()
    {
        return new HashSet<Integer>(catSelected);
    }
    
    /**
     * Returns a copy of the set of numerical features selected by the search 
     * algorithm. 
     * 
     * @return the set of numeric features to use
     */
    public Set<Integer> getSelectedNumerical()
    {
        return new HashSet<Integer>(numSelected);
    }

    private void search(DataSet dataSet, int maxFeatures, int folds, Object evaluator)
    {
        Random rand = new Random();
        int nF = dataSet.getNumFeatures();
        int nCat = dataSet.getNumCategoricalVars();
        
        //True selected, also used for SFS
        catSelected = new HashSet<Integer>(dataSet.getNumCategoricalVars());
        numSelected = new HashSet<Integer>(dataSet.getNumNumericalVars());
        
        //Structs for SFS side
        Set<Integer> availableSFS = new HashSet<Integer>();
        ListUtils.addRange(availableSFS, 0, nF, 1);
        
        
        Set<Integer> catToRemoveSFS = new HashSet<Integer>(dataSet.getNumCategoricalVars());
        Set<Integer> numToRemoveSFS = new HashSet<Integer>(dataSet.getNumNumericalVars());
        ListUtils.addRange(catToRemoveSFS, 0, nCat, 1);
        ListUtils.addRange(numToRemoveSFS, 0, nF-nCat, 1);
        
        ///Structes fro SBS side
        Set<Integer> availableSBS = new HashSet<Integer>();
        ListUtils.addRange(availableSBS, 0, nF, 1);
        Set<Integer> catSelecteedSBS = new HashSet<Integer>(dataSet.getNumCategoricalVars());
        Set<Integer> numSelectedSBS = new HashSet<Integer>(dataSet.getNumNumericalVars());
        
        Set<Integer> catToRemoveSBS = new HashSet<Integer>(dataSet.getNumCategoricalVars());
        Set<Integer> numToRemoveSBS = new HashSet<Integer>(dataSet.getNumNumericalVars());

        //Start will all selected, and prune them out
        ListUtils.addRange(catSelecteedSBS, 0, nCat, 1);
        ListUtils.addRange(numSelectedSBS, 0, nF-nCat, 1);
        
        double[] pBestScore0 = new double[]{Double.POSITIVE_INFINITY};
        double[] pBestScore1 = new double[]{Double.POSITIVE_INFINITY};
        int max = Math.min(maxFeatures, nF/2);
        for(int i = 0; i < max; i++)
        {
            //Find and keep one good one
            int mustKeep = SFS.SFSSelectFeature(availableSFS, dataSet, 
                    catToRemoveSFS, numToRemoveSFS, catSelected, 
                    numSelected, evaluator, folds, rand, pBestScore0, max);
            availableSBS.remove(mustKeep);
            SFS.removeFeature(mustKeep, nCat, catToRemoveSBS, numToRemoveSBS);
            
            //Find and remove one bad one
            int mustRemove = SBS.SBSRemoveFeature(availableSBS, dataSet, 
                    catToRemoveSBS, numToRemoveSBS, catSelecteedSBS, 
                    numSelectedSBS, evaluator, folds, rand, max, 
                    pBestScore1, 0.0);
            availableSFS.remove(mustRemove);
            SFS.addFeature(mustRemove, nCat, catToRemoveSFS, numToRemoveSFS);
        }
        
        catSelecteedSBS.clear();
        numToRemoveSBS.clear();
        ListUtils.addRange(catSelecteedSBS, 0, nCat, 1);
        ListUtils.addRange(numSelectedSBS, 0, nF-nCat, 1);
        
        catSelecteedSBS.removeAll(catSelected);
        numSelectedSBS.removeAll(numSelected);
        
        finalTransform = new RemoveAttributeTransform(dataSet, catToRemoveSBS, numToRemoveSBS);
    }
    
    static public class BDSFactory implements DataTransformFactory
    {
        private Classifier classifier;
        private Regressor regressor;
        private int featureCount;
        
        /**
         * Creates a new BDS factory
         * @param evaluater  the classifier to use in determining accuracy given
         * a feature subset
         * @param featureCount the number of features to select
         */
        public BDSFactory(Classifier evaluater, int featureCount)
        {
            this.classifier = evaluater;
            this.featureCount = featureCount;
        }
        
        /**
         * Creates a new BDS factory
         * @param evaluater the regressor to use in determining accuracy given a 
         * feature subset
         * @param featureCount the number of features to select 
         */
        public BDSFactory(Regressor evaluater, int featureCount)
        {
            this.regressor = evaluater;
            this.featureCount = featureCount;
        }

        @Override
        public DataTransform getTransform(DataSet dataset)
        {
            if(dataset instanceof ClassificationDataSet)
                return new BDS(featureCount, (ClassificationDataSet)dataset, 
                        classifier, 5);
            else
                return new BDS(featureCount, (RegressionDataSet)dataset,
                        regressor, featureCount);
        }
        
    }
}
