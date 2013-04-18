package jsat.datatransform.featureselection;

import static java.lang.Math.log;
import java.util.HashSet;
import java.util.Set;
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.datatransform.*;
import jsat.exceptions.FailedToFitException;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.utils.IndexTable;

/**
 * Performs greedy feature selection based on Mutual Information of the features
 * with respect to the class values. This is an attempt to select features that 
 * are discriminative for classification tasks. <br>
 * The method of performing Mutual Information on numeric attributes is 
 * controlled by {@link NumericalHandeling}
 * 
 * @author Edward Raff
 */
public class MutualInfoFS extends RemoveAttributeTransform
{
    /**
     * The definition for mutual information for continuous attributes requires 
     * an integration of an unknown function, as such requires some form of 
     * approximation. This controls how the approximation is done
     */
    public enum NumericalHandeling
    {
        /**
         * Mutual information for numeric attributes is not computed, so no 
         * numeric attributes will be removed - and are ignored completely. The 
         * number of features to select does not include the numeric attributes 
         * in this case. 
         */
        NONE, 
        /**
         * Numeric attributes are treated as nominal features with binary 
         * values. The false value is if the value is zero, and the true value 
         * is any non zero value. 
         */
        BINARY,
    }

    /**
     * Creates a new Mutual Information feature selection object. Numeric 
     * attributes are handled by {@link NumericalHandeling#BINARY}  
     * 
     * @param dataSet the classification data set to perform feature selection
     * from
     * @param featureCount the number of features to select
     */
    public MutualInfoFS(ClassificationDataSet dataSet, int featureCount)
    {
        this(dataSet, featureCount, NumericalHandeling.BINARY);
    }
    
    /**
     * Creates a new Mutual Information feature selection object.
     *
     * @param dataSet the classification data set to perform feature selection
     * from
     * @param featureCount the number of features to select
     * @param numericHandling the way to handle the computation of mutual 
     * information for numeric attributes 
     */
    public MutualInfoFS(ClassificationDataSet dataSet, int featureCount, NumericalHandeling numericHandling)
    {
        super();
        if(featureCount <= 0)
            throw new RuntimeException("Number of features to select must be positive");
        final int N = dataSet.getSampleSize();
        double[] classPriors = dataSet.getPriors();
        double[] logClassPriors = new double[classPriors.length];
        for(int i = 0; i < logClassPriors.length; i++)
            logClassPriors[i] = log(classPriors[i]);
        
        int numCatVars;
        int consideredCount = numCatVars = dataSet.getNumCategoricalVars();
        if(numericHandling != NumericalHandeling.NONE)
            consideredCount = dataSet.getNumFeatures();
        
        /**
         * 1st index is the feature
         * 2nd index is the option #
         */
        double[][] featPriors = new double[consideredCount][];
        
        CategoricalData[] catInfo = dataSet.getCategories();

        /**
         * 1st index is the feature
         * 2nd index is the option #
         * 3rd index is the class
         */
        double[][][] jointProb = new double[consideredCount][][];
        for(int i = 0; i < jointProb.length; i++)
        {
            if(i < dataSet.getNumCategoricalVars())//Cat value
            {
                int options = catInfo[i].getNumOfCategories();
                jointProb[i] = new double[options][logClassPriors.length];
                featPriors[i] = new double[options];
            }
            else//Numeric value
            {
                //Yes/No, but only keep track of the yes values
                jointProb[i] = new double[2][logClassPriors.length];
                featPriors[i] = new double[1];//feature for numeric is just 1.0-other
            }
        }
        
        double weightSum = 0.0;
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            DataPoint dp = dataSet.getDataPoint(i);
            int trueClass = dataSet.getDataPointCategory(i);
            double weight = dp.getWeight();
            weightSum += weight;
            
            int[] catVals = dp.getCategoricalValues();
            for(int j = 0; j < catVals.length; j++)
            {
                featPriors[j][catVals[j]] += weight;
                jointProb[j][catVals[j]][trueClass] += weight;
            }
            
            if(numericHandling == NumericalHandeling.BINARY)
            {
                Vec numeric = dp.getNumericalValues();
                
                for(IndexValue iv : numeric)
                {
                    featPriors[iv.getIndex()+numCatVars][0] += weight;
                    jointProb[iv.getIndex()+numCatVars][0][trueClass] += weight;
                }
            }
        }

        /**
         * Mutual Information for each index
         */
        double[] mis = new double[consideredCount];
        
        for(int i = 0; i < consideredCount; i++)
        {
            double mi = 0.0;
            if( i < dataSet.getNumCategoricalVars())//Cat attribute
            {
                for(int tVal = 0; tVal < jointProb[i].length; tVal++)
                {
                    double featPrior = featPriors[i][tVal]/weightSum;
                    if(featPrior == 0.0)
                        continue;
                    double logFeatPrior = log(featPrior);
                    for (int tClass = 0; tClass < logClassPriors.length; tClass++)
                    {
                        double jp = jointProb[i][tVal][tClass] / weightSum;
                        if (jp == 0)
                            continue;
                        mi += jp * (log(jp) - logFeatPrior - logClassPriors[tClass]);
                    }
                }
            }
            else//Numeric attribute & it is binary
            {
                for(int tClass = 0; tClass < classPriors.length; tClass++)
                {
                    double jpNeg = jointProb[i][0][tClass]/weightSum;
                    double jpPos = (classPriors[tClass]*N - jointProb[i][0][tClass])/weightSum;
                    double posPrio = featPriors[i][0]/weightSum;
                    double negPrio = 1.0-posPrio;
                    
                    if (jpNeg != 0 && negPrio != 0)
                        mi += jpNeg * (log(jpNeg) - log(negPrio) - logClassPriors[tClass]);
                    if (jpPos != 0 && posPrio != 0)
                        mi += jpPos * (log(jpPos) - log(posPrio) - logClassPriors[tClass]);
                }
            }
            mis[i] = mi;
        }
        
        
        IndexTable sortedOrder = new IndexTable(mis);
        
        Set<Integer> catToRemove = new HashSet<Integer>();
        Set<Integer> numToRemove = new HashSet<Integer>();
        
        for(int i = 0; i < consideredCount-featureCount; i++)
        {
            int removingIndex = sortedOrder.index(i);
            if(removingIndex < numCatVars)
                catToRemove.add(removingIndex);
            else
                numToRemove.add(removingIndex-numCatVars);
        }
        
        setUp(dataSet, catToRemove, numToRemove);
    }
    
    public static class MutualInfoFSFactory implements DataTransformFactory
    {
        private int featureCount;
        private NumericalHandeling handling;

        /**
         * Creates a new MutalInfoFS factory that uses 
         * {@link NumericalHandeling#BINARY} handling for numeric attributes. 
         * 
         * @param featureCount  the number of features to select
         */
        public MutualInfoFSFactory(int featureCount)
        {
            this(featureCount, NumericalHandeling.BINARY);
        }
        
        /**
         * Creates a new MutualInfoFS factory
         * 
         * @param featureCount the number of features to select
         * @param handling the way to handle numeric attributes
         */
        public MutualInfoFSFactory(int featureCount, NumericalHandeling handling)
        {
            this.featureCount = featureCount;
            this.handling = handling;
        }

        
        @Override
        public DataTransform getTransform(DataSet dataset)
        {
            if(!(dataset instanceof ClassificationDataSet))
                throw new FailedToFitException("The given data set was not a classification data set");
            ClassificationDataSet cds = (ClassificationDataSet) dataset;
            return new MutualInfoFS(cds, featureCount, handling);
        }
    }
}
