
package jsat.classifiers.trees;

import java.util.Set;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.distributions.Distribution;
import jsat.distributions.empirical.KernelDensityEstimator;
import jsat.distributions.empirical.kernelfunc.EpanechnikovKF;
import jsat.distributions.empirical.kernelfunc.GaussKF;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.OnLineStatistics;
import jsat.math.rootfinding.Zeroin;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.PairedReturn;
import static java.lang.Math.*;

/**
 * This class is a 1-rule. It creates one rule that is used to classify all inputs, 
 * making it a decision tree with only one node. It can be used as a weak learner 
 * for ensemble learners, or as the nodes in a true decision tree. 
 * <br><br>
 * Categorical values are handled similarly under all circumstances. <br>
 * During classification, numeric attributes are separated based on most 
 * likely probability into their classes. <br>
 * During regression, numeric attributes are done with only binary splits,
 * finding the split that minimizes the total squared error sum. 
 * 
 * @author Edward Raff
 */
public class DecisionStump implements Classifier, Regressor
{
    /**
     * Indicates which attribute to split on 
     */
    private int splittingAttribute;
    /**
     * Used only when trained for classification. Contains information about the class being predicted
     */
    private CategoricalData predicting;
    /**
     * Contains the information about the attributes in the data set
     */
    private CategoricalData[] catAttributes;
    /**
     * Used only in classification. Contains the numeric boundaries to split on
     */
    private List<Double> boundries;
    /**
     * Used only in classification. Contains the most likely class corresponding to each boundary split 
     */
    private List<Integer> owners;
    /**
     * Used only in classification. Contains the results for each of the split options 
     */
    private CategoricalResults[] results;
    /**
     * Only used during regression. Contains the averages for each branch
     */
    private double[] regressionResults;
    private GainMethod gainMethod;
    private boolean removeContinuousAttributes;

    /**
     * Creates a new decision stump
     */
    public DecisionStump()
    {
        gainMethod = GainMethod.GAINRATIO;
        removeContinuousAttributes = false;
    }

    /**
     * Unlike categorical values, when a continuous attribute is selected to split on, not 
     * all values of the attribute become the same. It can be useful to split on the same 
     * attribute multiple times. If set true, continuous attributes will be removed from 
     * the options list. Else, they will be left in the options list. 
     * 
     * @param removeContinuousAttributes whether or not to remove continuous attributes on a call to {@link #trainC(java.util.List, java.util.Set) }
     */
    public void setRemoveContinuousAttributes(boolean removeContinuousAttributes)
    {
        this.removeContinuousAttributes = removeContinuousAttributes;
    }
    
    public void setGainMethod(GainMethod gainMethod)
    {
        this.gainMethod = gainMethod;
    }

    public GainMethod getGainMethod()
    {
        return gainMethod;
    }
    
    /**
     * Returns the attribute that this stump has decided to used to compute results. 
     * @return 
     */
    public int getSplittingAttribute()
    {
        return splittingAttribute;
    }

    /**
     * Sets the DecisionStump's predicting information. This will be set automatically
     * by calling {@link #trainC(jsat.classifiers.ClassificationDataSet) } or 
     * {@link #trainC(jsat.classifiers.ClassificationDataSet, java.util.concurrent.ExecutorService) },
     * but it must be called before using {@link #trainC(java.util.List, java.util.Set) }. 
     * 
     * @param predicting the information about the attribute that will be predicted by this classifier
     */
    public void setPredicting(CategoricalData predicting)
    {
        this.predicting = predicting;
    }

    public double regress(DataPoint data)
    {
        if(regressionResults == null)
            throw new RuntimeException("Decusion stump has not been trained for regression");
        return regressionResults[whichPath(data)];
    }

    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        train(dataSet);
    }

    public void train(RegressionDataSet dataSet)
    {
        Set<Integer> options = new HashSet<Integer>(dataSet.getNumFeatures());
        for(int i = 0; i < dataSet.getNumFeatures(); i++)
            options.add(i);
        trainR(dataSet.getDPPList(), options);
    }
    
    public static enum GainMethod
    {
        GAIN, GAINRATIO
    }
    
    /**
     * Computes the entropy of the list of data points. 
     * Each data point is paired with its corresponding class. 
     * 
     * @param dataPoints the list of data points paired with their corresponding class value
     * @return the entropy of this set of data points
     */
    public static double entropy(List<DataPointPair<Integer>> dataPoints)
    {
        //Normaly we would know the number of categories apriori, but to make life easier 
        //on the user we will just add as needed, and wasit a small amount of memory
        //We actually will use less memory when the number of categories is thinned out for large N 
        List<Double> probabilites = new ArrayList<Double>();
        double sumOfWeights = 0.0;
        for(DataPointPair<Integer> dpp : dataPoints)
        {
            while(probabilites.size()-1 < dpp.getPair())//Grow to the number of categories
                probabilites.add(0.0);
            probabilites.set(dpp.getPair(), probabilites.get(dpp.getPair())+dpp.getDataPoint().getWeight());
            sumOfWeights += dpp.getDataPoint().getWeight();
        }
        //Normalize from counts to proabilities 
        for(int i = 0; i < probabilites.size(); i++)
            probabilites.set(i, probabilites.get(i)/sumOfWeights);
        
        
        double entropy = 0.0;
        /*
         * Entropy = 
         *     n
         *  =====
         *  \
         *-  >    p  log/p \
         *  /      i    \ i/
         *  =====
         *  i = 1
         * 
         * and 0 log(0) is taken to be equal to 0
         */
        
        for(Double p : probabilites )
            if(p > 0)
                entropy += p*log(p)/log(2);
        
        return abs(entropy);
    }
    
    /**
     * A value that is just above zero
     */
    private static final double almost0 = 1e-6;
    /**
     * A value that is just below one
     */
    private static final double almost1 = 1.0-almost0;
    
    /**
     * This method finds a value that is the overlap of the two distributions, representing a separation point. 
     * This method works in 3 steps. It first determines if the two distributions have no overlap, and will 
     * return the value in-between the distributions. <br>
     * If there is overlap, it attempts to find the point between the means that marks the overlap <br>
     * If this fails, it attempts to find an overlapping point by starting at the least probable value
     * appearing at either end of the real numbers. <br>
     * <br>
     * This method may fail on some pairs of distributions, especially if the standard deviations are
     * significantly different from each other and have similar means. 
     * 
     * @param dist1 the distribution of values for the first class, may be null so long as the other distribution is not
     * @param dist2 the distribution of values for the second class, may be null so long as the other distribution is not
     * @return an double, indicating the separating  point, and an integer indicating 
     * which class is most likely when on the left. 0 indicates <tt>dist1</tt>, 
     * and 1 indicates <tt>dist2</tt>
     * @throws ArithmeticException if finding the splitting point between the two distributions is non trivial 
     */
    public static PairedReturn<Integer, Double> threshholdSplit(final Distribution dist1, final Distribution dist2)
    {
        if(dist1 == null && dist2 == null)
            throw new ArithmeticException("No Distributions given");
        else if(dist1 == null)
            return new PairedReturn<Integer, Double>(1, Double.POSITIVE_INFINITY);
        else if(dist2 == null)
            return new PairedReturn<Integer, Double>(0, Double.POSITIVE_INFINITY);
        
        double tmp1, tmp2;
        //Special case: no overlap if there is no overlap between the two distributions,we can easily return a seperating value 
        if( (tmp1 = dist1.invCdf(almost0)) >  (tmp2 = dist2.invCdf(almost1) ) )//If dist1 is completly to the right of dist2
            return new PairedReturn<Integer, Double>(1, (tmp1+tmp2)*0.5);
        else if( (tmp1 = dist1.invCdf(almost1)) <  (tmp2 = dist2.invCdf(almost0) ) )//If dist2 is completly to the right of dist1
            return new PairedReturn<Integer, Double>(0, (tmp1+tmp2)*0.5);
        
        //Define a function we would like to find the root of. There may be multiple roots, but we will only use one. 
        Function f = new Function() {

            public double f(double... x)
            {
                return dist1.pdf(x[0]) - dist2.pdf(x[0]);
            }

            public double f(Vec x)
            {
                return dist1.pdf(x.get(0)) - dist2.pdf(x.get(0));
            }
        };
        
        double minRange = Math.min(dist1.mean(), dist2.mean());
        double maxRange = Math.max(dist1.mean(), dist2.mean());
        
        //use zeroin because it can fall back to bisection in bad cases,
        //and it is very likely that this function will have non diferentiable points 
        double split = Double.POSITIVE_INFINITY;
        try
        {
            split = Zeroin.root(1e-8, minRange, maxRange, f, 0.0);
        }
        catch(ArithmeticException ex)//Was not in the range, so we will use the invCDF to find better values
        {
            minRange = Math.min(dist1.invCdf(almost0), dist2.invCdf(almost0));
            maxRange = Math.max(dist1.invCdf(almost1), dist2.invCdf(almost1));
            
            split = Zeroin.root(1e-8, minRange, maxRange, f, 0.0);
        }
        
        
        double minStnd = Math.min(dist1.standardDeviation(), dist2.standardDeviation());
        
        int left = 0;
        if(dist2.pdf(split-minStnd/2) > dist1.pdf(split-minStnd/2))
            left = 1;
        return new PairedReturn<Integer, Double>(left, split);
    }
    
    /**
     * Return null as a failure value, indicating there was no way to compute the result. <br>
     * Else, 2 lists are returned. Each are the same length, and their values are matched up. 
     * The list of doubles is in sorted order. The last element is always positive Infinity. 
     * For index i, the double value at index i indicates that for all values between the 
     * double indices for i and (i-1), is most likely to belong to the class indicated from
     * the integer list for index  i. 
     * 
     * @param dists the distributions for each options
     * @return the paired lists that describe the most probable distribution
     */
    public static PairedReturn<List<Double>, List<Integer>> intersections(final List<Distribution> dists)
    {
        double minRange = Double.MAX_VALUE;
        double maxRange = Double.MIN_VALUE;
        //we choose the step size to be the smallest of the standard deviations, and then divice by a constant
        double stepSize = Double.MAX_VALUE;
        
        final List<Integer> belongsTo = new ArrayList<Integer>();
        final List<Double> splitPoints = new ArrayList<Double>();
        
        for(Distribution cd : dists)
        {
            if(cd == null)
                continue;
            minRange = min(minRange, cd.invCdf(almost0));
            maxRange = max(maxRange, cd.invCdf(almost1));
            double stndDev = cd.standardDeviation();
            if(stndDev > 0 )//zero is a valid standard deviation, we dont want to deal with that! 
                stepSize = min(stepSize, stndDev);
        }
        stepSize/=4;
        //TODO is there a better way to avoid small step sizes? 
        if((maxRange-minRange)/stepSize > 50*dists.size())//Limi to 50*|Dists| iterations 
            stepSize = (maxRange-minRange)/(50*dists.size());
        else if( (maxRange - minRange) == 0.0 || minRange+stepSize == minRange)//Range is too small to search!
            return null;
        
        //First value
        belongsTo.add(maxPDF(dists, minRange));
        double curPos = minRange+stepSize;
        while(curPos <= maxRange)
        {
            final int newMax = maxPDF(dists, curPos);
            if(newMax != belongsTo.get(belongsTo.size()-1))//Change
            {
                //Create a function to use root finding to find the cross over point 
                Function f = new Function() {

                    public double f(double... x)
                    {
                        return dists.get(belongsTo.get(belongsTo.size()-1)).pdf(x[0]) - dists.get(newMax).pdf(x[0]);
                    }

                    public double f(Vec x)
                    {
                        return dists.get(belongsTo.get(belongsTo.size()-1)).pdf(x.get(0)) - dists.get(newMax).pdf(x.get(0));
                    }
                };
                
                double crossOverPoint;
                try//Try and get exact cross over, possible to fail when values are very small - espeically final the distributions are far appart from eachother
                {
                    crossOverPoint = Zeroin.root(almost0, curPos-stepSize, curPos, f, 0.0);
                }
                catch (ArithmeticException ex)
                {
                    crossOverPoint = (curPos*2-stepSize)*0.5;//Rough estimate 
                }
                
                splitPoints.add(crossOverPoint);
                belongsTo.add(newMax);
            }
            curPos += stepSize;
        }
        
        splitPoints.add(Double.POSITIVE_INFINITY);
        
        return new PairedReturn<List<Double>, List<Integer>>(splitPoints, belongsTo);
    }
    
    /**
     * Returns the index of the distribution that has the largest PDF value at the given point.
     * 
     * @param dits the list of distributions to test, null values will be skipped over 
     * @param x the value to test the PDF of each distribution at
     * @return the index of the most likely distribution at the given point
     */
    private static int maxPDF(List<Distribution> dits, double x)
    {
        double maxVal = -1;
        int best = -1;
        for(int i = 0; i < dits.size(); i++)
        {
            if(dits.get(i) == null)
                continue;
            double tmp = dits.get(i).pdf(x);
            if(tmp > maxVal)
            {
                maxVal = tmp;
                best = i;
            }
        }
        
        return best;
    }

    /**
     * Determines which split path this data point would follow from this decision stump. 
     * Works for both classification and regression. 
     * 
     * @param data the data point in question
     * @return the integer indicating which path to take. -1 returned if stump is not trained
     */
    public int whichPath(DataPoint data)
    {
        int paths = getNumberOfPaths();
        if(paths < 0)
            return paths;//Not trained
        else if(paths == 1)//ONLY one option, entropy was zero
            return 0;
        else if(splittingAttribute < catAttributes.length)//Same for classification and regression
            return data.getCategoricalValue(splittingAttribute);
        //else, is Numerical attribute - but regression or classification?
        int numerAttribute = splittingAttribute - catAttributes.length;
        if(results != null)//Categorical!
        {
            int pos = Collections.binarySearch(boundries, data.getNumericalValues().get(numerAttribute));
            pos = pos < 0 ? -pos-1 : pos;
            return owners.get(pos);
        }
        else//Regression! It is trained, it would have been grabed at the top if not
        {
            if(data.getNumericalValues().get(numerAttribute) <= regressionResults[2])
                return 0;
            else
                return 1;
        }
    }
    
    /**
     * Returns the number of paths that this decision stump leads to. The stump may not ever 
     * direct a data point on some of the paths. A result of 1 path means that all data points 
     * will be given the same decision, and is generated when the entropy of a set is 0.0.
     * <br><br>
     * -1 is returned for an untrained stump
     * 
     * @return the number of paths this decision stump has stored
     */
    public int getNumberOfPaths()
    {
        if(results != null)//Categorical!
            return results.length;
        else if(catAttributes != null)//Regression!
            if(splittingAttribute < catAttributes.length)//Categorical
                return catAttributes[splittingAttribute].getNumOfCategories();
            else//Numerical is always binary
                return 2;
        return -1;//Not trained!
    }
    
    public CategoricalResults classify(DataPoint data)
    {
        if(results == null)
            throw new RuntimeException("DecisionStump has not been trained for classification");
        return results[whichPath(data)];
    }
    
    /**
     * Returns the categorical result of the i'th path. 
     * @param i the path to get the result for
     * @return the result that would be returned if a data point went down the given path
     * @throws IndexOutOfBoundsException if an invalid path is given
     * @throws NullPointerException if the stump has not been trained for classification
     */
    public CategoricalResults result(int i)
    {
        if(i < 0 || i >= getNumberOfPaths())
            throw new IndexOutOfBoundsException("Invalid path, can to return a result for path " + i);
        return results[i];
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        Set<Integer> splitOptions = new HashSet<Integer>(dataSet.getNumFeatures());
        for(int i = 0; i < dataSet.getNumFeatures(); i++)
            splitOptions.add(i);
        
        this.predicting = dataSet.getPredicting();
        
        trainC(dataSet.getAsDPPList(), splitOptions);
    }
    
    /**
     * This is a helper function that does the work of training this stump. It may be 
     * called directly by other classes that are creating decision trees to avoid 
     * redundant repackaging of lists. 
     * 
     * @param dataPoints the lists of datapoint to train on, paired with the true category of each training point
     * @param options the set of attributes that this classifier may choose from. The attribute it does choose will be removed from the set. 
     * @return the a list of lists, containing all the datapoints that would have followed each path. Useful for training a decision tree
     */
    public List<List<DataPointPair<Integer>>> trainC(List<DataPointPair<Integer>> dataPoints, Set<Integer> options)
    {
        //TODO remove paths that have zero probability of occuring, so that stumps do not have an inflated branch value 
        if(predicting == null)
            throw new RuntimeException("Predicting value has not been set");
        catAttributes = dataPoints.get(0).getDataPoint().getCategoricalData();
        double entropy =  entropy(dataPoints);
        
        if(entropy == 0.0)//Then all data points belond to the same category!
        {
            results = new CategoricalResults[1];//Only one path! 
            results[0] = new CategoricalResults(predicting.getNumOfCategories());
            results[0].setProb(dataPoints.get(0).getPair(), 1.0);
            List<List<DataPointPair<Integer>>> toReturn = new ArrayList<List<DataPointPair<Integer>>>();
            toReturn.add(dataPoints);
            return toReturn;
        }
        
        double totalSize = 0.0; 
        for(DataPointPair<Integer> dpp :  dataPoints)
            totalSize += dpp.getDataPoint().getWeight();
        
        /**
         * The splitting for the split on the attribute with the best gain
         */
        List<List<DataPointPair<Integer>>> bestSplit = null;
        /**
         * best gain in information we have seen so far 
         */
        double bestGain = -1;
        /**
         * The best attribute to split on
         */
        splittingAttribute = -1;
        for(int attribute :  options)
        {
            List<List<DataPointPair<Integer>>> aSplit;
            PairedReturn<List<Double>, List<Integer>> tmp = null;//Used on numerical attributes
            
            if(attribute < catAttributes.length)//Then we are doing a categorical split
            {
                //Create a list of lists to hold the split variables
                aSplit = listOfLists(catAttributes[attribute].getNumOfCategories());
                
                //Now seperate the values in our current list into their proper split bins 
                for(DataPointPair<Integer> dpp :  dataPoints)
                    aSplit.get(dpp.getDataPoint().getCategoricalValue(attribute)).add(dpp);
            }
            else//Spliting on a numerical value
            {
                attribute -= catAttributes.length;
                int N = predicting.getNumOfCategories();
                
                //This requires more set up and work then just spliting on categories 
                //First we need to seperate class values on the attribute to create distributions to compare
                List<List<Double>> weights = new ArrayList<List<Double>>(N);
                List<List<Double>> values = new ArrayList<List<Double>>(N);
                for(int i = 0; i < N; i++)
                {
                    weights.add(new ArrayList<Double>());
                    values.add(new ArrayList<Double>());
                }
                //Collect values and their weights seperated by class 
                for(DataPointPair<Integer> dpp :  dataPoints)
                {
                    int theClass = dpp.getPair();
                    double value = dpp.getVector().get(attribute);
                    weights.get(theClass).add(dpp.getDataPoint().getWeight());
                    values.get(theClass).add(value);
                }
                //Convert to usable formats 
                Distribution[] dist = new Distribution[N];
                for(int i = 0; i < N; i++)
                {
                    if(weights.get(i).isEmpty())
                    {
                        dist[i] = null;
                        continue;
                    }
                    Vec theVals = new DenseVector(weights.get(i).size());
                    double[] theWeights = new double[theVals.length()];
                    for(int j = 0; j < theWeights.length; j++)
                    {
                        theVals.set(j, values.get(i).get(j));
                        theWeights[j] = weights.get(i).get(j);
                    }
                    dist[i] = new KernelDensityEstimator(theVals, new EpanechnikovKF(), theWeights);
                }
                
                //Now compute the speration boundrys 
                tmp = intersections(Arrays.asList(dist));
                if(tmp == null)
                    continue;
                List<Double> tmpBoundries = tmp.getFirstItem();
                List<Integer> tmpOwners = tmp.getSecondItem();
                
                //Create a list of lists to hold the split variables
                aSplit = listOfLists(N);
                
                //Now seperate the values in our current list into their proper split bins 
                for(DataPointPair<Integer> dpp :  dataPoints)
                {
                    int pos = Collections.binarySearch(tmpBoundries, dpp.getVector().get(attribute));
                    pos = pos < 0 ? -pos-1 : pos;
                    aSplit.get(tmpOwners.get(pos)).add(dpp);
                }
                
                //Fix it back so it can be used below
                attribute+= catAttributes.length;
            }
            
            //Now everything is seperated! 
            
            //Compute gain
            /**
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
            double splitEntropy = 0.0;
            double splitInfo = 0.0;
            for (List<DataPointPair<Integer>> subSet : aSplit)
            {
                double subSetSize = 0.0;
                for(DataPointPair<Integer> dpp : subSet)
                    subSetSize += dpp.getDataPoint().getWeight();
                double SiOverS = subSetSize / totalSize;
                splitEntropy += SiOverS * entropy(subSet);
                
                if(gainMethod == GainMethod.GAINRATIO)
                    if(SiOverS > 0)//log(0)= NaN, but we want it to behave as zero
                        splitInfo += SiOverS * log(SiOverS)/log(2);
            }

            splitInfo = abs(splitInfo);
            if(splitInfo == 0.0)
                splitInfo = 1.0;//Divisino by 1 effects nothing
            double gain= (entropy - splitEntropy)/splitInfo;
            
            if(gain > bestGain)
            {
                bestGain = gain;
                splittingAttribute = attribute;
                bestSplit = aSplit;
                if(attribute >= catAttributes.length)
                {
                    boundries = tmp.getFirstItem();
                    owners = tmp.getSecondItem();
                }
            }
        }
        
        if(bestGain <= 1e-9 || splittingAttribute == -1)//We could not find a good split at all (as good as zero)
        {
            bestSplit = new ArrayList<List<DataPointPair<Integer>>>(1);
            bestSplit.add(dataPoints);
            CategoricalResults badResult = new CategoricalResults(predicting.getNumOfCategories());
            for(DataPointPair<Integer> dpp : dataPoints)
                badResult.incProb(dpp.getPair(), 1.0);
            badResult.normalize();
            results = new CategoricalResults[] {badResult};
            return bestSplit;
        }
        if(splittingAttribute < catAttributes.length || removeContinuousAttributes)
            options.remove(splittingAttribute);
        results = new CategoricalResults[bestSplit.size()];
        for(int i = 0; i < bestSplit.size(); i++)
        {
            results[i] = new CategoricalResults(predicting.getNumOfCategories());
            for(DataPointPair<Integer> dpp : bestSplit.get(i))
                results[i].incProb(dpp.getPair(), dpp.getDataPoint().getWeight());
            results[i].normalize();
        }
        
        return bestSplit;
    }
    
    public List<List<DataPointPair<Double>>> trainR(List<DataPointPair<Double>> dataPoints, Set<Integer> options)
    {
        catAttributes = dataPoints.get(0).getDataPoint().getCategoricalData();
        
        List<List<DataPointPair<Double>>> bestSplit = null;
        double lowestSplitSqrdError = Double.MAX_VALUE;
        
        for(int attribute :  options)
        {
            List<List<DataPointPair<Double>>> thisSplit = null;
            //The squared error for this split 
            double thisSplitSqrdErr = Double.MAX_VALUE;
            //Contains the means of each split 
            double[] thisMeans = null;
            
            if(attribute < catAttributes.length)
            {
                thisSplit = listOfListsD(catAttributes[attribute].getNumOfCategories());
                OnLineStatistics[] stats = new OnLineStatistics[thisSplit.size()];
                for(int i = 0; i < thisSplit.size(); i++)
                    stats[i] = new OnLineStatistics();
                //Now seperate the values in our current list into their proper split bins 
                for(DataPointPair<Double> dpp : dataPoints)
                {
                    int category = dpp.getDataPoint().getCategoricalValue(attribute);
                    thisSplit.get(category).add(dpp);
                    stats[category].add(dpp.getPair(), dpp.getDataPoint().getWeight());
                }
                thisMeans = new double[stats.length];
                thisSplitSqrdErr = 0.0;
                for(int i = 0; i < stats.length; i++)
                {
                    thisSplitSqrdErr += stats[i].getVarance()*stats[i].getSumOfWeights();
                    thisMeans[i] = stats[i].getMean();
                }
            }
            else//Findy a binary split that reduces the variance!
            {
                final int numAttri = attribute - catAttributes.length;
                //We need our list in sorted order by attribute!
                Comparator<DataPointPair<Double>> dppDoubleSorter = new Comparator<DataPointPair<Double>>()
                {
                    public int compare(DataPointPair<Double> o1, DataPointPair<Double> o2)
                    {
                        return Double.compare(o1.getVector().get(numAttri), o2.getVector().get(numAttri));
                    }
                };
                Collections.sort(dataPoints, dppDoubleSorter);
                
                //2 passes, first to sum up the right side, 2nd to move down the grow the left side 
                OnLineStatistics rightSide = new OnLineStatistics();
                OnLineStatistics leftSide = new OnLineStatistics();
                
                for(DataPointPair<Double> dpp : dataPoints)
                    rightSide.add(dpp.getPair(), dpp.getDataPoint().getWeight());
                int bestS = 0;
                thisSplitSqrdErr = rightSide.getVarance()*rightSide.getSumOfWeights();
                
                thisMeans = new double[3];
                
                for(int i = 0; i < dataPoints.size(); i++)
                {
                    DataPointPair<Double> dpp = dataPoints.get(i);
                    double weight = dpp.getDataPoint().getWeight();
                    double val = dpp.getPair();
                    rightSide.remove(val, weight);
                    leftSide.add(val, weight);
                    
                    double tmpSVariance = rightSide.getVarance()*rightSide.getSumOfWeights() 
                            + leftSide.getVarance()*leftSide.getSumOfWeights();
                    if(tmpSVariance < thisSplitSqrdErr && !Double.isInfinite(tmpSVariance))//Infinity can occur once the weights get REALY small
                    {
                        thisSplitSqrdErr = tmpSVariance;
                        bestS = i;
                        thisMeans[0] = leftSide.getMean();
                        thisMeans[1] = rightSide.getMean();
                        //Third spot contains the split value!
                        thisMeans[2] = (dataPoints.get(bestS).getVector().get(numAttri) 
                                + dataPoints.get(bestS+1).getVector().get(numAttri))/2.0;
                    }
                }
                //Now we have the binary split that minimizes the variances of the 2 sets, 
                thisSplit = listOfListsD(2);
                thisSplit.get(0).addAll(dataPoints.subList(0, bestS+1));
                thisSplit.get(1).addAll(dataPoints.subList(bestS+1, dataPoints.size()));
            }
            //Now compare what weve done
            if(thisSplitSqrdErr < lowestSplitSqrdError)
            {
                lowestSplitSqrdError = thisSplitSqrdErr;
                bestSplit = thisSplit;
                splittingAttribute = attribute;
                regressionResults = thisMeans;
            }
        }
        
        //Removal of attribute from list if needed
        if(splittingAttribute < catAttributes.length || removeContinuousAttributes)
            options.remove(splittingAttribute);
        
        
        return bestSplit;
    }
    
    private static List<List<DataPointPair<Integer>>> listOfLists(int n )
    {
        List<List<DataPointPair<Integer>>> aSplit =
                new ArrayList<List<DataPointPair<Integer>>>(n);
        for (int i = 0; i < n; i++)
            aSplit.add(new ArrayList<DataPointPair<Integer>>());
        return aSplit;
    }
    
    private static List<List<DataPointPair<Double>>> listOfListsD(int n )
    {
        List<List<DataPointPair<Double>>> aSplit =
                new ArrayList<List<DataPointPair<Double>>>(n);
        for (int i = 0; i < n; i++)
            aSplit.add(new ArrayList<DataPointPair<Double>>());
        return aSplit;
    }

    public boolean supportsWeightedData()
    {
        return true;
    }

    @Override
    public DecisionStump clone()
    {
        DecisionStump copy = new DecisionStump();
        if(this.catAttributes != null)
            copy.catAttributes = CategoricalData.copyOf(catAttributes);
        if(this.results != null)
        {
            copy.results = new CategoricalResults[this.results.length];
            for(int i = 0; i < this.results.length; i++ )
                copy.results[i] = this.results[i].clone();
        }
        copy.removeContinuousAttributes = this.removeContinuousAttributes;
        copy.splittingAttribute = this.splittingAttribute;
        if(this.boundries != null)
            copy.boundries = new ArrayList<Double>(this.boundries);
        if(this.owners != null)
            copy.owners = new ArrayList<Integer>(this.owners);
        if(this.predicting != null)
            copy.predicting = this.predicting.clone();
        if(regressionResults != null)
            copy.regressionResults = Arrays.copyOf(this.regressionResults, this.regressionResults.length);
        
        return copy;
    }
}
