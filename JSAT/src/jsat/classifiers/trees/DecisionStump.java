
package jsat.classifiers.trees;

import java.util.Set;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.distributions.ContinousDistribution;
import jsat.distributions.empirical.KernelDensityEstimator;
import jsat.distributions.empirical.kernelfunc.EpanechnikovKF;
import jsat.distributions.empirical.kernelfunc.GaussKF;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.rootfinding.Zeroin;
import jsat.utils.PairedReturn;
import static java.lang.Math.*;

/**
 * This class is a 1-rule. It creates one rule that is used to classify all inputs, 
 * making it a decision tree with only one node. It can be used as a weak learner 
 * for ensemble learners, or as the nodes in a true decision tree. 
 * 
 * @author Edward Raff
 */
public class DecisionStump implements Classifier
{
    private int splittingAttribute;
    private CategoricalData predicting;
    private CategoricalData[] catAttributes;
    private List<Double> boundries;
    private List<Integer> owners;
    private CategoricalResults[] results;
    private GainMethod gainMethod;

    /**
     * Creates a new decision stump
     */
    public DecisionStump()
    {
        gainMethod = GainMethod.GAINRATIO;
    }

    public void setGainMethod(GainMethod gainMethod)
    {
        this.gainMethod = gainMethod;
    }

    public GainMethod getGainMethod()
    {
        return gainMethod;
    }
    
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
     * 
     * @param dist1 the distribution of values for the first class
     * @param dist2 the distribution of values for the second class
     * @return an double, indicating the separating  point, and an integer indicating 
     * which class is most likely when on the left. 0 indicates <tt>dist1</tt>, 
     * and 1 indicates <tt>dist2</tt>
     */
    public static PairedReturn<Integer, Double> threshholdSplit(final ContinousDistribution dist1, final ContinousDistribution dist2)
    {
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
        
        /*
         * We use the inverse CDF to find a practical starting point to each function.
         * This is the left most value, so we take the largest of the two - since we want to find the zero point of their intersection, as it is the first point that the 2 distributinos overlap 
         */
        double minRange = Math.min(dist1.mean(), dist2.mean());
        double maxRange = Math.max(dist1.mean(), dist2.mean());
        
        //use zeroin because it can fall back to bisection in bad cases,
        //and it is very likely that this function will have non diferentiable points 
        double split = Zeroin.root(1e-8, minRange, maxRange, f, 0.0);
        
        double minStnd = Math.min(dist1.standardDeviation(), dist2.standardDeviation());
        
        int left = 0;
        if(dist2.pdf(split-minStnd/2) > dist1.pdf(split-minStnd/2))
            left = 1;
        return new PairedReturn<Integer, Double>(left, split);
    }
    
    public static PairedReturn<List<Double>, List<Integer>> intersections(final List<ContinousDistribution> dists)
    {
        double minRange = Double.MAX_VALUE;
        double maxRange = Double.MIN_VALUE;
        //we choose the step size to be the smallest of the standard deviations, and then divice by a constant
        double stepSize = Double.MAX_VALUE;
        for(ContinousDistribution cd : dists)
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
        
        final List<Integer> belongsTo = new ArrayList<Integer>();
        final List<Double> splitPoints = new ArrayList<Double>();
        
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
    private static int maxPDF(List<ContinousDistribution> dits, double x)
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
     * @param data the data point in question
     * @return the integer indicating which path to take. 
     */
    public int whichPath(DataPoint data)
    {
        if(getNumberOfPaths() == 1)//ONLY one option, entropy was zero
            return 0;
        else if(splittingAttribute < catAttributes.length)
            return data.getCategoricalValue(splittingAttribute);
        //else, is Numerical attribute 
        int pos = Collections.binarySearch(boundries, data.getNumericalValues().get(splittingAttribute-catAttributes.length));
        pos = pos < 0 ? -pos-1 : pos;
        return owners.get(pos);
    }
    
    /**
     * Returns the number of paths that this decision stump leads to. The stump may not ever 
     * direct a data point on some of the paths. A result of 1 path means that all data points 
     * will be given the same decision, and is generated when the entropy of a set is 0.0 
     * 
     * @return the number of paths this decision stump has stored
     */
    public int getNumberOfPaths()
    {
        return results.length;
    }
    
    public CategoricalResults classify(DataPoint data)
    {
        return results[whichPath(data)];
    }
    
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
        Set<Integer> splitOptions = new HashSet<Integer>();
        for(int i = 0; i < dataSet.getNumCategoricalVars() + dataSet.getNumNumericalVars(); i++)
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
                ContinousDistribution[] dist = new ContinousDistribution[N];
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
        
        //Now that we know the best attribute, we remove it from the options and compute resutls
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
    
    private static List<List<DataPointPair<Integer>>> listOfLists(int n )
    {
        List<List<DataPointPair<Integer>>> aSplit =
                new ArrayList<List<DataPointPair<Integer>>>(n);
        for (int i = 0; i < n; i++)
            aSplit.add(new ArrayList<DataPointPair<Integer>>());
        return aSplit;
    }

    public boolean supportsWeightedData()
    {
        return true;
    }

    public Classifier clone()
    {
        DecisionStump copy = new DecisionStump();
        copy.catAttributes = CategoricalData.copyOf(catAttributes);
        copy.results = new CategoricalResults[this.results.length];
        for(int i = 0; i < this.results.length; i++ )
            copy.results[i] = this.results[i].clone();
        copy.splittingAttribute = this.splittingAttribute;
        if(this.boundries != null)
            copy.boundries = new ArrayList<Double>(this.boundries);
        if(this.owners != null)
            copy.owners = new ArrayList<Integer>(this.owners);
        copy.predicting = this.predicting.clone();
        return copy;
    }
}
