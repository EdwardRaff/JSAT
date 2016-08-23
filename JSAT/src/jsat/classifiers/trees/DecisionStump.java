
package jsat.classifiers.trees;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;

import jsat.classifiers.*;
import jsat.classifiers.trees.ImpurityScore.ImpurityMeasure;
import jsat.exceptions.FailedToFitException;
import jsat.linear.Vec;
import jsat.math.OnLineStatistics;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.*;
import jsat.utils.concurrent.AtomicDouble;

/**
 * This class is a 1-rule. It creates one rule that is used to classify all inputs, 
 * making it a decision tree with only one node. It can be used as a weak learner 
 * for ensemble learners, or as the nodes in a true decision tree. 
 * <br><br>
 * Categorical values are handled similarly under all circumstances. <br>
 * During classification, numeric attributes are separated based on most 
 * likely probability into their classes. <br>
 * During regression, numeric attributes are done with only binary splits,
 * finding the split that minimizes the total squared error sum. <br>
 * <br>
 * The Decision Stump supports missing values in training and prediction. 
 * 
 * @author Edward Raff
 */
public class DecisionStump implements Classifier, Regressor, Parameterized
{

    private static final long serialVersionUID = -2849268862089019514L;
    
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
     * The number of numeric features in the dataset that this Stump was trained from
     */
    private int numNumericFeatures;
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
     * How much of the data went to each path  
     */
    protected double[] pathRatio;
    /**
     * Only used during regression. Contains the averages for each branch in 
     * the first and 2nd index. 3rd index contains the split value. 
     * If no split could be done, the length is zero and it contains only the 
     * return value
     */
    private double[] regressionResults;
    private ImpurityMeasure gainMethod;
    private boolean removeContinuousAttributes;
    /**
     * The minimum number of points that must be inside the split result for a 
     * split to occur.
     */
    private int minResultSplitSize = 10;

    /**
     * Creates a new decision stump
     */
    public DecisionStump()
    {
        gainMethod = ImpurityMeasure.INFORMATION_GAIN_RATIO;
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
    
    public void setGainMethod(ImpurityMeasure gainMethod)
    {
        this.gainMethod = gainMethod;
    }

    public ImpurityMeasure getGainMethod()
    {
        return gainMethod;
    }
    
    /**
     *
     * @return The number of numeric features in the dataset that this Stump was
     * trained from
     */
    protected int numNumeric()
    {
        return numNumericFeatures;
    }

    /**
     *
     * @return the number of categorical features in the dataset that this Stump
     * was trained from.
     */
    protected int numCategorical()
    {
        return catAttributes.length;
    }


    /**
     * When a split is made, it may be that outliers cause the split to 
     * segregate a minority of points from the majority. The min result split
     * size parameter specifies the minimum allowable number of points to end up
     * in one of the splits for it to be admisible for consideration. 
     * 
     * @param minResultSplitSize the minimum result split size to use
     */
    public void setMinResultSplitSize(int minResultSplitSize)
    {
        if(minResultSplitSize <= 1)
            throw new ArithmeticException("Min split size must be a positive value ");
        this.minResultSplitSize = minResultSplitSize;
    }

    /**
     * Returns the minimum result split size that may be considered for use as 
     * the attribute to split on. 
     * 
     * @return the minimum result split size in use
     */
    public int getMinResultSplitSize()
    {
        return minResultSplitSize;
    }
    
    /**
     * Returns the attribute that this stump has decided to use to compute
     * results. Numeric features start from 0, and categorical features start
     * from the number of numeric features.
     *
     * @return the attribute that this stump has decided to use to compute results.
     */
    public int getSplittingAttribute()
    {
        //TODO refactor the splittingAttribute to just be in this order already
        if(splittingAttribute < catAttributes.length)//categorical feature
            return numNumericFeatures+splittingAttribute;
        //else, is Numerical attribute
        int numerAttribute = splittingAttribute - catAttributes.length;
        return numerAttribute;
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

    @Override
    public double regress(DataPoint data)
    {
        if(regressionResults == null)
            throw new RuntimeException("Decusion stump has not been trained for regression");
        int path = whichPath(data);
        if(path >= 0)
            return regressionResults[path];
        //else, was missing, average
        double avg = 0;
        for(int i = 0; i < pathRatio.length; i++)
            avg += pathRatio[i]*regressionResults[i];
        return avg;
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, new FakeExecutor());
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        Set<Integer> options = new IntSet(dataSet.getNumFeatures());
        for(int i = 0; i < dataSet.getNumFeatures(); i++)
            options.add(i);
        List<List<DataPointPair<Double>>> split = trainR(dataSet.getDPPList(), options, threadPool);
        if(split == null)
            throw new FailedToFitException("Tree could not be fit, make sure your data is good. Potentially file a bug");
    }

    /**
     * From the score for the original set that is being split, this computes 
     * the gain as the improvement in classification from the original split. 
     * @param origScore the score of the unsplit set
     * @param aSplit the splitting of the data points 
     * @return the gain score for this split 
     */
    protected double getGain(ImpurityScore origScore, List<List<DataPointPair<Integer>>> aSplit)
    {
        
        ImpurityScore[] scores = getSplitScores(aSplit);
       
        return ImpurityScore.gain(origScore, scores);
    }

    private ImpurityScore[] getSplitScores(List<List<DataPointPair<Integer>>> aSplit)
    {
        ImpurityScore[] scores = new ImpurityScore[aSplit.size()];
        for(int i = 0; i < aSplit.size(); i++)
            scores[i] = getClassGainScore(aSplit.get(i));
        return scores;
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
        double val = data.getNumericalValues().get(numerAttribute);
        if(Double.isNaN(val))
            return -1;//missing
        
        if (results != null)//Categorical!
        {
            int pos = Collections.binarySearch(boundries, val);
            pos = pos < 0 ? -pos-1 : pos;
            return owners.get(pos);
        }
        else//Regression! It is trained, it would have been grabed at the top if not
        {
            if(regressionResults.length == 1)
                return 0;
            else if(val <= regressionResults[2])
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
            if(regressionResults.length == 1)
                return 1;
            else if(splittingAttribute < catAttributes.length)//Categorical
                return catAttributes[splittingAttribute].getNumOfCategories();
            else//Numerical is always binary
                return 2;
        return Integer.MIN_VALUE;//Not trained!
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(results == null)
            throw new RuntimeException("DecisionStump has not been trained for classification");
        int path = whichPath(data);
        if(path >= 0)
            return results[path];
        else//missing value case, so average
        {
            Vec tmp = results[0].getVecView().clone();
            tmp.mutableMultiply(pathRatio[0]);
            for(int i = 1; i < results.length; i++)
                tmp.mutableAdd(pathRatio[i], results[i].getVecView());
            return new CategoricalResults(tmp.arrayCopy());
        }
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

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        Set<Integer> splitOptions = new IntSet(dataSet.getNumFeatures());
        for(int i = 0; i < dataSet.getNumFeatures(); i++)
            splitOptions.add(i);
        
        this.predicting = dataSet.getPredicting();
        
        trainC(dataSet.getAsDPPList(), splitOptions, threadPool);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
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
        return trainC(dataPoints, options, null);
    }
    
    public List<List<DataPointPair<Integer>>> trainC(final List<DataPointPair<Integer>> dataPoints, Set<Integer> options, ExecutorService ex)
    {
        //TODO remove paths that have zero probability of occuring, so that stumps do not have an inflated branch value 
        if(predicting == null)
            throw new RuntimeException("Predicting value has not been set");
        if(ex == null)
            ex = new FakeExecutor();
        catAttributes = dataPoints.get(0).getDataPoint().getCategoricalData();
        numNumericFeatures = dataPoints.get(0).getVector().length();
        final ImpurityScore origScoreObj = getClassGainScore(dataPoints);
        double origScore =  origScoreObj.getScore();
        
        if(origScore == 0.0 || dataPoints.size() < minResultSplitSize*2)//Then all data points belond to the same category!
        {
            results = new CategoricalResults[1];//Only one path! 
            results[0] = new CategoricalResults(predicting.getNumOfCategories());
            results[0].setProb(dataPoints.get(0).getPair(), 1.0);
            pathRatio = new double[]{0};
            List<List<DataPointPair<Integer>>> toReturn = new ArrayList<List<DataPointPair<Integer>>>();
            toReturn.add(dataPoints);
            return toReturn;
        }
        
        
        
        /**
         * The splitting for the split on the attribute with the best gain
         */
        final List<List<DataPointPair<Integer>>> bestSplit = Collections.synchronizedList(new ArrayList<List<DataPointPair<Integer>>>());
        /**
         * best gain in information we have seen so far 
         */
        final AtomicDouble bestGain = new AtomicDouble(-1);
        final DoubleList bestRatio = new DoubleList();
        /**
         * The best attribute to split on
         */
        splittingAttribute = -1;
        final CountDownLatch latch = new CountDownLatch(options.size());
        
        
        final ThreadLocal<List<DataPointPair<Integer>>> localList = new ThreadLocal<List<DataPointPair<Integer>>>(){

            @Override
            protected List<DataPointPair<Integer>> initialValue()
            {
                return new ArrayList<DataPointPair<Integer>>(dataPoints);
            }
            
        };
        
        for(final int attribute_to_consider :  options)
        {
            ex.submit(new Runnable()
            {

                @Override
                public void run()
                {
                    List<DataPointPair<Integer>> DPs = localList.get();
                    int attribute = attribute_to_consider;
                    final double[] gainRet = new double[]{Double.NaN};
                    gainRet[0] = Double.NaN;
                    List<List<DataPointPair<Integer>>> aSplit;
                    PairedReturn<List<Double>, List<Integer>> tmp = null;//Used on numerical attributes

                    ImpurityScore[] split_scores = null;//used for cat
                    double weightScale = 1.0;

                    if(attribute < catAttributes.length)//Then we are doing a categorical split
                    {
                        //Create a list of lists to hold the split variables
                        aSplit = listOfLists(catAttributes[attribute].getNumOfCategories());
                        split_scores = new ImpurityScore[aSplit.size()];
                        for(int i=0; i < split_scores.length; i++)
                            split_scores[i] = new ImpurityScore(predicting.getNumOfCategories(), gainMethod);

                        List<DataPointPair<Integer>> wasMissing = new ArrayList<DataPointPair<Integer>>(0);
                        double missingSum = 0.0;
                        //Now seperate the values in our current list into their proper split bins 
                        for(DataPointPair<Integer> dpp :  DPs)
                        {
                            int val = dpp.getDataPoint().getCategoricalValue(attribute);
                            double weight = dpp.getDataPoint().getWeight();
                            if(val >= 0)
                            {
                                aSplit.get(val).add(dpp);
                                split_scores[val].addPoint(weight, dpp.getPair());
                            }
                            else
                            {
                                wasMissing.add(dpp);
                                missingSum += weight;
                            }

                        }

                        int pathsTaken = 0;
                        for(List<DataPointPair<Integer>> split : aSplit)
                            if(!split.isEmpty())
                                pathsTaken++;
                        if(pathsTaken <= 1)//not a good path, avoid looping on this junk. Can be caused by missing data
                        {
                            latch.countDown();
                            return;
                        }

                        if(missingSum > 0)//move missing values into others
                        {
                            double newSum = (origScoreObj.getSumOfWeights()-missingSum);
                            weightScale = newSum/origScoreObj.getSumOfWeights();
                            double[] fracs = new double[split_scores.length];
                            for(int i = 0; i < fracs.length; i++)
                                fracs[i] = split_scores[i].getSumOfWeights()/newSum;

                            distributMissing(aSplit, fracs, wasMissing);
                        }
                    }
                    else//Spliting on a numerical value
                    {
                        attribute -= catAttributes.length;
                        int N = predicting.getNumOfCategories();

                        //Create a list of lists to hold the split variables
                        aSplit = listOfLists(2);//Size at least 2
                        split_scores = new ImpurityScore[2];
                        tmp = createNumericCSplit(DPs, N, attribute, aSplit, 
                                origScoreObj, gainRet, split_scores);
                        if(tmp == null)
                        {
                            latch.countDown();
                            return;
                        }

                        //Fix it back so it can be used below
                        attribute+= catAttributes.length;
                    }

                    //Now everything is seperated!
                    double gain;//= Double.isNaN(gainRet[0]) ?  : gainRet[0];
                    if(!Double.isNaN(gainRet[0]))
                        gain = gainRet[0];
                    else 
                    {
                        if(split_scores == null)
                            split_scores = getSplitScores(aSplit);
                        gain = ImpurityScore.gain(origScoreObj, weightScale, split_scores);
                    }

                    if(gain > bestGain.get())
                    {
                        synchronized(bestRatio)
                        {
                            if(gain > bestGain.get())//double check incase changed
                            {
                                bestGain.set(gain);
                                splittingAttribute = attribute;
                                bestSplit.clear();
                                bestSplit.addAll(aSplit);
                                bestRatio.clear();

                                double sum = 1e-8;
                                for(int i = 0; i < split_scores.length; i++)
                                {
                                    sum += split_scores[i].getSumOfWeights();
                                    bestRatio.add(split_scores[i].getSumOfWeights());
                                }
                                for(int i = 0; i < split_scores.length; i++)
                                    bestRatio.set(i, bestRatio.getD(i)/sum);

                                if(attribute >= catAttributes.length)
                                {
                                    boundries = tmp.getFirstItem();
                                    owners = tmp.getSecondItem();
                                }
                            }
                        }
                    }
                    
                    latch.countDown();
                }
            });
        }
        
        try
        {
            latch.await();
        }
        catch (InterruptedException ex1)
        {
            Logger.getLogger(DecisionStump.class.getName()).log(Level.SEVERE, null, ex1);
            throw new FailedToFitException(ex1);
        }
        
        if(bestGain.get() <= 1e-9 || splittingAttribute == -1)//We could not find a good split at all (as good as zero)
        {
            bestSplit.clear();
            bestSplit.add(dataPoints);
            CategoricalResults badResult = new CategoricalResults(predicting.getNumOfCategories());
            for(DataPointPair<Integer> dpp : dataPoints)
                badResult.incProb(dpp.getPair(), 1.0);
            badResult.normalize();
            results = new CategoricalResults[] {badResult};
            pathRatio = new double[]{1};
            return bestSplit;
        }
        if(splittingAttribute < catAttributes.length || removeContinuousAttributes)
            options.remove(splittingAttribute);
        results = new CategoricalResults[bestSplit.size()];
        pathRatio = bestRatio.getVecView().arrayCopy();
        for(int i = 0; i < bestSplit.size(); i++)
        {
            results[i] = new CategoricalResults(predicting.getNumOfCategories());
            for(DataPointPair<Integer> dpp : bestSplit.get(i))
                results[i].incProb(dpp.getPair(), dpp.getDataPoint().getWeight());
            results[i].normalize();
        }
        
        return bestSplit;
    }
    
    /**
     * 
     * @param dataPoints the original list of data points 
     * @param N number of predicting target options
     * @param attribute the numeric attribute to try and find a split on
     * @param aSplit the list of lists to place the results of splitting in
     * @param origScore the score value for the data set we are splitting
     * @param finalGain array used to reference a double that can be returned. 
     * If this method determined the gain in order to find the split, it sets 
     * the value at index zero to the gain it computed. May be null, in which 
     * case it is ignored. 
     * @return A pair of lists of the same size. The list of doubles containing 
     * the split boundaries, and the integers containing the path number. 
     * Multiple splits could go down the same path. 
     */
    private PairedReturn<List<Double>, List<Integer>> createNumericCSplit(
            List<DataPointPair<Integer>> dataPoints, int N, final int attribute,
            List<List<DataPointPair<Integer>>> aSplit, ImpurityScore origScore, double[] finalGain, ImpurityScore[] subScores)
    {
        //cache misses are killing us, move data into a double[] to get more juice!
        double[] vals = new double[dataPoints.size()];//TODO put this in a thread local somewhere and re-use
        int wasNaN = 0;
        for(int i = 0; i < dataPoints.size()-wasNaN; i++)
        {
            double val = dataPoints.get(i).getVector().get(attribute);
            if(!Double.isNaN(val))
                vals[i] = val;
            else
            {
                Collections.swap(dataPoints, vals.length-wasNaN-1, i);
                wasNaN++;
                i--;//go back and do this one again!
            }
        }
        //do what i want!
        Collection<List<?>> paired = (Collection<List<?>> )(Collection<?> )Arrays.asList(dataPoints);
        QuickSort.sort(vals, 0, vals.length-wasNaN, paired );//sort the numeric values and put our original list of data points in the correct order at the same time

        double bestGain = Double.NEGATIVE_INFINITY;
        double bestSplit = Double.NEGATIVE_INFINITY;
        int splitIndex = -1;

        ImpurityScore rightSide = origScore.clone();
        ImpurityScore leftSide = new ImpurityScore(N, gainMethod);
        //remove any Missing Value nodes from considering from the start 
        double nanWeightRemoved = 0;
        for(int i = dataPoints.size()-wasNaN; i < dataPoints.size(); i++)
        {
            double weight = dataPoints.get(i).getDataPoint().getWeight();
            int truth = dataPoints.get(i).getPair();

            nanWeightRemoved += weight;
            rightSide.removePoint(weight, truth);
        }
        double wholeRescale = rightSide.getSumOfWeights()/(rightSide.getSumOfWeights()+nanWeightRemoved);

        for(int i = 0; i < minResultSplitSize; i++)
        {
            if(i >= dataPoints.size())
                System.out.println("WHAT?");
            double weight = dataPoints.get(i).getDataPoint().getWeight();
            int truth = dataPoints.get(i).getPair();

            leftSide.addPoint(weight, truth);
            rightSide.removePoint(weight, truth);
        }

        for(int i = minResultSplitSize; i < dataPoints.size()-minResultSplitSize-1-wasNaN; i++)
        {
            DataPointPair<Integer> dpp = dataPoints.get(i);
            rightSide.removePoint(dpp.getDataPoint(), dpp.getPair());
            leftSide.addPoint(dpp.getDataPoint(), dpp.getPair());
            double leftVal = vals[i];
            double rightVal = vals[i+1];
            if( (rightVal-leftVal) < 1e-14 )//Values are too close!
                continue;

            subScores[0] = leftSide;
            subScores[1] = rightSide;
            double curGain = ImpurityScore.gain(origScore, wholeRescale, leftSide, rightSide);

            if(curGain >= bestGain)
            {
                double curSplit = (leftVal + rightVal) / 2;
                bestGain = curGain;
                bestSplit = curSplit;
                splitIndex = i+1;
            }
        }
        if(splitIndex == -1)
            return null;

        if(finalGain != null)
            finalGain[0] = bestGain;
        aSplit.set(0, new ArrayList<DataPointPair<Integer>>(dataPoints.subList(0, splitIndex)));
        aSplit.set(1, new ArrayList<DataPointPair<Integer>>(dataPoints.subList(splitIndex, dataPoints.size()-wasNaN)));
        if(wasNaN > 0)
        {
            double weightScale = leftSide.getSumOfWeights()/(leftSide.getSumOfWeights() + rightSide.getSumOfWeights()+0.0);
            distributMissing(aSplit, new double[]{weightScale, 1-weightScale}, dataPoints.subList(dataPoints.size()-wasNaN, dataPoints.size()));
        }
        PairedReturn<List<Double>, List<Integer>> tmp = 
                new PairedReturn<List<Double>, List<Integer>>(
                Arrays.asList(bestSplit, Double.POSITIVE_INFINITY),
                Arrays.asList(0, 1));

        return tmp;
        
    }
    
    /**
     * Distributes a list of datapoints that had missing values to each split, re-weighted by the indicated fractions 
     * @param splits a list of lists, where each inner list is a split
     * @param hadMissing the list of datapoints that had missing values
     */
    static protected <T> void distributMissing(List<List<DataPointPair<T>>> splits, List<DataPointPair<T>> hadMissing)
    {
        double[] fracs = new double[splits.size()];
        for(int i = 0; i < splits.size(); i++)
            for(DataPointPair<T> dpp : splits.get(i))
                fracs[i] += dpp.getDataPoint().getWeight();
        double sum = 0;
        for(double d : fracs)
            sum += d;
        for(int i = 0; i < fracs.length; i++)
            fracs[i] /= sum;
        distributMissing(splits, fracs, hadMissing);
    }
    
    /**
     * Distributes a list of datapoints that had missing values to each split, re-weighted by the indicated fractions 
     * @param splits a list of lists, where each inner list is a split
     * @param fracs the fraction of weight to each split, should sum to one
     * @param hadMissing the list of datapoints that had missing values
     */
    static protected <T> void distributMissing(List<List<DataPointPair<T>>> splits, double[] fracs, List<DataPointPair<T>> hadMissing)
    {
        for (DataPointPair<T> dpp : hadMissing)
        {
            DataPoint dp = dpp.getDataPoint();
            Vec vec = dp.getNumericalValues();
            int[] cats = dp.getCategoricalValues();
            CategoricalData[] lab = dp.getCategoricalData();

            for (int i = 0; i < fracs.length; i++)
            {
                double nw = fracs[i] * dp.getWeight();
                if(Double.isNaN(nw))//happens when no weight is available
                    continue;
                if(nw <= 1e-13)
                    continue;
                DataPointPair<T> dp_i = new DataPointPair<T>(new DataPoint(vec, cats, lab, nw), dpp.getPair());
                splits.get(i).add(dp_i);
            }
        }
    }
    
    public List<List<DataPointPair<Double>>> trainR(final List<DataPointPair<Double>> dataPoints, Set<Integer> options)
    {
        return trainR(dataPoints, options, new FakeExecutor());
    }
    
    public List<List<DataPointPair<Double>>> trainR(final List<DataPointPair<Double>> dataPoints, Set<Integer> options, ExecutorService ex)
    {
        catAttributes = dataPoints.get(0).getDataPoint().getCategoricalData();
        numNumericFeatures = dataPoints.get(0).getVector().length();
        //Not enough points for a split to occur
        if(dataPoints.size() <= minResultSplitSize*2)
        {
            splittingAttribute = catAttributes.length;
            regressionResults = new double[1];
            double avg = 0.0;
            double sum = 0.0;
            for(DataPointPair<Double> dpp : dataPoints )
            {
                double weight = dpp.getDataPoint().getWeight();
                avg += dpp.getPair()*weight;
                sum += weight;
            }
            regressionResults[0] = avg/sum;
            
            List<List<DataPointPair<Double>>> toRet = new ArrayList<List<DataPointPair<Double>>>(1);
            toRet.add(dataPoints);
            return toRet;
        }
        
        final List<List<DataPointPair<Double>>> bestSplit = new ArrayList<List<DataPointPair<Double>>>();
        final AtomicDouble lowestSplitSqrdError = new AtomicDouble(Double.MAX_VALUE);
        
        final ThreadLocal<List<DataPointPair<Double>>> localList = new ThreadLocal<List<DataPointPair<Double>>>(){

            @Override
            protected List<DataPointPair<Double>> initialValue()
            {
                return new ArrayList<DataPointPair<Double>>(dataPoints);
            }
            
        };
        
        final CountDownLatch latch = new CountDownLatch(options.size());
        for(int attribute_to_consider :  options)
        {
            final int attribute = attribute_to_consider;
            ex.submit(new Runnable()
            {

                @Override
                public void run()
                {
                    final List<DataPointPair<Double>> DPs =  localList.get();
                    List<List<DataPointPair<Double>>> thisSplit = null;
                    //The squared error for this split 
                    double thisSplitSqrdErr = Double.MAX_VALUE;
                    //Contains the means of each split 
                    double[] thisMeans = null;
                    double[] thisRatio;

                    if(attribute < catAttributes.length)
                    {
                        thisSplit = listOfListsD(catAttributes[attribute].getNumOfCategories());
                        OnLineStatistics[] stats = new OnLineStatistics[thisSplit.size()];
                        thisRatio = new double[thisSplit.size()];
                        for(int i = 0; i < thisSplit.size(); i++)
                            stats[i] = new OnLineStatistics();
                        //Now seperate the values in our current list into their proper split bins 
                        List<DataPointPair<Double>> wasMissing = new ArrayList<DataPointPair<Double>>(0);
                        for(DataPointPair<Double> dpp : DPs)
                        {
                            int category = dpp.getDataPoint().getCategoricalValue(attribute);
                            if(category >= 0)
                            {
                                thisSplit.get(category).add(dpp);
                                stats[category].add(dpp.getPair(), dpp.getDataPoint().getWeight());
                            }
                            else//was negative, missing value
                            {
                                wasMissing.add(dpp);
                            }
                        }
                        thisMeans = new double[stats.length];
                        thisSplitSqrdErr = 0.0;
                        double sum = 0;
                        for(int i = 0; i < stats.length; i++)
                        {
                            sum += (thisRatio[i] = stats[i].getSumOfWeights());
                            thisSplitSqrdErr += stats[i].getVarance()*stats[i].getSumOfWeights();
                            thisMeans[i] = stats[i].getMean();
                        }
                        for(int i = 0; i < stats.length; i++)
                            thisRatio[i] /= sum;

                        if(!wasMissing.isEmpty())
                            distributMissing(thisSplit, thisRatio, wasMissing);
                    }
                    else//Findy a binary split that reduces the variance!
                    {
                        final int numAttri = attribute - catAttributes.length;
                        //We need our list in sorted order by attribute!
                        Comparator<DataPointPair<Double>> dppDoubleSorter = new Comparator<DataPointPair<Double>>()
                        {
                            @Override
                            public int compare(DataPointPair<Double> o1, DataPointPair<Double> o2)
                            {
                                return Double.compare(o1.getVector().get(numAttri), o2.getVector().get(numAttri));
                            }
                        };
                        Collections.sort(DPs, dppDoubleSorter);//this will put nans to the right

                        //2 passes, first to sum up the right side, 2nd to move down the grow the left side 
                        OnLineStatistics rightSide = new OnLineStatistics();
                        OnLineStatistics leftSide = new OnLineStatistics();

                        int nans = 0;
                        for(DataPointPair<Double> dpp : DPs)
                            if(!Double.isNaN(dpp.getVector().get(numAttri)))
                                rightSide.add(dpp.getPair(), dpp.getDataPoint().getWeight());
                            else
                                nans++;

                        int bestS = 0;
                        thisSplitSqrdErr = Double.POSITIVE_INFINITY;

                        final double allWeight = rightSide.getSumOfWeights();
                        thisMeans = new double[3];
                        thisRatio = new double[2];

                        for(int i = 0; i < DPs.size()-nans; i++)
                        {
                            DataPointPair<Double> dpp = DPs.get(i);
                            double weight = dpp.getDataPoint().getWeight();
                            double val = dpp.getPair();
                            rightSide.remove(val, weight);
                            leftSide.add(val, weight);


                            if(i < minResultSplitSize)
                                continue;
                            else if(i > DPs.size()-minResultSplitSize-nans)
                                break;

                            double tmpSVariance = rightSide.getVarance()*rightSide.getSumOfWeights() 
                                    + leftSide.getVarance()*leftSide.getSumOfWeights();
                            if(tmpSVariance < thisSplitSqrdErr && !Double.isInfinite(tmpSVariance))//Infinity can occur once the weights get REALY small
                            {
                                thisSplitSqrdErr = tmpSVariance;
                                bestS = i;
                                thisMeans[0] = leftSide.getMean();
                                thisMeans[1] = rightSide.getMean();
                                //Third spot contains the split value!
                                thisMeans[2] = (DPs.get(bestS).getVector().get(numAttri) 
                                        + DPs.get(bestS+1).getVector().get(numAttri))/2.0;
                                thisRatio[0] = leftSide.getSumOfWeights()/allWeight;
                                thisRatio[1] = rightSide.getSumOfWeights()/allWeight;
                            }
                        }

                        if(DPs.size() - nans >= minResultSplitSize)
                        {
                            //Now we have the binary split that minimizes the variances of the 2 sets, 
                            thisSplit = listOfListsD(2);
                            thisSplit.get(0).addAll(DPs.subList(0, bestS+1));
                            thisSplit.get(1).addAll(DPs.subList(bestS+1, DPs.size()-nans));
                            if(nans > 0)
                                distributMissing(thisSplit, thisRatio, DPs.subList(DPs.size()-nans, DPs.size()));
                        }
                        else//not a good split, we can't trust it
                            thisSplitSqrdErr = Double.NEGATIVE_INFINITY;
                    }
                    
                    //numerical issue check. When we get a REALLy good split, error can be a tiny negative value due to numerical instability. Check and swap sign if small
                    if(Math.abs(thisSplitSqrdErr) < 1e-13)//no need to check sign, make simpler
                        thisSplitSqrdErr = Math.abs(thisSplitSqrdErr);
                    //Now compare what weve done
                    if(thisSplitSqrdErr >= 0 && thisSplitSqrdErr < lowestSplitSqrdError.get())//how did we get -Inf?
                    {
                        synchronized(bestSplit)
                        {
                            if(thisSplitSqrdErr < lowestSplitSqrdError.get())
                            {
                                lowestSplitSqrdError.set(thisSplitSqrdErr);
                                bestSplit.clear();
                                bestSplit.addAll(thisSplit);
                                splittingAttribute = attribute;
                                regressionResults = thisMeans;
                                pathRatio = thisRatio;
                            }
                        }
                    }
                    
                    latch.countDown();
                }
            });
        }
        
        try
        {
            latch.await();
        }
        catch (InterruptedException ex1)
        {
            Logger.getLogger(DecisionStump.class.getName()).log(Level.SEVERE, null, ex1);
            throw new FailedToFitException(ex1);
        }
        
        //Removal of attribute from list if needed
        if(splittingAttribute < catAttributes.length || removeContinuousAttributes)
            options.remove(splittingAttribute);
        
        if(bestSplit.size() == 0)//no good option selected. Keep old behavior, return null in that case
            return null;
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

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }
    
    private ImpurityScore getClassGainScore(List<DataPointPair<Integer>> dataPoints)
    {
        ImpurityScore cgs = new ImpurityScore(predicting.getNumOfCategories(), gainMethod);
        
        for(DataPointPair<Integer> dpp : dataPoints)
            cgs.addPoint(dpp.getDataPoint(), dpp.getPair());
        
        return cgs;
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
            copy.boundries = new DoubleList(this.boundries);
        if(this.owners != null)
            copy.owners = new IntList(this.owners);
        if(this.predicting != null)
            copy.predicting = this.predicting.clone();
        if(regressionResults != null)
            copy.regressionResults = Arrays.copyOf(this.regressionResults, this.regressionResults.length);
        if(pathRatio != null)
            copy.pathRatio = Arrays.copyOf(this.pathRatio, this.pathRatio.length);
        copy.minResultSplitSize = this.minResultSplitSize;
        copy.gainMethod = this.gainMethod;
        copy.numNumericFeatures = this.numNumericFeatures;
        return copy;
    }
    
    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
}
