
package jsat.classifiers.calibration;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.FakeExecutor;

/**
 * This abstract class provides the frame work for an algorithm to perform 
 * probability calibration based on the outputs of a base learning algorithm for 
 * binary classification problems. 
 * <br><br>
 * Calibration can be performed directly on output values, though it may cause
 * over-fitting. For this reason, the {@link CalibrationMode} may be set to an 
 * alternative method. 
 * <br><br>
 * The parameters include the calibration parameters, and any parameters that
 * would be returned by the base model. 
 * 
 * @author Edward Raff
 */
public abstract class BinaryCalibration implements Classifier, Parameterized
{

	private static final long serialVersionUID = 2356311701854978890L;
	/**
     * The base classifier to train and calibrate the outputs of
     */
    @ParameterHolder
    protected BinaryScoreClassifier base;
    /**
     * The number of CV folds
     */
    protected int folds = 3;
    /**
     * The proportion of the data set to hold out for calibration
     */
    protected double holdOut = 0.3;
    /**
     * The calibration mode to use
     */
    protected CalibrationMode mode;

    /**
     * Creates a new Binary Calibration object
     * @param base the base learning algorithm 
     * @param mode the calibration mode to use
     */
    public BinaryCalibration(BinaryScoreClassifier base, CalibrationMode mode)
    {
        this.base = base;
        setCalibrationMode(mode);
    }
    
    /**
     * Controls how the scores are obtained for producing a "training set" to 
     * calibrate the output of the underlying model. 
     */
    public static enum CalibrationMode
    {
        /**
         * The naive methods trains the classifier on the whole data set, and 
         * then produces the scores for each training point. This may cause 
         * over fitting. 
         */
        NAIVE,
        /**
         * The model will be trained by cross validation, using the specified 
         * number of {@link #setCalibrationFolds(int) }. The default is 3 folds,
         * where the classifier will be trained on the folds not in the set, and
         * then produce scores for the unobserved test points in the held out 
         * fold. 
         */
        CV,
        /**
         * The model will have a random {@link #setCalibrationHoldOut(double) 
         * fraction} of the data set held out, and trained on the rest of the 
         * data. The scores will then be produced for the held out data and used
         * for calibration. 
         */
        HOLD_OUT,
    }
    
    /**
     * Trains the model on the given data set
     * @param train the data set to train on
     * @param threadPool the source of threads, may be null
     */
    private void train(ClassificationDataSet train, ExecutorService threadPool)
    {
        if(threadPool == null || threadPool instanceof FakeExecutor)
            base.trainC(train);
        else
            base.trainC(train, threadPool);
    }
    
    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        double[] deci = new double[dataSet.getSampleSize()];//array of SVM decision values
        boolean[] label = new boolean[deci.length];//array of booleans: is the example labeled +1?
        int len = label.length;
        
        if (mode == CalibrationMode.CV)
        {
            List<ClassificationDataSet> foldList = dataSet.cvSet(folds);
            int pos = 0;
            for(int i = 0; i < foldList.size(); i++)
            {
                ClassificationDataSet test = foldList.get(i);
                ClassificationDataSet train = ClassificationDataSet.comineAllBut(foldList, i);
                train(train, threadPool);
                
                for(int j = 0; j < test.getSampleSize(); j++)
                {
                    deci[pos] = base.getScore(test.getDataPoint(j));
                    label[pos] = test.getDataPointCategory(j) == 1;
                    pos++;
                }
            }
            
            train(dataSet, threadPool);
        }
        else if (mode == CalibrationMode.HOLD_OUT)
        {
            List<DataPointPair<Integer>> wholeSet = dataSet.getAsDPPList();
            Collections.shuffle(wholeSet);
            
            int splitMark = (int) (wholeSet.size()*(1-holdOut));
            ClassificationDataSet train = new ClassificationDataSet(wholeSet.subList(0, splitMark), dataSet.getPredicting());
            ClassificationDataSet test = new ClassificationDataSet(wholeSet.subList(splitMark, wholeSet.size()), dataSet.getPredicting());
            
            train(train, threadPool);
            for(int i = 0; i < test.getSampleSize(); i++)
            {
                deci[i] = base.getScore(test.getDataPoint(i));
                label[i] = test.getDataPointCategory(i) == 1;
            }
            
            len = test.getSampleSize();
            
            train(dataSet, threadPool);
        }
        else
        {
            train(dataSet, threadPool);

            for (int i = 0; i < len; i++)
            {
                DataPoint dp = dataSet.getDataPoint(i);
                deci[i] = base.getScore(dp);
                label[i] = dataSet.getDataPointCategory(i) == 1;
            }
        }
        
        calibrate(label, deci, len);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
    }
    
    /**
     * This method perform the model calibration on the outputs verse the class 
     * labels. 
     * @param label the set of labels, where {@code true} indicates the positive
     * class label, and {@code false} indicates the negative class label. 
     * @param scores the score associated with each label from the learning 
     * algorithm. 
     * @param len the number of values (from zero) of the label and scores array
     * to use. This value may be less than the actual array size 
     */
    abstract protected void calibrate(boolean[] label, double[] scores, final int len);
    
    /**
     * If the calibration mode is set to {@link CalibrationMode#CV}, this 
     * controls how many folds of cross validation will be used. The default is
     * 3. 
     * @param folds the number of cross validation folds to perform
     */
    public void setCalibrationFolds(int folds)
    {
        if(folds < 1)
            throw new IllegalArgumentException("Folds must be a positive value, not " + folds);
        this.folds = folds;
    }

    /**
     * Returns the number of cross validation folds to use
     * @return the number of cross validation folds to use
     */
    public int getCalibrationFolds()
    {
        return folds;
    }

    /**
     * If the calibration mode is set to {@link CalibrationMode#HOLD_OUT}, this 
     * what portion of the data set is randomly selected to be the hold out set. 
     * The default is 0.3. 
     * 
     * @param holdOut the portion in (0, 1) to hold out
     */
    public void setCalibrationHoldOut(double holdOut)
    {
        if(Double.isNaN(holdOut) || holdOut <= 0 || holdOut >= 1)
            throw new IllegalArgumentException("HoldOut must be in (0, 1), not " + holdOut);
        this.holdOut = holdOut;
    }

    /**
     * Returns the portion of the data set that will be held out for calibration
     * @return the portion of the data set that will be held out for calibration
     */
    public double getCalibrationHoldOut()
    {
        return holdOut;
    }

    /**
     * Sets which calibration mode will be used during training
     * @param mode the calibration mode to use during training. 
     */
    public void setCalibrationMode(CalibrationMode mode)
    {
        this.mode = mode;
    }

    /**
     * Returns the calibration mode used during training
     * @return the calibration mode used during training
     */
    public CalibrationMode getCalibrationMode()
    {
        return mode;
    }

    @Override
    abstract public BinaryCalibration clone();
    
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
