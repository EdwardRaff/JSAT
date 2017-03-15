
package jsat.classifiers.neuralnetwork;

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.*;
import java.util.concurrent.ExecutorService;

import jsat.classifiers.*;
import jsat.clustering.SeedSelectionMethods;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.linear.vectorcollection.DefaultVectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionFactory;
import jsat.math.decayrates.*;
import jsat.parameters.*;
import jsat.utils.FakeExecutor;
import jsat.utils.random.RandomUtil;

/**
 * Learning Vector Quantization (LVQ) is an algorithm that extends {@link SOM} 
 * to take advantage of label information to perform classification. It creates 
 * a number of representatives, or learning vectors, for each class. The LVs are
 * then updated iteratively to push away from the wrong class and pull closer to
 * the correct class. LVQ is equivalent to a type of 2 layer neural network. 
 * 
 * @author Edward Raff
 */
public class LVQ implements Classifier, Parameterized
{

	private static final long serialVersionUID = -3911765006048793222L;
	/**
     * The default number of iterations is {@value #DEFAULT_ITERATIONS}
     */
    public static final int DEFAULT_ITERATIONS = 200;
    /**
     * The default learning rate {@value #DEFAULT_LEARNING_RATE}
     */
    public static final double DEFAULT_LEARNING_RATE = 0.1;
    /**
     * The default eps distance factor between the two wining vectors 
     * {@value #DEFAULT_EPS}
     */
    public static final double DEFAULT_EPS = 0.3;
    /**
     * The default scaling factor for the {@link LVQVersion#LVQ3} case is 
     * {@value #DEFAULT_MSCALE}
     */
    public static final double DEFAULT_MSCALE = (0.5-0.1)/2+0.1;
    /**
     * The default method of LVQ to use LVQ3
     */
    public static final LVQVersion DEFAULT_LVQ_METHOD = LVQVersion.LVQ3; 
    /**
     * The default number of representatives per class is 
     * {@value #DEFAULT_REPS_PER_CLASS}
     */
    public static final int DEFAULT_REPS_PER_CLASS = 3;
    /**
     * The default stopping distance for convergence is 
     * {@value #DEFAULT_STOPPING_DIST}
     */
    public static final double DEFAULT_STOPPING_DIST = 1e-3;

    /**
     * The default seed selection method is SeedSelection.KPP
     */
    public static final SeedSelection DEFAULT_SEED_SELECTION= SeedSelection.KPP;
    
    private DecayRate learningDecay;
    private int iterations;
    private double learningRate;
    /**
     * The distance metric to use
     */
    protected DistanceMetric dm;
    private LVQVersion lvqVersion;
    private double eps;
    private double mScale;
    private double stoppingDist;
    private int representativesPerClass;
    /**
     * Array containing the learning vectors
     */
    protected Vec[] weights;
    /**
     * Array of the class that each learning vector represents
     */
    protected int[] weightClass;
    /**
     * Records the number of times each neuron won and was off the correct class
     * during training. Neurons that end with a count of zero wins will be ignored
     */
    protected int[] wins;
    private SeedSelectionMethods.SeedSelection seedSelection;
    /**
     * Contains the Learning vectors paired with their index in the weights array
     */
    protected VectorCollection<VecPaired<Vec, Integer>> vc;
    
    private VectorCollectionFactory<VecPaired<Vec, Integer>> vcf;

    /**
     * Creates a new LVQ instance 
     * @param dm the distance metric to use
     * @param iterations the number of iterations to perform
     */
    public LVQ(DistanceMetric dm, int iterations)
    {
        this(dm, iterations, DEFAULT_LEARNING_RATE, DEFAULT_REPS_PER_CLASS);
    }

    /**
     * Creates a new LVQ instance
     * @param dm the distance metric to use
     * @param iterations the number of iterations to perform
     * @param learningRate the learning rate to use when updating
     * @param representativesPerClass the number of representatives to create 
     * for each class
     */
    public LVQ(DistanceMetric dm, int iterations, double learningRate, 
            int representativesPerClass)
    {
        this(dm, iterations, learningRate, representativesPerClass, DEFAULT_LVQ_METHOD, new ExponetialDecay());
    }
    
    /**
     * Creates a new LVQ instance
     * @param dm the distance metric to use
     * @param iterations the number of iterations to perform
     * @param learningRate the learning rate to use when updating
     * @param representativesPerClass the number of representatives to create 
     * for each class
     * @param lvqVersion the version of LVQ to use
     * @param learningDecay the amount of decay to apply to the learning rate
     */
    public LVQ(DistanceMetric dm, int iterations, double learningRate, 
            int representativesPerClass, LVQVersion lvqVersion, 
            DecayRate learningDecay)
    {
        setLearningDecay(learningDecay);
        setIterations(iterations);
        setLearningRate(learningRate);
        setDistanceMetric(dm);
        setLVQMethod(lvqVersion);
        setEpsilonDistance(DEFAULT_EPS);
        setMScale(DEFAULT_MSCALE);
        setSeedSelection(DEFAULT_SEED_SELECTION);
        setVecCollectionFactory(new DefaultVectorCollectionFactory<VecPaired<Vec, Integer>>());
        setRepresentativesPerClass(representativesPerClass);
    }
    
    /**
     * Copy Constructor
     * @param toCopy version to copy
     */
    protected LVQ(LVQ toCopy)
    {
        this(toCopy.dm.clone(), toCopy.iterations, toCopy.learningRate, 
                toCopy.representativesPerClass, toCopy.lvqVersion, 
                toCopy.learningDecay);
        if(toCopy.weights != null)
        {
            wins = Arrays.copyOf(toCopy.wins, toCopy.wins.length);
            weights = new Vec[toCopy.weights.length];
            weightClass = Arrays.copyOf(toCopy.weightClass, toCopy.weightClass.length);

            for(int i = 0; i < toCopy.weights.length; i++)
                this.weights[i] = toCopy.weights[i].clone();
        }
        setEpsilonDistance(toCopy.eps);
        setMScale(toCopy.getMScale());
        setSeedSelection(toCopy.getSeedSelection());
        if(toCopy.vc != null)
            this.vc = toCopy.vc.clone();
        setVecCollectionFactory(toCopy.vcf.clone());
        
    }

    /**
     * When using {@link LVQVersion#LVQ3}, a 3rd case exists where up to two 
     * learning vectors can be updated at the same time if they have the same 
     * class. To avoid over fitting, an additional regularizing weight is placed
     * upon the learning rate for their update. THis sets the additional 
     * multiplied. It is suggested to use a value in the range of [0.1, 0.5]
     * 
     * @param mScale the multiplication factor to apply to the learning vectors
     */
    public void setMScale(double mScale)
    {
        if(mScale <= 0 || Double.isInfinite(mScale) || Double.isNaN(mScale))
            throw new ArithmeticException("Scale factor must be a positive constant, not " + mScale);
        this.mScale = mScale;
    }

    /**
     * Returns the scale used for the LVQ 3 learning algorithm update set.
     * @return a scale used during LVQ3
     */
    public double getMScale()
    {
        return mScale;
    }

    /**
     * Sets the epsilon multiplier that controls the maximum distance two 
     * learning vectors can be from each other in order to be updated at the 
     * same time. If they are too far apart, only one can be updated. It is 
     * recommended to use a value in the range [0.1, 0.3]
     * 
     * @param eps the scale factor of the maximum distance for two learning 
     * vectors to be updated at the same time
     */
    public void setEpsilonDistance(double eps)
    {
        if(eps <= 0 || Double.isInfinite(eps) || Double.isNaN(eps))
            throw new ArithmeticException("eps factor must be a positive constant, not " + eps);
        this.eps = eps;
    }

    /**
     * Sets the epsilon scale distance between learning vectors that may be 
     * allowed to two at a time. 
     * 
     * @return the scale of the allowable distance between learning vectors when
     * updating
     */
    public double getEpsilonDistance()
    {
        return eps;
    }

    /**
     * Sets the learning rate of the algorithm. It should be set in accordance 
     * with {@link #setLearningDecay(jsat.math.decayrates.DecayRate) }. 
     * 
     * @param learningRate the learning rate to use
     */
    public void setLearningRate(double learningRate)
    {
        if(learningRate <= 0 || Double.isInfinite(learningRate) || Double.isNaN(learningRate))
            throw new ArithmeticException("learning rate must be a positive constant, not " + learningRate);
        this.learningRate = learningRate;
    }

    /**
     * Returns the learning rate at which to apply updates during the algorithm.
     * 
     * @return the learning rate to use
     */
    public double getLearningRate()
    {
        return learningRate;
    }

    /**
     * Sets the decay rate to apply to the learning rate. 
     * 
     * @param learningDecay the rate to decay the learning rate 
     */
    public void setLearningDecay(DecayRate learningDecay)
    {
        this.learningDecay = learningDecay;
    }

    /**
     * Returns the method used to decay the learning rate over each iteration
     * @return the decay rate used at each iteration
     */
    public DecayRate getLearningDecay()
    {
        return learningDecay;
    }

    /**
     * Sets the number of learning iterations that will occur. 
     * 
     * @param iterations the number of iterations for the algorithm to use
     */
    public void setIterations(int iterations)
    {
        if(iterations < 0)
            throw new ArithmeticException("Can not perform a negative number of iterations");
        this.iterations = iterations;
    }

    /**
     * Returns the number of iterations of the algorithm to apply
     * @return the number of iterations to perform
     */
    public int getIterations()
    {
        return iterations;
    }
    
    /**
     * Sets the number of representatives to create for each class. It is 
     * possible to have an unbalanced number of representatives per class, but 
     * that is not currently supported. Increasing the number of representatives
     * per class increases the complexity of the decision boundary that can be 
     * learned. 
     * 
     * @param representativesPerClass the number of representatives to create 
     * for each class
     */
    public void setRepresentativesPerClass(int representativesPerClass)
    {
        this.representativesPerClass = representativesPerClass;
    }

    /**
     * Returns the number of representatives to create for each class. 
     * @return the number of representatives to create for each class. 
     */
    public int getRepresentativesPerClass()
    {
        return representativesPerClass;
    }

    /**
     * Sets the version of LVQ used. 
     * 
     * @param lvqMethod the version of LVQ to use
     */
    public void setLVQMethod(LVQVersion lvqMethod)
    {
        this.lvqVersion = lvqMethod;
    }

    /**
     * Returns the version of the LVQ algorithm to use.
     * @return the version of the LVQ algorithm to use.
     */
    public LVQVersion getLVQMethod()
    {
        return lvqVersion;
    }

    /**
     * Sets the distance used for learning
     * @param dm the distance metric to use
     */
    public void setDistanceMetric(DistanceMetric dm)
    {
        this.dm = dm;
    }

    /**
     * Returns the distance metric to use 
     * @return the distance metric to use
     */
    public DistanceMetric getDistanceMetric()
    {
        return dm;
    }

    /**
     * The algorithm terminates early if the learning vectors are only moving 
     * small distances. The stopping distance is the minimum distance that one 
     * of the learning vectors must move for the algorithm to continue.
     * 
     * @param stoppingDist the minimum distance for each learning vector to move
     */
    public void setStoppingDist(double stoppingDist)
    {
        if(stoppingDist < 0 || Double.isInfinite(stoppingDist) || Double.isNaN(stoppingDist))
            throw new ArithmeticException("stopping dist must be a zero or positive constant, not " + stoppingDist);
        this.stoppingDist = stoppingDist;
    }

    /**
     * Returns the stopping distance used to terminate the algorithm early
     * @return the stopping distance used toe nd the algorithm early
     */
    public double getStoppingDist()
    {
        return stoppingDist;
    }

    /**
     * Sets the seed selection method used to select the initial learning vectors 
     * @param seedSelection the method of initialing LVQ
     */
    public void setSeedSelection(SeedSelection seedSelection)
    {
        this.seedSelection = seedSelection;
    }

    /**
     * Returns the method of seed selection used 
     * @return the method of seed selection used 
     */
    public SeedSelection getSeedSelection()
    {
        return seedSelection;
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
    
    /**
     * There are several LVQ versions, each one adding an additional case in 
     * which two LVs instead of one can be updated. 
     */
    public enum LVQVersion 
    {
        /**
         * LVQ1 will only update one LV
         */
        LVQ1, 
        /**
         * Two vectors will be updated if they are close enough together. The 
         * closest was the wrong class but the 2nd closet was the correct class. 
         */
        LVQ2, 
        /**
         * Two vectors will be updated if they are close enough together and do 
         * not belong to the same class if one of them was the correct class to 
         * a training vector. 
         */
        LVQ21, 
        /**
         * Two vectors will be updated if they are close enough together and are
         * of the same class as the training vector.
         */
        LVQ3
    }

    /**
     * Sets the vector collection factory to use when storing the final learning vectors
     * @param vcf the vector collection factory to use
     */
    public void setVecCollectionFactory(VectorCollectionFactory<VecPaired<Vec, Integer>> vcf)
    {
        this.vcf = vcf;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(weightClass.length/representativesPerClass);
        
        
        int index = vc.search(data.getNumericalValues(), 1).get(0).getVector().getPair();
        cr.setProb(weightClass[index], 1.0);
        
        return cr;
    }

    /**
     * Returns true if the two distance values are within an acceptable epsilon 
     * ratio of each other. 
     * @param minDist the first distance
     * @param minDist2 the second distance
     * @return <tt>true</tt> if the are acceptable close
     */
    protected boolean epsClose(double minDist, double minDist2)
    {
        return min(minDist/minDist2, minDist2/minDist) > (1 - eps)
                        && max(minDist/minDist2, minDist2/minDist) < (1 + eps);
    }
    
    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        if(threadPool == null || threadPool instanceof FakeExecutor)
            TrainableDistanceMetric.trainIfNeeded(dm, dataSet);
        else
            TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadPool);
        Random rand = RandomUtil.getRandom();
        int classCount = dataSet.getPredicting().getNumOfCategories();
        weights = new Vec[classCount*representativesPerClass];
        Vec[] weightsPrev = new Vec[weights.length];
        weightClass = new int[weights.length];
        wins = new int[weights.length];

        //Generate weights that are hopefully close to their final positions
        
        int curClass = 0;
        int curPos = 0;
        while(curClass < classCount)
        {
            List<DataPoint> origSubList = dataSet.getSamples(curClass);
            List<DataPointPair<Integer>> subList =
                    new ArrayList<DataPointPair<Integer>>(origSubList.size());
            for(DataPoint dp : origSubList)
                subList.add(new DataPointPair<Integer>(dp, curClass));
            ClassificationDataSet subSet = 
                    new ClassificationDataSet(subList, dataSet.getPredicting());
            List<Vec> classSeeds = 
                    SeedSelectionMethods.selectIntialPoints(subSet, 
                    representativesPerClass, dm, rand, seedSelection);
            for(Vec v : classSeeds)
            {
                weights[curPos] = v.clone();
                weightsPrev[curPos] = weights[curPos].clone();
                weightClass[curPos++] = curClass;
            }
            curClass++;
        }
        Vec tmp = weights[0].clone();

        for(int iteration = 0; iteration < iterations; iteration++)
        {
            for(int j = 0; j < weights.length; j++)
                weights[j].copyTo(weightsPrev[j]);
            Arrays.fill(wins, 0);
            double alpha = learningDecay.rate(iteration, iterations, learningRate);
            for(int i = 0; i < dataSet.getSampleSize(); i++)
            {
                Vec x = dataSet.getDataPoint(i).getNumericalValues();
                int closestClass = -1;
                int minDistIndx = 0, minDistIndx2 = 0;
                double minDist = Double.POSITIVE_INFINITY, minDist2 = Double.POSITIVE_INFINITY;
                
                for(int j = 0; j < weights.length; j++)
                {
                    double dist = dm.dist(x, weights[j]);
                    if(dist < minDist)
                    {
                        if(lvqVersion == LVQVersion.LVQ2)
                        {
                            minDist2 = minDist;
                            minDistIndx2 = minDistIndx;
                        }
                        minDist = dist;
                        minDistIndx = j;
                        closestClass = dataSet.getDataPointCategory(i);
                        
                    }
                }

                if (lvqVersion.ordinal() >= LVQVersion.LVQ2.ordinal()
                        && weightClass[minDistIndx] != weightClass[minDistIndx2]
                        && closestClass == weightClass[minDistIndx2]
                        && epsClose(minDist, minDist2))
                {//Update both vectors 
                    //Move the closest farther away
                    x.copyTo(tmp);
                    tmp.mutableSubtract(weights[minDistIndx]);
                    weights[minDistIndx].mutableSubtract(alpha, tmp);
                    //And the 2nd closest closer
                    x.copyTo(tmp);
                    tmp.mutableSubtract(weights[minDistIndx2]);
                    weights[minDistIndx2].mutableAdd(alpha, tmp);
                    wins[minDistIndx2]++;
                }
                else if (lvqVersion.ordinal() >= LVQVersion.LVQ21.ordinal()
                        && weightClass[minDistIndx] != weightClass[minDistIndx2]
                        && closestClass == weightClass[minDistIndx]
                        && epsClose(minDist, minDist2))
                {//Update both vectors 
                    //Move the closest closer
                    x.copyTo(tmp);
                    tmp.mutableSubtract(weights[minDistIndx]);
                    weights[minDistIndx].mutableAdd(alpha, tmp);
                    wins[minDistIndx]++;
                    //And the 2nd closest farther away
                    x.copyTo(tmp);
                    tmp.mutableSubtract(weights[minDistIndx2]);
                    weights[minDistIndx2].mutableSubtract(alpha, tmp);
                }
                else if (lvqVersion.ordinal() >= LVQVersion.LVQ3.ordinal()
                        && weightClass[minDistIndx] == weightClass[minDistIndx2]
                        && min(minDist/minDist2, minDist2/minDist) > (1-eps)*(1+eps))
                {//Update both vectors in the same direction
                    x.copyTo(tmp);
                    tmp.mutableSubtract(weights[minDistIndx]);
                    weights[minDistIndx].mutableAdd(mScale*alpha, tmp);

                    x.copyTo(tmp);
                    tmp.mutableSubtract(weights[minDistIndx2]);
                    weights[minDistIndx2].mutableAdd(mScale*alpha, tmp);
                    wins[minDistIndx]++;
                    wins[minDistIndx2]++;
                }
                else //Base case, can only update one vector
                {
                    x.copyTo(tmp);
                    tmp.mutableSubtract(weights[minDistIndx]);
                    if(closestClass == weightClass[minDistIndx])//Move closer to the right class
                    {
                        wins[minDistIndx]++;
                        weights[minDistIndx].mutableAdd(alpha, tmp);
                    }
                    else//Move farther away
                    {
                        weights[minDistIndx].mutableSubtract(alpha, tmp);
                    }
                }
                
            }
            //Check for early convergence
            boolean stopEarly = true;
            for(int j = 0; j < weights.length; j++)
                if(stopEarly && dm.dist(weights[j], weightsPrev[j]) > stoppingDist)
                    stopEarly = false;
            if(stopEarly)
                break;
        }
        
        List<VecPaired<Vec, Integer>> finalLVs = new ArrayList<VecPaired<Vec, Integer>>(weights.length);
        for(int i = 0; i < weights.length; i++)
            if(wins[i] == 0)
                continue;
            else
                finalLVs.add(new VecPaired<Vec, Integer>(weights[i], i));
        if(threadPool == null || threadPool instanceof FakeExecutor)
            vc = vcf.getVectorCollection(finalLVs, dm);
        else
            vc = vcf.getVectorCollection(finalLVs, dm, threadPool);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public LVQ clone()
    {   
    	return new LVQ(this);
    }
    
}
