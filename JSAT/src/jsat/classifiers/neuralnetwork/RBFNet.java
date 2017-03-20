package jsat.classifiers.neuralnetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicIntegerArray;

import jsat.DataSet;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.svm.DCDs;
import jsat.clustering.kmeans.HamerlyKMeans;
import jsat.clustering.SeedSelectionMethods;
import jsat.datatransform.DataTransform;
import jsat.distributions.Distribution;
import jsat.distributions.Uniform;
import jsat.distributions.discrete.UniformDiscrete;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.MahalanobisDistance;
import jsat.math.OnLineStatistics;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.BoundedSortedList;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.IntList;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * This provides a highly configurable implementation of a Radial Basis Function
 * Neural Network. A RBF network is a type of neural network that contains one
 * hidden layer, and is related to the {@link LVQ} algorithm. In a classical RBF
 * Network, the distance between two data points is generally the
 * {@link EuclideanDistance} or {@link MahalanobisDistance}. This implementation
 * allows the specification of any arbitrary distance metric. <br>
 * <br>
 * Another restriction on classical RBF Nets is that a weighted sum of the 
 * output of the hidden units be used to make the final decision. Instead this 
 * implementation allows the specification of an arbitrary Classifier or 
 * Regressor to estimate the outputs based on the hidden unit activations. 
 * Whether or not the predictor supports Classification, Regression, and what
 * classification features it supports - will determine what the RBF Network 
 * supports. This allows for models technically more complicated and powerful 
 * than the standard RBF network. <br>
 * <br>
 * The initial phases of a RBF Network is to learn the neuron locations and 
 * activations. This part can also be seen as learning a data transformation. As
 * such, the RBF Network can be used as a DataTransform itself. <br>
 * The last phase of the network is to learn the model based on the data point 
 * activations. <br>
 * <br>
 * It is highly recommended to use a base learning method that can efficiently 
 * use sparse vectors. 
 * 
 * @author Edward Raff
 */
public class RBFNet implements Classifier, Regressor, DataTransform, Parameterized
{

    private static final long serialVersionUID = 5418896646203518062L;
    private int numCentroids;
    private Phase1Learner p1l;
    private Phase2Learner p2l;
    private double alpha;
    private int p;
    private DistanceMetric dm;
    private boolean normalize = true;
    
    private Classifier baseClassifier;
    private Regressor baseRegressor;
    
    private List<Double> centroidDistCache;
    private List<Vec> centroids;
    private double[] bandwidths;

    /**
     * Creates a new RBF Network suitable for binary classification or
     * regression and uses 100 hidden nodes. One of the other constructors
     * should be used if you need classification for multi-class or if you need
     * probability outputs. <br>
     * <br>
     * This will use {@link Phase1Learner#K_MEANS} for neuron selection and
     * {@link Phase2Learner#NEAREST_OTHER_CENTROID_AVERAGE} for activation
     * tuning. The {@link EuclideanDistance} will be use as the metric.
     *
     */
    public RBFNet()
    {
        this(100);
    }
    
    /**
     * Creates a new RBF Network suitable for binary classification or 
     * regression. One of the other constructors should be used if you need 
     * classification for multi-class or if you need probability outputs. <br>
     * <br>
     * This will use {@link Phase1Learner#K_MEANS} for neuron selection and 
     * {@link Phase2Learner#NEAREST_OTHER_CENTROID_AVERAGE} for activation tuning. 
     * The {@link EuclideanDistance} will be use as the metric. 
     * 
     * @param numCentroids the number of centroids or neurons to use in the 
     * network's hidden layer
     */
    public RBFNet(int numCentroids)
    {
        this(numCentroids, Phase1Learner.K_MEANS, Phase2Learner.NEAREST_OTHER_CENTROID_AVERAGE, 3, 3, new EuclideanDistance(), (Classifier) new DCDs());
    }

    /**
     * Creates a new RBF Network for classification tasks. If the classifier can
     * also perform regression, then the network will be able to perform both.
     * 
     * @param numCentroids the number of centroids or neurons to use in the 
     * network's hidden layer
     * @param cl the method to learn the neuron locations
     * @param bl the method to learn the neuron activations 
     * @param alpha a parameter that may have an effect on the neuron activation
     * learning method. 
     * @param p a parameter that may have an effect on the neuron activation 
     * learning method
     * @param dm the distance metric to use
     * @param baseClassifier the base classifier to learn on top of the hidden 
     * layer activations. 
     */
    public RBFNet(int numCentroids, Phase1Learner cl, Phase2Learner bl, double alpha, int p, DistanceMetric dm, Classifier baseClassifier)
    {
        setNumCentroids(numCentroids);
        setPhase1Learner(cl);
        setPhase2Learner(bl);
        setAlpha(alpha);
        setP(p);
        setDistanceMetric(dm);
        this.baseClassifier = baseClassifier;
        if(baseClassifier instanceof Regressor)
            baseRegressor = (Regressor) baseClassifier;
    }
    /**
     * Creates a new RBF Network for regression tasks. If the regressor can
     * also perform classification, then the network will be able to perform 
     * both. 
     * 
     * @param numCentroids the number of centroids or neurons to use in the 
     * network's hidden layer
     * @param cl the method to learn the neuron locations
     * @param bl the method to learn the neuron activations 
     * @param alpha a parameter that may have an effect on the neuron activation
     * learning method. 
     * @param p a parameter that may have an effect on the neuron activation 
     * learning method
     * @param dm the distance metric to use
     * @param baseRegressor the base regressor to learn on op of the hidden 
     * layer activations. 
     */
    public RBFNet(int numCentroids, Phase1Learner cl, Phase2Learner bl, double alpha, int p, DistanceMetric dm, Regressor baseRegressor)
    {
        setNumCentroids(numCentroids);
        setPhase1Learner(cl);
        setPhase2Learner(bl);
        setAlpha(alpha);
        setP(p);
        setDistanceMetric(dm);
        this.baseRegressor = baseRegressor;
        if(baseRegressor instanceof Classifier)
            baseClassifier = (Classifier) baseRegressor;
    }

    /**
     * Copy constructor
     * @param toCopy the network to copy
     */
    public RBFNet(RBFNet toCopy)
    {
        setNumCentroids(toCopy.getNumCentroids());
        setPhase1Learner(toCopy.getPhase1Learner());
        setPhase2Learner(toCopy.getPhase2Learner());
        setAlpha(toCopy.getAlpha());
        setP(toCopy.getP());
        setDistanceMetric(toCopy.getDistanceMetric().clone());
        if(toCopy.baseRegressor != null)
        {
            this.baseRegressor = toCopy.baseRegressor.clone();
            if(baseRegressor instanceof Classifier)
                baseClassifier = (Classifier) baseRegressor;
        }
        else if(toCopy.baseClassifier != null)
        {
            this.baseClassifier = toCopy.baseClassifier.clone();
            if(baseClassifier instanceof Regressor)
                baseRegressor = (Regressor) baseClassifier;
        }
        if(toCopy.centroids != null)
        {
            this.centroids = new ArrayList<Vec>(toCopy.centroids.size());
            for(Vec v : toCopy.centroids)
                this.centroids.add(v.clone());
            if(toCopy.centroidDistCache != null)
                this.centroidDistCache = new DoubleList(toCopy.centroidDistCache);
        }
        
        if(toCopy.bandwidths != null)
            this.bandwidths = Arrays.copyOf(toCopy.bandwidths, toCopy.bandwidths.length);
    }

    @Override
    public DataPoint transform(DataPoint dp)
    {
        final Vec x = dp.getNumericalValues();
        final List<Double> qi = dm.getQueryInfo(x);
        Vec sv = new SparseVector(numCentroids);
        double sum = 0;
        /*
         * Keep track of the highest activation in case none of the neurons have
         * a numericaly stable activation value. if this occurs we will do our 
         * best by simply setting the one largest activation
         */
        double maxActivation = Double.NEGATIVE_INFINITY;
        int highestNeuron = -1;
                
        for(int i = 0; i < centroids.size(); i++)
        {
            double dist = dm.dist(i, x, qi, centroids, centroidDistCache);
            double sig = bandwidths[i];
            double activation = Math.exp(-(dist*dist)/(sig*sig*2));
            
            if(activation > maxActivation)
            {
                maxActivation = activation;
                highestNeuron = i;
            }
            
            if(activation > 1e-16)
            {
                sv.set(i, activation);
                sum += activation;
            }
        }
        
        if(sv.nnz() == 0)//no activations
        {
            sv.set(highestNeuron, maxActivation);
            sum = maxActivation;
        }
            
        
        if(normalize && sum != 0.0)//-0.0 not an issue with rbf kernel
            sv.mutableDivide(sum);
        if(sv.nnz() > sv.length()/2)//at this point we would be using more memory than needed. Just switch to dense
            sv = new DenseVector(sv);
        
        return new DataPoint(sv, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
    }
    /**
     * The first phase of learning a  RBF Neural Network is to determine the 
     * neuron locations. This enum controls which method is used. 
     */
    public static enum Phase1Learner
    {
        /**
         * Selects the Neurons at random from the training data 
         */
        RANDOM
        {
            @Override
            protected List<Vec> getCentroids(DataSet data, int centroids, DistanceMetric dm, ExecutorService ex)
            {
                Random rand = RandomUtil.getRandom();
                List<Vec> toRet = new ArrayList<Vec>();
                Set<Integer> points = new IntSet();
                
                while (points.size() < centroids)
                    points.add(rand.nextInt(data.getSampleSize()));
                    
                for (int i : points)
                    toRet.add(data.getDataPoint(i).getNumericalValues());
                
                return toRet;
            }
        },
        /**
         * Selects the Neurons by performing k-Means clustering on the data
         */
        K_MEANS
        {
            @Override
            protected List<Vec> getCentroids(DataSet data, int centroids, DistanceMetric dm, ExecutorService ex)
            {
                HamerlyKMeans kmeans = new HamerlyKMeans(dm, SeedSelectionMethods.SeedSelection.KPP);
                
                if(ex == null || ex instanceof FakeExecutor)
                    kmeans.cluster(data, centroids);
                else
                    kmeans.cluster(data, centroids, ex);
                
                return kmeans.getMeans();
            }
        };
        
        /**
         * Obtains the centroids for the given data set
         * @param data the data set to get the centroids for
         * @param centroids the number of centroids to obtain
         * @param dm the distance metric that is being used
         * @param ex the source of threads for parallel computation
         * @return a list of the centroid vectors
         */
        abstract protected List<Vec> getCentroids(DataSet data, int centroids, DistanceMetric dm, ExecutorService ex);
    }
    
    /**
     * The second phase of learning a RBF Neural Network is to determine how the
     * neurons are activated to produce the output of the hidden layer. This 
     * enum control which method is used. 
     */
    public static enum Phase2Learner
    {
        /**
         * This method sets the bandwidth for each neuron based on the distances
         * to the neuron from each data point that is closest to said neuron. If
         * &mu; is the average distance to the neuron, and &sigma; the standard 
         * deviation, then the bandwidth <i>b</i> of the <i>j</i>'th neuron is 
         * seto to <i>b<sub>j</sub> = &mu;<sub>j</sub> + 
         * {@link #setAlpha(double) &alpha;} &sigma;<sub>j</sub></i>
         */
        CENTROID_DISTANCE
        {
            @Override
            protected double[] estimateBandwidths(double alpha, int p, DataSet data, final List<Vec> centroids, final List<Double> centroidDistCache, final DistanceMetric dm, ExecutorService threadpool)
            {
                final double[] bandwidths = new double[centroids.size()];
                final OnLineStatistics[] averages = new OnLineStatistics[bandwidths.length];
                for(int i = 0; i < averages.length; i++)
                    averages[i] = new OnLineStatistics();
                
                final List<Future<OnLineStatistics[]>> futures = new ArrayList<Future<OnLineStatistics[]>>(SystemInfo.LogicalCores);
                
                /**
                 * Compute the stats for each subset and then merge them
                 */
                for (final List<Vec> subList : ListUtils.splitList((List< Vec>) data.getDataVectors(), SystemInfo.LogicalCores))
                {
                    Future<OnLineStatistics[]> future = threadpool.submit(new Callable<OnLineStatistics[]>() 
                    {

                        @Override
                        public OnLineStatistics[] call()
                        {
                            final OnLineStatistics[] localAverages = new OnLineStatistics[bandwidths.length];
                            for (int i = 0; i < localAverages.length; i++)
                                localAverages[i] = new OnLineStatistics();
                            
                            for(Vec x : subList)
                            {
                                double minDist = Double.POSITIVE_INFINITY;
                                int minI = 0;
                                for(int i = 0; i < centroids.size(); i++)
                                {
                                    double dist = dm.dist(i, x, centroids, centroidDistCache);
                                    if(dist < minDist)
                                    {
                                        minDist = dist;
                                        minI = i;
                                    }
                                }
                                localAverages[minI].add(minDist);
                            }
                            return localAverages;
                        }
                    });
                    futures.add(future);
                }
                try
                {
                    ///Wait for all the work to finish
                    for (OnLineStatistics[] localAverages : ListUtils.collectFutures(futures))
                    {
                        for (int i = 0; i < localAverages.length; i++)
                        {
                            if (localAverages[i].getSumOfWeights() == 0)
                                continue;
                            averages[i] = OnLineStatistics.add(averages[i], localAverages[i]);
                        }
                    }

                    for(int i = 0; i < bandwidths.length; i++)
                        bandwidths[i] = averages[i].getMean()+averages[i].getStandardDeviation()*alpha;
                }
                catch (InterruptedException ex)
                {
                    throw new FailedToFitException(ex);
                }
                catch (ExecutionException ex)
                {
                    throw new FailedToFitException(ex);
                }
                
                return bandwidths;
            }
        },
        /**
         * This bandwidth estimator only works for classification problems. Each
         * neuron is assigned a class based on the majority class labels of the
         * data points closes to said neuron. The bandwidth is then estimated as
         * {@link #setAlpha(double) &alpha;} times the distance from the neuron
         * to the closest neuron with a different class label.<br>
         * <br>
         * For this method &alpha; values between (0, 1) usually work best, 0.25 is a 
         * good starting value. The value of &alpha; can go past 1. 
         */
        CLOSEST_OPPOSITE_CENTROID
        {
            @Override
            protected double[] estimateBandwidths(final double alpha, int p, DataSet data, final List<Vec> centroids, final List<Double> centroidDistCache, final DistanceMetric dm, ExecutorService threadpool)
            {
                final ClassificationDataSet cds;
                if(data instanceof ClassificationDataSet )
                    cds = (ClassificationDataSet) data;
                else
                    throw new FailedToFitException("CLOSEST_OPPOSITE_CENTROID only works for classification data sets");
                
                final double[] bandwidths = new double[centroids.size()];
                final CountDownLatch latch0 = new CountDownLatch(SystemInfo.LogicalCores);
                
                /**
                 * An array of arrays. Each centroid gets its own atomic array, 
                 * where each value indicates how many objects of class is stored. 
                 */
                final AtomicIntegerArray[] classLabels = new AtomicIntegerArray[centroids.size()];
                for(int i =0; i < classLabels.length; i++)
                    classLabels[i] = new AtomicIntegerArray(cds.getClassSize());
                IntList indices = new IntList(data.getSampleSize());
                ListUtils.addRange(indices, 0, data.getSampleSize(), 1);
                for(final List<Integer> subList : ListUtils.splitList(indices, SystemInfo.LogicalCores))
                {
                    threadpool.submit(new Runnable() 
                    {
                        @Override
                        public void run()
                        {
                            for(int id : subList)
                            {
                                final Vec x = cds.getDataPoint(id).getNumericalValues();
                                double minDist = Double.POSITIVE_INFINITY;
                                int minI = 0;
                                for(int i = 0; i < centroids.size(); i++)
                                {
                                    double dist = dm.dist(i, x, centroids, centroidDistCache);
                                    if(dist < minDist)
                                    {
                                        minDist = dist;
                                        minI = i;
                                    }
                                }
                                
                                classLabels[minI].incrementAndGet(cds.getDataPointCategory(id));
                            }
                            
                            latch0.countDown();
                        }
                    });
                }
                try
                {
                    ///Wait for all the work to finish
                    latch0.await();
                }
                catch (InterruptedException ex)
                {
                    throw new FailedToFitException(ex);
                }
                
                //Figure out the class label for each neuron
                final int[] neuronClass = new int[centroids.size()];
                for(int i = 0; i < neuronClass.length; i++)
                {
                    int maxVal = -1;
                    int maxClass = 0;
                    for(int j = 0; j < classLabels[i].length(); j++)
                    {
                        if(classLabels[i].get(j) > maxVal)
                        {
                            maxClass = j;
                            maxVal = classLabels[i].get(j);
                        }
                    }
                    neuronClass[i] = maxClass;
                }
                //Now set the bandwidth based on the distance to the nearest centroid with a different class label
     
                final CountDownLatch latch1 = new CountDownLatch(centroids.size());
                for (int i = 0; i < centroids.size(); i++)
                {
                    final int center = i;
                    threadpool.submit(new Runnable()
                    {
                        @Override
                        public void run()
                        {
                            double minDist = Double.POSITIVE_INFINITY;
                            for (int i = 0; i < centroids.size(); i++)
                                if (neuronClass[center] != neuronClass[i])//dont check for ourselves b/c we have the same class as ourselves, so no need
                                    minDist = Math.min(minDist, dm.dist(i, center, centroids, centroidDistCache));

                            if (Double.isInfinite(minDist))//possible if there is high class imbalance, run again but lie
                                for (int i = 0; i < centroids.size(); i++)
                                    if (center != i)
                                        minDist = Math.min(minDist, dm.dist(i, center, centroids, centroidDistCache));

                            bandwidths[center] = alpha * minDist;
                            latch1.countDown();
                        }
                    });
                }
                
                try
                {
                    latch1.await();
                }
                catch (InterruptedException ex)
                {
                    throw new FailedToFitException(ex);
                }
                
                return bandwidths;
            }
        },
        /**
         * This method sets the bandwidth for each neuron based on the average 
         * distance of the {@link #setP(int) p} nearest neurons. The number of 
         * standard deviations to add to the activation is controlled by 
         * {@link #setAlpha(double) &alpha;}
         */
        NEAREST_OTHER_CENTROID_AVERAGE
        {
            @Override
            protected double[] estimateBandwidths(final double alpha, final int p, DataSet data, final List<Vec> centroids, final List<Double> centroidDistCache, final DistanceMetric dm, ExecutorService threadpool)
            {
                final double[] bandwidths = new double[centroids.size()];
                final CountDownLatch latch = new CountDownLatch(centroids.size());
                for(int i = 0; i < centroids.size(); i++)
                {
                    final int center = i;
                    threadpool.submit(new Runnable() 
                    {
                        @Override
                        public void run()
                        {
                            BoundedSortedList<Double> closestDistances = new BoundedSortedList<Double>(p);
                            for(int i = 0; i < centroids.size(); i++)
                                if(i != center)
                                    closestDistances.add(dm.dist(i, center, centroids, centroidDistCache));
                            OnLineStatistics stats = new OnLineStatistics();
                            for(double dist : closestDistances)
                                stats.add(dist);
                            bandwidths[center] = stats.getMean()+alpha*stats.getStandardDeviation();
                            latch.countDown();
                        }
                    });
                }
                return bandwidths;
            }
        };
        abstract protected double[] estimateBandwidths(double alpha, int p, final DataSet data, final List<Vec> centroids, final List<Double> centroidDistCache, final DistanceMetric dm, ExecutorService threadpool);
    }

    /**
     * Sets the alpha parameter. This value is used for certain
     * {@link Phase2Learner} learners as a parameter. A good default value for
     * most methods is often 1 or 3. However the parameter must always be
     * a non-negative value.
     *
     * @param alpha a non negative value that controls the width of the learned
     * bandwidths.
     */
    public void setAlpha(double alpha)
    {
        if(alpha < 0 || Double.isInfinite(alpha) || Double.isNaN(alpha))
            throw new IllegalArgumentException("Alpha must be a positive value, not " + alpha);
        this.alpha = alpha;
    }

    /**
     * Returns the alpha bandwidth learning parameter
     * @return the alpha bandwidth learning parameter
     * @see #setAlpha(double) 
     */
    public double getAlpha()
    {
        return alpha;
    }
    
    /**
     * Guesses the distribution for the {@link #setAlpha(double) } parameter
     * @param data the data to create a guess for
     * @return a guess for the distribution of the Alpha parameter
     */
    public static Distribution guessAlpha(DataSet data)
    {
        return new Uniform(0.8, 3.5);
    }

    /**
     * Sets the nearest neighbor parameter. This value is used for certain
     * {@link Phase2Learner} learners as a parameter. It is used to control the
     * number of neighbors taken into account in learning the parameter value.
     * It must always be a positive value. 3 is usually a good value for
     * this parameter.
     *
     * @param p the positive integer used that controls the width of the learned
     * bandwidths
     */
    public void setP(int p)
    {
        if(p < 1)
            throw new IllegalArgumentException("neighbors parameter must be positive, not "+p);
        this.p = p;
    }

    /**
     * Returns the nearest neighbors parameter. 
     * @return the nearest neighbors parameter. 
     * @see #setP(int) 
     */
    public int getP()
    {
        return p;
    }

    /**
     * Guesses the distribution for the {@link #setP(int) } parameter
     * @param data the data to create a guess for
     * @return a guess for the distribution of the P parameter
     */
    public static Distribution guessP(DataSet data)
    {
        return new UniformDiscrete(2, 5);
    }

    /**
     * Sets the number of centroids to learn for this model. Increasing the
     * number of centroids increases the complexity of the model as well as
     * training and evaluation time. The centroids serve as the hidden units in
     * the network.
     * <br><br>
     * The centroids learned are controlled via the 
     * {@link #setPhase1Learner(jsat.classifiers.neuralnetwork.RBFNet.Phase1Learner)} 
     * method
     *
     * @param numCentroids the number of centroids to use in the model
     */
    public void setNumCentroids(int numCentroids)
    {
        if(numCentroids < 1)
            throw new IllegalArgumentException("Number of centroids must be positive, not " + numCentroids);
        this.numCentroids = numCentroids;
    }

    /**
     * Returns the number of centroids to use when training
     * @return * Returns the number of centroids to use when training
     */
    public int getNumCentroids()
    {
        return numCentroids;
    }
    
    /**
     * Guesses the distribution for the {@link #setNumCentroids(int)  } parameter
     * @param data the data to create a guess for
     * @return a guess for the distribution of the number of centroids to use
     */
    public static Distribution guessNumCentroids(DataSet data)
    {
        return new UniformDiscrete(25, 1000);//maybe change in the future
    }

    /**
     * Sets the distance metric used to determine neuron activations. 
     * @param dm the distance metric to use
     */
    public void setDistanceMetric(DistanceMetric dm)
    {
        this.dm = dm;
    }

    /**
     * Returns the distance metric in use
     * @return the distance metric in use
     */
    public DistanceMetric getDistanceMetric()
    {
        return dm;
    }

    /**
     * Sets the method used for learning the centroids (or hidden units) of the
     * network.
     *
     * @param p1l the learning method to use
     */
    public void setPhase1Learner(Phase1Learner p1l)
    {
        this.p1l = p1l;
    }

    /**
     * Returns the method to use for learning the centroids of the network.
     *
     * @return the method to use for learning the centroids of the network.
     */
    public Phase1Learner getPhase1Learner()
    {
        return p1l;
    }

    /**
     * Sets the method used for learning the bandwidths for each centroid in the
     * network. Depending on the method used, {@link #setAlpha(double) } or 
     * {@link #setP(int)} may impact the learned bandwidths.
     *
     * @param p2l the learning method to use
     */
    public void setPhase2Learner(Phase2Learner p2l)
    {
        this.p2l = p2l;
    }

    /**
     * Returns the learning method to use for determining the bandwidths of each
     * center in the network.  
     * @return the learning method to use for the bandwidths
     */
    public Phase2Learner getPhase2Learner()
    {
        return p2l;
    }

    /**
     * Sets whether or not to normalize the outputs of the neurons in the
     * network so that the activations sum to one. Normalizing the outputs can
     * increase the generalization ability of the network. By default this is
     * set to {@code true}
     *
     * @param normalize {@code true} to normalize the neuron outputs,
     * {@code false} to use the raw activation values.
     */
    public void setNormalize(boolean normalize)
    {
        this.normalize = normalize;
    }

    /**
     * Returns whether or not the network is currently normalizing its neuron
     * outputs.
     *
     * @return whether or not the neuron outputs are normalized
     */
    public boolean isNormalize()
    {
        return normalize;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        return baseClassifier.classify(transform(data));
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        if(baseClassifier == null)
            throw new FailedToFitException("RBFNet was not given a base classifier");
        if(threadPool == null)
            threadPool = new FakeExecutor();
        //Learn Centroids
        centroids = p1l.getCentroids(dataSet, numCentroids, dm, threadPool);
        centroidDistCache = dm.getAccelerationCache(centroids, threadPool);
        
        //Learn Parameter Values
        bandwidths = p2l.estimateBandwidths(alpha, p, dataSet, centroids, centroidDistCache, dm, threadPool);
        
        //apply transform
        ClassificationDataSet transformedData = dataSet.shallowClone();
        transformedData.applyTransform(this, threadPool);
        
        //learn final model on transformed inputs
        baseClassifier.trainC(transformedData, threadPool);
        
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
    }

    @Override
    public boolean supportsWeightedData()
    {
        if(baseClassifier != null)
            return baseClassifier.supportsWeightedData();
        else 
            return baseRegressor.supportsWeightedData();
    }

    @Override
    public double regress(DataPoint data)
    {
        return baseRegressor.regress(transform(data));
    }

    @Override
    public void fit(DataSet data)
    {
        if (data instanceof ClassificationDataSet)
            trainC((ClassificationDataSet) data);
        else if(data instanceof RegressionDataSet)
            train((RegressionDataSet) data);
        else
            throw new FailedToFitException("Data must be a classifiation or regression dataset, not " + data.getClass().getSimpleName());
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        if(baseRegressor == null)
            throw new FailedToFitException("RBFNet was not given a base classifier");
        if(threadPool == null)
            threadPool = new FakeExecutor();
        //Learn Centroids
        centroids = p1l.getCentroids(dataSet, numCentroids, dm, threadPool);
        centroidDistCache = dm.getAccelerationCache(centroids, threadPool);
        
        //Learn Parameter Values
        bandwidths = p2l.estimateBandwidths(alpha, p, dataSet, centroids, centroidDistCache, dm, threadPool);
        
        //apply transform
        RegressionDataSet transformedData = dataSet.shallowClone();
        transformedData.applyTransform(this, threadPool);
        
        //learn final model on transformed inputs
        baseRegressor.train(transformedData, threadPool);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, null);
    }

    @Override
    public RBFNet clone()
    {
        return new RBFNet(this);
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
