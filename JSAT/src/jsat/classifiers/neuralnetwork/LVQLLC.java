package jsat.classifiers.neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.PriorClassifier;
import jsat.classifiers.bayesian.MultivariateNormals;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.math.decayrates.DecayRate;

/**
 * LVQ with Locally Learned Classifier (LVQ-LLC) is an adaption of the LVQ algorithm 
 * I have come up with. Given a classification data set, LVQ develops and moves 
 * several prototype vectors throughout the space, trying to place them as good 
 * representatives. Classification is then done Nearest Neighbor style among the 
 * prototypes. <br>
 * LVQ-LLC trains a local classifier on all of the data points that belong to 
 * the prototype, and the data points that lie across the border but are still 
 * near the prototype using the {@link #getEpsilonDistance() } parameter that is used to 
 * update two prototypes at the same time. Classification can then be done by 
 * getting the Classifier for the nearest prototype, or averaging the results of
 * the two closest prototypes if the point is near a boundary. <br>
 * This is done because, given a complex decision boundary and a sufficient 
 * number of prototypes, each prototype's domain will be a smaller subset of the
 * problem and will hopefully resemble a simpler decision problem that can be 
 * solved by a less complicated local learner. <br>
 * LVQ-LLC has the following advantages over LVQ:
 * <ul>
 * <li>Can return probabilities instead of hard classifications</li>
 * <li>Approximate decision boundaries can be more complicated than voronoi diagrams</li>
 * <li>Increase accuracy given a smaller number of prototypes per class</li>
 * </ul>
 * <br>
 * By default, the local classifier is the {@link MultivariateNormals}. 
 * 
 * @author Edward Raff
 */
public class LVQLLC extends LVQ
{

	private static final long serialVersionUID = 3602640001545233744L;
	private Classifier localClassifier;
    private Classifier[] localClassifeirs;

    /**
     * Creates a new LVQ-LLC instance that uses {@link MultivariateNormals} as 
     * the local classifier. 
     * @param dm the distance metric to use
     * @param iterations the number of iterations to perform
     */
    public LVQLLC(DistanceMetric dm, int iterations)
    {
        this(dm, iterations, new MultivariateNormals(true));
    }
    
    /**
     * Creates a new LVQ-LLC instance
     * @param dm the distance metric to use
     * @param iterations the number of iterations to perform
     * @param localClasifier the classifier to use as a local classifier for each prototype 
     */
    public LVQLLC(DistanceMetric dm, int iterations, Classifier localClasifier)
    {
        super(dm, iterations);
        setLocalClassifier(localClasifier);
    }

    /**
     * Creates a new LVQ-LLC instance
     * @param dm the distance metric to use
     * @param iterations the number of iterations to perform
     * @param localClasifier the classifier to use as a local classifier for each prototype 
     * @param learningRate the learning rate to use when updating
     * @param representativesPerClass the number of representatives to create 
     * for each class
     */
    public LVQLLC(DistanceMetric dm, int iterations, Classifier localClasifier, double learningRate, int representativesPerClass)
    {
        super(dm, iterations, learningRate, representativesPerClass);
        setLocalClassifier(localClasifier);
    }

    /**
     * Creates a new LVQ-LLC instance
     * @param dm the distance metric to use
     * @param iterations the number of iterations to perform
     * @param localClasifier the classifier to use as a local classifier for each prototype 
     * @param learningRate the learning rate to use when updating
     * @param representativesPerClass the number of representatives to create 
     * for each class
     * @param lvqVersion the version of LVQ to use
     * @param learningDecay the amount of decay to apply to the learning rate
     */
    public LVQLLC(DistanceMetric dm, int iterations, Classifier localClasifier, double learningRate, int representativesPerClass, LVQVersion lvqVersion, DecayRate learningDecay)
    {
        super(dm, iterations, learningRate, representativesPerClass, lvqVersion, learningDecay);
        setLocalClassifier(localClasifier);
    }

    protected LVQLLC(LVQLLC toCopy)
    {
        super(toCopy);
        if(toCopy.localClassifier != null)
            this.localClassifier = toCopy.localClassifier.clone();
        if(toCopy.localClassifeirs != null)
        {
            this.localClassifeirs = new Classifier[toCopy.localClassifeirs.length];
            for(int i = 0; i < this.localClassifeirs.length; i++)
                this.localClassifeirs[i] = toCopy.localClassifeirs[i].clone();
        }
    }

    /**
     * Each prototype will create a classifier that is local to itself, and 
     * trained on the points that belong to the prototype and those near the 
     * border of the prototype. This sets the classifier that will be used
     * 
     * @param localClassifier the local classifier to use for each prototype
     */
    public void setLocalClassifier(Classifier localClassifier)
    {
        this.localClassifier = localClassifier;
    }

    /**
     * Returns the classifier used for each prototype
     * @return the classifier used for each prototype
     */
    public Classifier getLocalClassifier()
    {
        return localClassifier;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> nns = 
                vc.search(data.getNumericalValues(), 2);
        double d1 = nns.get(0).getPair();
        int index1 = nns.get(0).getVector().getPair();
        double d2 = nns.get(1).getPair();
        int index2 = nns.get(1).getVector().getPair();
        
        CategoricalResults r1 = localClassifeirs[index1].classify(data);
        
        if(getLVQMethod().ordinal() >= LVQVersion.LVQ2.ordinal() && epsClose(d1, d2))
        {
            CategoricalResults result = new CategoricalResults(r1.size());
            CategoricalResults r2 = localClassifeirs[index2].classify(data);
            double distSum = d1+d2;
            
            for(int i = 0; i < r1.size(); i++)
            {
                result.incProb(i, r1.getProb(i)*(distSum-d1));
                result.incProb(i, r2.getProb(i)*(distSum-d2));
            }
            result.normalize();
            return result;
        }
        else
            return r1;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        super.trainC(dataSet, threadPool);
        
        List<List<DataPointPair<Integer>>> listOfLocalPoints = new ArrayList<List<DataPointPair<Integer>>>(weights.length);
        for (int i = 0; i < weights.length; i++)
            listOfLocalPoints.add(new ArrayList<DataPointPair<Integer>>(wins[i] * 3 / 2));
        for (DataPointPair<Integer> dpp : dataSet.getAsDPPList())
        {
            Vec x = dpp.getVector();
            int minDistIndx = 0, minDistIndx2 = 0;
            double minDist = Double.POSITIVE_INFINITY, minDist2 = Double.POSITIVE_INFINITY;

            List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> closestWeightVecs = vc.search(x, 2);
            
            VecPaired<VecPaired<Vec, Integer>, Double> closest = closestWeightVecs.get(0);
            minDistIndx = closest.getVector().getPair();
            minDist = closest.getPair();
            
            VecPaired<VecPaired<Vec, Integer>, Double> closest2nd = closestWeightVecs.get(0);
            minDistIndx2 = closest2nd.getVector().getPair();
            minDist2 = closest2nd.getPair();
            

            listOfLocalPoints.get(minDistIndx).add(dpp);
            double tmpEps = getEpsilonDistance();
            if(Math.min(minDist/minDist2, minDist2/minDist) > (1 - tmpEps) 
                    && Math.max(minDist/minDist2, minDist2/minDist) < (1 + tmpEps))
            {
                listOfLocalPoints.get(minDistIndx2).add(dpp);
            }
                
        }

        localClassifeirs = new Classifier[weights.length];
        for(int i = 0; i < weights.length; i++)
        {
            if(wins[i] == 0)
                continue;
            ClassificationDataSet localSet = new ClassificationDataSet(listOfLocalPoints.get(i), dataSet.getPredicting());
            if(wins[i] < 10)
            {
                CategoricalResults cr = new CategoricalResults(dataSet.getPredicting().getNumOfCategories());
                cr.setProb(weightClass[i], 1.0);
                localClassifeirs[i] = new PriorClassifier(cr);
            }
            else
            {
                localClassifeirs[i] = localClassifier.clone();
                localClassifeirs[i].trainC(localSet);
            }
        }
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
    }

    @Override
    public LVQLLC clone()
    {
        return new LVQLLC(this);
    }
}
