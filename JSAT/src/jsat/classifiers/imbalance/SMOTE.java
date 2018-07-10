/*
 * Copyright (C) 2017 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.classifiers.imbalance;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.DefaultVectorCollection;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionUtils;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.FakeExecutor;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

/**
 * This class implements the Synthetic Minority Over-sampling TEchnique (SMOTE)
 * for dealing with class imbalance. It does this by over-sampling the minority
 * classes to bring their total count up to parity (or some target ratio) with
 * the majority class. This is done by interpolating between minority points and
 * their neighbors to create new synthetic points that are not present in the
 * current dataset. For this reason SMOTE only works with numeric feature
 * vectors.<br>
 * <br>
 * See: Chawla, N., Bowyer, K., Hall, L., & Kegelmeyer, P. (2002). SMOTE:
 * synthetic minority over-sampling technique. Artificial Intelligence Research,
 * 16, 321–357. Retrieved from <a href="http://arxiv.org/abs/1106.1813">here</a>
 * @author Edward Raff
 */
public class SMOTE implements Classifier, Parameterized
{
    @ParameterHolder
    protected Classifier baseClassifier;
    protected DistanceMetric dm;
    protected int smoteNeighbors;
    protected double targetRatio;
    
    /**
     * Creates a new SMOTE model that will over-sample the minority classes so
     * that there is a balanced number of data points in each class.<br>
     * This implementation extends the original SMOTE algorithm to the
     * multi-class case.
     *
     *
     * @param baseClassifier the base classifier to use after the SMOTEing is
     * done.
     */
    public SMOTE(Classifier baseClassifier)
    {
        this(baseClassifier, new EuclideanDistance());
    }
    
    /**
     * Creates a new SMOTE model that will over-sample the minority classes so
     * that there is a balanced number of data points in each class.
     *
     * @param baseClassifier the base classifier to use after the SMOTEing is
     * done.
     * @param dm the distance metric to use for determining nearest neighbors
     */
    public SMOTE(Classifier baseClassifier, DistanceMetric dm)
    {
        this(baseClassifier, dm, 1.0);
    }
    
    /**
     * Creates a new SMOTE model.
     *
     * @param baseClassifier the base classifier to use after the SMOTEing is
     * done.
     * @param dm the distance metric to use for determining nearest neighbors
     * @param targetRatio the desired ratio of samples for each class with respect to the majority class. 
     */
    public SMOTE(Classifier baseClassifier, DistanceMetric dm, double targetRatio)
    {
        this(baseClassifier, dm, 5, targetRatio);
    }

    /**
     * Creates a new SMOTE object 
     * @param baseClassifier the base classifier to use after the SMOTEing is done. 
     * @param dm the distance metric to use for determining nearest neighbors
     * @param smoteNeighbors the number of neighbors to look at when interpolating points
     * @param targetRatio the desired ratio of samples for each class with respect to the majority class. 
     */
    public SMOTE(Classifier baseClassifier, DistanceMetric dm, int smoteNeighbors, double targetRatio)
    {
        setBaseClassifier(baseClassifier);
        setDistanceMetric(dm);
        setSmoteNeighbors(smoteNeighbors);
        setTargetRatio(targetRatio);
    }

    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public SMOTE(SMOTE toCopy)
    {
        this.baseClassifier = toCopy.baseClassifier.clone();
        this.dm = toCopy.dm.clone();
        this.smoteNeighbors = toCopy.smoteNeighbors;
        this.targetRatio = toCopy.targetRatio;
    }

    /**
     * Sets the metric used to determine the nearest neighbors of each point. 
     * @param dm the distance metric to use. 
     */
    public void setDistanceMetric(DistanceMetric dm)
    {
        this.dm = dm;
    }

    /**
     * 
     * @return the distance metric to use
     */
    public DistanceMetric getDistanceMetric()
    {
        return dm;
    }

    /**
     * Sets the number of neighbors that will be used to be candidates for
     * interpolation. The default value recommended in the original paper is 5.
     *
     * @param smoteNeighbors the number of candidate neighbors to select from
     * when creating synthetic data points.
     */
    public void setSmoteNeighbors(int smoteNeighbors)
    {
        if(smoteNeighbors < 1)
            throw new IllegalArgumentException("number of neighbors considered must be a positive value");
        this.smoteNeighbors = smoteNeighbors;
    }

    /**
     * 
     * @return the number of candidate neighbors to select from
     * when creating synthetic data points.
     */
    public int getSmoteNeighbors()
    {
        return smoteNeighbors;
    }

    /**
     * Sets the desired ratio of samples for each class compared to the majority
     * class. A ratio of 1.0 will oversample the minority classes until they
     * have just as many data points as the majority class. If any minority
     * class already exists at a ratio equal to or above this ratio, no over
     * samples will be created for that class. If the target ratio is greater
     * than one, all classes <i>including the majority class</i> will be
     * over-sampled to the desired ratio.
     *
     * @param targetRatio the target ratio between each class and the majority
     * class
     */
    public void setTargetRatio(double targetRatio)
    {
        this.targetRatio = targetRatio;
    }

    /**
     * 
     * @return the target ratio between each class and the majority
     * class
     */
    public double getTargetRatio()
    {
        return targetRatio;
    }
    
    

    /**
     * Sets the classifier to use after the dataset has been modified
     * @param baseClassifier the classifier to use for training and prediction
     */
    public void setBaseClassifier(Classifier baseClassifier)
    {
        this.baseClassifier = baseClassifier;
    }

    /**
     * 
     * @return the classifier used by the model
     */
    public Classifier getBaseClassifier()
    {
        return baseClassifier;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        return baseClassifier.classify(data);
    }

    @Override
    public void train(final ClassificationDataSet dataSet, boolean parallel)
    {
        if(dataSet.getNumCategoricalVars() != 0)
            throw new FailedToFitException("SMOTE only works with numeric-only feature values");
        
        List<Vec> vAll = dataSet.getDataVectors();
        IntList[] classIndex = new IntList[dataSet.getClassSize()];
        for(int i = 0; i < classIndex.length; i++)
            classIndex[i] = new IntList();
        for(int i = 0; i < dataSet.size(); i++)
            classIndex[dataSet.getDataPointCategory(i)].add(i);
        
        double[] priors = dataSet.getPriors();
        Vec ratios = DenseVector.toDenseVec(priors).clone();//yes, make a copy - I want the priors around too!
        /**
         * How many samples does it take to reach parity with the majority class
         */
        final int majorityNum = (int) (dataSet.size()*ratios.max());
        ratios.mutableDivide(ratios.max());
        
        final List<DataPointPair<Integer>> synthetics = new ArrayList<>();
        
        //Go through and perform oversampling of each class
        for(final int classID : ListUtils.range(0, dataSet.getClassSize()))
        {
            final int samplesNeeded = (int) (majorityNum * targetRatio - classIndex[classID].size());
            if(samplesNeeded <= 0)
                continue;
            //collect the vectors we need to interpolate with
            final List<Vec> V_id = new ArrayList<>();
            for(int i : classIndex[classID])
                V_id.add(vAll.get(i));
            VectorCollection<Vec> VC_id = new DefaultVectorCollection<>(dm, V_id, parallel);
            //find all the nearest neighbors for each point so we know who to interpolate with
            List<List<Integer>> neighbors = new ArrayList<>();
            List<List<Double>> distances = new ArrayList<>();
            VC_id.search(VC_id, smoteNeighbors+1, neighbors, distances, parallel);
            
            ParallelUtils.run(parallel, samplesNeeded, (start, end)->
            {
                Random rand = RandomUtil.getRandom();
                List<DataPoint> local_new = new ArrayList<>();
                for (int i = start; i < end; i++)
                {
                    int sampleIndex = i % V_id.size();
                    //which of the neighbors should we use?
                    int nn = rand.nextInt(smoteNeighbors) + 1;//index 0 is ourselve
                    Vec vec_nn = VC_id.get(neighbors.get(sampleIndex).get(nn));
                    double gap = rand.nextDouble();

                    // x ~ U(0, 1)
                    //new = sample + x * diff
                    //where diff = (sample - other)
                    //equivalent to
                    //new = sample * (x+1) + other * x
                    Vec newVal = V_id.get(sampleIndex).clone();
                    newVal.mutableMultiply(gap + 1);
                    newVal.mutableAdd(gap, vec_nn);
                    local_new.add(new DataPoint(newVal));
                }

                synchronized (synthetics)
                {
                    for (DataPoint v : local_new)
                        synthetics.add(new DataPointPair<>(v, classID));
                }
            });
            
        }
        
        ClassificationDataSet newDataSet = new ClassificationDataSet(ListUtils.mergedView(synthetics, dataSet.getAsDPPList()), dataSet.getPredicting());
        
        baseClassifier.train(newDataSet, parallel);
    }

    @Override
    public void train(ClassificationDataSet dataSet)
    {
        train(dataSet, false);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public SMOTE clone()
    {
        return new SMOTE(this);
    }
}
