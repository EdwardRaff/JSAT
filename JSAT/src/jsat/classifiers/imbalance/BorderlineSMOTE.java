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
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

/**
 * This class implements the Borderline extension of the {@link SMOTE} algorithm
 * for dealing with class imbalance. SMOTE over-samples from the minority class
 * at random points in the space. Borderline smote attempts to estimate which
 * points are on the border of the class bounder, and over-samples only from the
 * points on the boarder. Boarderline-SMOTE can also choose to
 * {@link #setMajorityInterpolation(boolean) perform interpolation using samples for the majority class},
 * which can sometimes improve performance. The border is impacted by changes to
 * the number of {@link #setSmoteNeighbors(int) neighbors} used. In the rare
 * event that a boarder can't be estimated, this implementation will fall back
 * to standard SMOTE.<br>
 * This implementation extends the original SMOTE algorithm to the
 * multi-class case.<br>
 * <br>
 * See: Han, H., Wang, W.-Y., & Mao, B.-H. (2005). Borderline-SMOTE: A New
 * Over-sampling Method in Imbalanced Data Sets Learning. In Proceedings of the
 * 2005 International Conference on Advances in Intelligent Computing - Volume
 * Part I (pp. 878–887). Berlin, Heidelberg: Springer-Verlag.
 * <a href="http://doi.org/10.1007/11538059_91">DOI:10.1007/11538059_91</a>
 * @author Edward Raff
 */
public class BorderlineSMOTE extends SMOTE
{
    private boolean majorityInterpolation;

    /**
     * Creates a new Borderline-SMOTE model that will over-sample the minority
     * classes so that there is a balanced number of data points in each class.
     * It will not use majority interpolation.
     *
     * @param baseClassifier the base classifier to use after the SMOTEing is
     * done.
     */
    public BorderlineSMOTE(Classifier baseClassifier)
    {
        this(baseClassifier, false);
    }
    
    /**
     * Creates a new Borderline-SMOTE model that will over-sample the minority
     * classes so that there is a balanced number of data points in each class.
     *
     * @param baseClassifier the base classifier to use after the SMOTEing is
     * done.
     * @param majorityInterpolation {@code true} if synthetic examples should
     * use the majority class as well, or {@code false} to use only the minority
     * class.
     */
    public BorderlineSMOTE(Classifier baseClassifier, boolean majorityInterpolation)
    {
        this(baseClassifier, new EuclideanDistance(), majorityInterpolation);
    }
    
    /**
     * Creates a new Borderline-SMOTE model that will over-sample the minority
     * classes so that there is a balanced number of data points in each class.
     *
     * @param baseClassifier the base classifier to use after the SMOTEing is
     * done.
     * @param dm the distance metric to use for determining nearest neighbors
     * @param majorityInterpolation {@code true} if synthetic examples should
     * use the majority class as well, or {@code false} to use only the minority
     * class.
     */
    public BorderlineSMOTE(Classifier baseClassifier, DistanceMetric dm, boolean majorityInterpolation)
    {
        this(baseClassifier, dm, 1.0, majorityInterpolation);
    }
    
    /**
     * Creates a new Borderline-SMOTE model.
     *
     * @param baseClassifier the base classifier to use after the SMOTEing is
     * done.
     * @param dm the distance metric to use for determining nearest neighbors
     * @param targetRatio the desired ratio of samples for each class with respect to the majority class. 
     * @param majorityInterpolation {@code true} if synthetic examples should
     * use the majority class as well, or {@code false} to use only the minority
     * class.
     */
    public BorderlineSMOTE(Classifier baseClassifier, DistanceMetric dm, double targetRatio, boolean majorityInterpolation)
    {
        this(baseClassifier, dm, 5, targetRatio, majorityInterpolation);
    }

    /**
     * Creates a new SMOTE object
     *
     * @param baseClassifier the base classifier to use after the SMOTEing is
     * done.
     * @param dm the distance metric to use for determining nearest neighbors
     * @param smoteNeighbors the number of neighbors to look at when
     * interpolating points
     * @param targetRatio the desired ratio of samples for each class with
     * respect to the majority class.
     * @param majorityInterpolation {@code true} if synthetic examples should
     * use the majority class as well, or {@code false} to use only the minority
     * class.
     */
    public BorderlineSMOTE(Classifier baseClassifier, DistanceMetric dm, int smoteNeighbors, double targetRatio, boolean majorityInterpolation)
    {
        super(baseClassifier, dm, smoteNeighbors, targetRatio);
        setMajorityInterpolation(majorityInterpolation);
    }

    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public BorderlineSMOTE(BorderlineSMOTE toCopy)
    {
        super((SMOTE)toCopy);
        this.majorityInterpolation = toCopy.majorityInterpolation;
    }

    /**
     * Sets whether the generation of synthetic samples can make use of the
     * majority samples (i.e., from other classes) or not. The use of majority
     * samples is "Borderline-SMOTE2" in the original paper. If majority samples
     * are not used, it is equivalent to "Borderline-SMOTE1".
     *
     * @param majorityInterpolation {@code true} if majority samples should be
     * used for interpolation, and {@code false} if only minority samples should
     * be used.
     */
    public void setMajorityInterpolation(boolean majorityInterpolation)
    {
        this.majorityInterpolation = majorityInterpolation;
    }

    /**
     * 
     * @return {@code true} if majority samples should be
     * used for interpolation, and {@code false} if only minority samples should
     * be used.
     */
    public boolean isMajorityInterpolation()
    {
        return majorityInterpolation;
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
        
        //Put ALL the vectors intoa single VC paired with their class label
        VectorCollection<Vec> VC_all = new DefaultVectorCollection<>(dm, vAll, parallel);
        
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
            //Step 1. For every p ii =( 1,2,..., pnum) in the minority class P, 
            //we calculate its m nearest neighbors from the whole training set T
            List<List<Integer>> allNeighbors = new ArrayList<>();
            List<List<Double>> allDistances = new ArrayList<>();
            VC_all.search(V_id, smoteNeighbors+1, allNeighbors, allDistances, parallel);
            /**
             * A list of the vectors for only the neighbors who were not members
             * of the same class. Used when majorityInterpolation is true
             */
            final List<List<Vec>> otherClassSamples = new ArrayList<>();
            if(majorityInterpolation)
                for(List<Integer> tmp : allNeighbors)
                    otherClassSamples.add(new ArrayList<>(smoteNeighbors));


            //Step 2. 
            final IntList danger_id = new IntList();
            
            for(int i = 0; i < VC_id.size(); i++)
            {
                int same_class = 0;
                List<Integer> neighors_of_i = allNeighbors.get(i);
                for(int j = 1; j < smoteNeighbors+1; j++)
                {
                    if(classID == dataSet.getDataPointCategory(neighors_of_i.get(j)))
                        same_class++;
                    else
                    {
                        if(majorityInterpolation)
                            otherClassSamples.get(i).add(VC_all.get(neighors_of_i.get(j)));
                    }
                }
                //are you in the DANZER ZONE!?
                
                //ratio of how many "majority" examples vs minority
                //we treat any other class as the "majority" to generalize to the multi-class case
                //for binary, will be equivalent to original paper
                double sOm = 1.0-same_class/(double)smoteNeighbors;
                if(0.5 <= sOm && sOm < 1.0)
                    danger_id.add(i);
                //else, you are either easily misclassified or easily classified - and thus skipped
            }
            
            
            //find all the nearest neighbors for each point so we know who to interpolate with
            List<List<Integer>> idNeighbors = new ArrayList<>();
            List<List<Double>> idDistances = new ArrayList<>();
            VC_id.search(VC_id, smoteNeighbors+1, idNeighbors, idDistances, parallel);
            
            ParallelUtils.run(parallel, samplesNeeded, (start, end)->
            {
                Random rand = RandomUtil.getRandom();
                List<DataPoint> local_new = new ArrayList<>();
                for (int i = start; i < end; i++)
                {
                    int sampleIndex;
                    if (danger_id.isEmpty())//danger zeon was empty? Fall back to SMOTE style
                        sampleIndex = i % V_id.size();
                    else
                        sampleIndex = danger_id.getI(i % danger_id.size());
                    Vec vec_nn;

                    //which of the neighbors should we use?
                    //Shoulwe we interpolate withing class or outside of or class?
                    boolean useOtherClass = rand.nextBoolean() && majorityInterpolation && !danger_id.isEmpty();

                    if (useOtherClass)
                    {
                        List<Vec> candidates = otherClassSamples.get(sampleIndex);
                        vec_nn = candidates.get(rand.nextInt(candidates.size()));
                    }
                    else
                    {
                        int nn = rand.nextInt(smoteNeighbors) + 1;//index 0 is ourself
                        vec_nn = VC_id.get(idNeighbors.get(sampleIndex).get(nn));
                    }
                    double gap = rand.nextDouble();
                    if (useOtherClass)
                        gap /= 2;//now in the range of [0, 0.5), so that the synthetic point is mostly of the minority class of interest

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
    public BorderlineSMOTE clone()
    {
        return new BorderlineSMOTE(this);
    }

}
