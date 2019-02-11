
package jsat.classifiers;

import java.util.*;
import java.util.concurrent.atomic.DoubleAdder;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.*;
import jsat.utils.DoubleList;
import jsat.utils.concurrent.ParallelUtils;

/**
 *
 * @author Edward Raff
 */
public class Rocchio implements Classifier
{

    private static final long serialVersionUID = 889524967453326516L;
    private List<Vec> rocVecs;
    private final DistanceMetric dm;
    private List<Double> rocCache;

    public Rocchio()
    {
        this(new EuclideanDistance());
    }

    public Rocchio(DistanceMetric dm)
    {
        this.dm = dm;
        rocVecs = null;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(rocVecs.size());
        double sum = 0;
        
        Vec target = data.getNumericalValues();
        List<Double> qi = dm.getQueryInfo(target);
        
        //Record the average for each class
        for(int i = 0; i < rocVecs.size(); i++)
        {
            double distance = dm.dist(i, target, qi, rocVecs, rocCache);
            sum += distance;
            cr.setProb(i, distance);
        }
        
        //now scale, set them all to 1-distance/sumOfDistances. We will call that out probablity
        for(int i = 0; i < rocVecs.size(); i++)
            cr.setProb(i, 1.0 - cr.getProb(i) / sum);
        
        return cr;
    }

    @Override
    public void train(ClassificationDataSet dataSet, boolean parallel)
    {
        if(dataSet.getNumCategoricalVars() != 0)
            throw new FailedToFitException("Classifier requires all variables be numerical");
        int C = dataSet.getClassSize();
        rocVecs = new ArrayList<>(C);
        
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, parallel);
        
        
        //dimensions
        int d = dataSet.getNumNumericalVars();
                
        //Set up a bunch of threads to add vectors together in the background
	DoubleAdder totalWeight = new DoubleAdder();
        rocVecs = new ArrayList<>(Arrays.asList(ParallelUtils.run(parallel, dataSet.size(), 
        //partial sum for each class
        (int start, int end) -> 
        {
            //find class vec sums
            Vec[] local_roc = new Vec[C];
            for(int i = 0; i < C; i++) 
                local_roc[i] = new DenseVector(d);
            for(int i  = start; i < end; i++)
            {
		double w = dataSet.getWeight(i);
                local_roc[dataSet.getDataPointCategory(i)].mutableAdd(w, dataSet.getDataPoint(i).getNumericalValues());
		totalWeight.add(w);
            }
            return local_roc;
        },
        //reduce down to a final sum per class
        (Vec[] t, Vec[] u) -> 
        {
            for(int i = 0; i < t.length; i++)
                t[i].mutableAdd(u[i]);
            return t;
        })));
        //Normalize each vec so we have the correct values in the end
        double[] priors = dataSet.getPriors();
        for(int i = 0; i < C; i++)
            rocVecs.get(i).mutableDivide(totalWeight.sum()*priors[i]);
        //prep cache for inference
        rocCache = dm.getAccelerationCache(rocVecs, parallel);
    }
    
    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }

    @Override
    public Rocchio clone()
    {
        Rocchio copy = new Rocchio(this.dm);
        if(this.rocVecs != null)
        {
            copy.rocVecs = new ArrayList<>(this.rocVecs.size());
            for(Vec v : this.rocVecs)
                copy.rocVecs.add(v.clone());
	    copy.rocCache = new DoubleList(rocCache);
        }
        return copy;
    }
    
}
