
package jsat.classifiers;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.FakeExecutor;

/**
 *
 * @author Edward Raff
 */
public class Rocchio implements Classifier
{

    private List<Vec> rocVecs;
    private final DistanceMetric dm;

    public Rocchio()
    {
        this(new EuclideanDistance());
    }

    public Rocchio(DistanceMetric dm)
    {
        this.dm = dm;
        rocVecs = null;
    }
    
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(rocVecs.size());
        double sum = 0;
        
        //Record the average for each class
        for(int i = 0; i < rocVecs.size(); i++)
        {
            double distance = dm.dist(rocVecs.get(i), data.getNumericalValues());
            sum += distance;
            cr.setProb(i, distance);
        }
        
        //now scale, set them all to 1-distance/sumOfDistances. We will call that out probablity
        for(int i = 0; i < rocVecs.size(); i++)
            cr.setProb(i, 1.0 - cr.getProb(i) / sum);
        
        return cr;
    }
    
    /*
     * Runnable that will sum all the vectors added until it is done
     * 
     */
    private class RocchioAdder implements Runnable
    {

        public RocchioAdder(CountDownLatch latch, Vec rocchioVec, List<DataPoint> input)
        {
            this.latch = latch;
            this.rocchioVec = rocchioVec;
            this.input = input;
            weightSum = 0;
        }

        double weightSum;

        final CountDownLatch latch;
        final Vec rocchioVec;
        final List<DataPoint> input;

        public void run()
        {
            for(DataPoint dp : input)
            {
                double w = dp.getWeight();
                Vec v = dp.getNumericalValues();
                if(w != 1.0)
                    v = v.multiply(w);//we cant alter the old one!
                weightSum += w;
                rocchioVec.mutableAdd(v);
            }
            
            rocchioVec.mutableDivide(weightSum);
            latch.countDown();
        }

    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        if(dataSet.getNumCategoricalVars() != 0)
            throw new RuntimeException("Classifier requires all variables be numerical");
        int N = dataSet.getPredicting().getNumOfCategories();
        rocVecs = new ArrayList<Vec>(N);
        
        //dimensions
        int d = dataSet.getNumNumericalVars();
        
        //Set up a bunch of threads to add vectors together in the background
        CountDownLatch cdl = new CountDownLatch(N);
        for(int i = 0; i < N; i++)
        {
            final Vec rochVec = new DenseVector(d);
            rocVecs.add(rochVec);
            
            threadPool.submit(new RocchioAdder(cdl, rochVec, dataSet.getSamples(i)));
        }
        
        try
        {
            cdl.await();
        }
        catch (InterruptedException ex)
        {
        }
        
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    public boolean supportsWeightedData()
    {
        return true;
    }

    public Classifier copy()
    {
        Rocchio copy = new Rocchio(this.dm);
        copy.rocVecs = new ArrayList<Vec>(this.rocVecs.size());
        for(Vec v : this.rocVecs)
            copy.rocVecs.add(v.copy());
        return copy;
    }
    
}
