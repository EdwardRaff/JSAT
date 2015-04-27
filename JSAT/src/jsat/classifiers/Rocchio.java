
package jsat.classifiers;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.*;
import jsat.utils.FakeExecutor;

/**
 *
 * @author Edward Raff
 */
public class Rocchio implements Classifier
{

	private static final long serialVersionUID = 889524967453326517L;
	private List<Vec> rocVecs;
    private final DistanceMetric dm;
    private final DenseSparseMetric dsdm;
    private double[] summaryConsts;

    public Rocchio()
    {
        this(new EuclideanDistance());
    }

    public Rocchio(DistanceMetric dm)
    {
        this.dm = dm;
        this.dsdm = dm instanceof DenseSparseMetric ? (DenseSparseMetric) dm : null;
        rocVecs = null;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(rocVecs.size());
        double sum = 0;
        
        Vec target = data.getNumericalValues();
        
        //Record the average for each class
        for(int i = 0; i < rocVecs.size(); i++)
        {
            double distance;
            if (summaryConsts == null)
                distance = dm.dist(rocVecs.get(i), target);
            else
                distance = dsdm.dist(summaryConsts[i], rocVecs.get(i), target);
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

        public RocchioAdder(CountDownLatch latch, int index, Vec rocchioVec, List<DataPoint> input)
        {
            this.latch = latch;
            this.index = index;
            this.rocchioVec = rocchioVec;
            this.input = input;
            weightSum = 0;
        }

        double weightSum;

        final CountDownLatch latch;
        final Vec rocchioVec;
        final List<DataPoint> input;
        final int index;

        @Override
        public void run()
        {
            for(DataPoint dp : input)
            {
                double w = dp.getWeight();
                Vec v = dp.getNumericalValues();
                weightSum += w;
                rocchioVec.mutableAdd(w, v);
            }
            
            rocchioVec.mutableDivide(weightSum);
            if(dsdm != null)
                summaryConsts[index] = dsdm.getVectorConstant(rocchioVec);
            latch.countDown();
        }

    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        if(dataSet.getNumCategoricalVars() != 0)
            throw new FailedToFitException("Classifier requires all variables be numerical");
        int N = dataSet.getClassSize();
        rocVecs = new ArrayList<Vec>(N);
        
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadPool);
        
        //dimensions
        int d = dataSet.getNumNumericalVars();
        
        summaryConsts = new double[d];
        
        //Set up a bunch of threads to add vectors together in the background
        CountDownLatch cdl = new CountDownLatch(N);
        for(int i = 0; i < N; i++)
        {
            final Vec rochVec = new DenseVector(d);
            rocVecs.add(rochVec);
            
            threadPool.submit(new RocchioAdder(cdl, i, rochVec, dataSet.getSamples(i)));
        }
        
        try
        {
            cdl.await();
        }
        catch (InterruptedException ex)
        {
        }
        
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
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
            copy.rocVecs = new ArrayList<Vec>(this.rocVecs.size());
            for(Vec v : this.rocVecs)
                copy.rocVecs.add(v.clone());
        }
        if(this.summaryConsts != null)
            copy.summaryConsts = Arrays.copyOf(summaryConsts, summaryConsts.length);
        return copy;
    }
    
}
