package jsat.classifiers.boosting;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;

/**
 * Arc-x4 is a ensemble-classifier that performs re-weighting of the data points 
 * based on the total number of errors that have occurred for the data point.
 * <br><br>
 * See: Breiman, L. (1998). <i>Arcing Classifiers</i>. The Annals of Statistics,
 * 26(3), 801–824.
 * 
 * @author Edward Raff
 */
public class ArcX4 implements Classifier, Parameterized
{

	private static final long serialVersionUID = 3831448932874147550L;
	private Classifier weakLearner;
    private int iterations;
    
    private double coef = 1;
    private double expo = 4;
    
    private CategoricalData predicing;
    private Classifier[] hypoths;
    
    /**
     * Creates a new Arc-X4 classifier 
     * 
     * @param weakLearner the weak learner to use
     * @param iterations the number of iterations to perform
     */
    public ArcX4(Classifier weakLearner, int iterations)
    {
        setWeakLearner(weakLearner);
        setIterations(iterations);
    }

    /**
     * Sets the weak learner used at each iteration of learning
     * @param weakLearner the weak learner to use
     */
    public void setWeakLearner(Classifier weakLearner)
    {
        if(!weakLearner.supportsWeightedData())
            throw new RuntimeException("Weak learners must support weighted data samples");
        this.weakLearner = weakLearner;
    }

    /**
     * Returns the weak learner used
     * @return the weak learner used
     */
    public Classifier getWeakLearner()
    {
        return weakLearner;
    }

    /**
     * Sets the number of iterations to perform
     * @param iterations the number of iterations to do
     */
    public void setIterations(int iterations)
    {
        this.iterations = iterations;
    }

    /**
     * Returns the number of iterations to learn
     * @return the number of iterations to learn
     */
    public int getIterations()
    {
        return iterations;
    }

    /**
     * Weights are updated as 1+coef*errors<sup>expo</sup>. This sets the 
     * coefficient used to update the errors
     * 
     * @param coef the multiplicative factor on the errors in weight construction
     */
    public void setCoefficient(double coef)
    {
        if(coef <= 0 || Double.isInfinite(coef) || Double.isNaN(coef))
            throw new ArithmeticException("The coefficient must be a positive constant");
        this.coef = coef;
    }

    /**
     * Returns the coefficient use when re-weighting
     * @return the coefficient use when re-weighting
     */
    public double getCoefficient()
    {
        return coef;
    }

    /**
     * Weights are updated as 1+coef*errors<sup>expo</sup>. This sets the 
     * exponent used to update the errors
     * @param expo the exponent to use
     */
    public void setExponent(double expo)
    {
        if(expo <= 0 || Double.isInfinite(expo) || Double.isNaN(expo))
            throw new ArithmeticException("The exponent must be a positive constant");
        this.expo = expo;
    }

    /**
     * Returns the exponent used when re-weighting 
     * @return the exponent used when re-weighting 
     */
    public double getExponent()
    {
        return expo;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(predicing.getNumOfCategories());
        
        for(Classifier hypoth : hypoths)
            cr.incProb(hypoth.classify(data).mostLikely(), 1.0);
        
        cr.normalize();
        
        return cr;
    }
    
    private class Tester implements Runnable
    {
        final ClassificationDataSet cds;
        final int[] errors;
        final int start;
        final int end;
        final Classifier hypoth;
        final CountDownLatch latch;

        public Tester(ClassificationDataSet cds, int[] errors, int start, int end, Classifier hypoth, CountDownLatch latch)
        {
            this.cds = cds;
            this.errors = errors;
            this.start = start;
            this.end = end;
            this.hypoth = hypoth;
            this.latch = latch;
        }
        

        @Override
        public void run()
        {
            for(int i = start; i < end; i++)
                if(hypoth.classify(cds.getDataPoint(i)).mostLikely() != cds.getDataPointCategory(i))
                    errors[i]++;
            latch.countDown();
        }
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        //Create a low memory clone that only has different dataPoint Objects to save space
        ClassificationDataSet cds = dataSet.shallowClone();
        for(int i = 0; i < cds.getSampleSize(); i++)
        {
            DataPoint dp = cds.getDataPoint(i);
            cds.setDataPoint(i, new DataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), dp.getCategoricalData()));
        }
        
        //Everyone starts with no errors
        int[] errors = new int[cds.getSampleSize()];
        
        final int blockSize = errors.length / SystemInfo.LogicalCores;
        
        hypoths = new Classifier[iterations];
        for(int t = 0; t < hypoths.length; t++)
        {
            for(int i = 0; i < cds.getSampleSize(); i++)
                cds.getDataPoint(i).setWeight(1+coef*Math.pow(errors[i], expo));
            
            Classifier hypoth = weakLearner.clone();
            
            if(threadPool == null || threadPool instanceof FakeExecutor)
                hypoth.trainC(cds);
            else
                hypoth.trainC(cds, threadPool);
            
            hypoths[t] = hypoth;
            if(blockSize > 0)
            {
                int extra = errors.length % SystemInfo.LogicalCores;
                CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
                int start = 0;
                while(start < errors.length)
                {
                    int end = start + blockSize;
                    if(extra-- > 0)
                        end++;
                    threadPool.submit(new Tester(cds, errors, start, end, hypoth, latch));
                    start = end;
                }
                try
                {
                    latch.await();
                }
                catch (InterruptedException ex)
                {
                    Logger.getLogger(ArcX4.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            else//not enough to do in parallel
            {
                new Tester(cds, errors, 0, errors.length, hypoth, new CountDownLatch(1)).run();
            }
            
        }
        
        this.predicing = cds.getPredicting();
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public ArcX4 clone()
    {
        ArcX4 clone = new ArcX4(weakLearner.clone(), iterations);
        
        clone.coef = this.coef;
        clone.expo = this.expo;
        
        if(this.predicing != null)
            clone.predicing = this.predicing.clone();
        if(this.hypoths != null)
        {
            clone.hypoths = new Classifier[this.hypoths.length];
            for(int i = 0; i < clone.hypoths.length; i++)
                clone.hypoths[i] = this.hypoths[i].clone();
        }
        
        return clone;
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
