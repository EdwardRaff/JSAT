
package jsat.classifiers;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.svm.PlatSMO;
import jsat.distributions.kernels.LinearKernel;
import jsat.utils.FakeExecutor;

/**
 * This classifier turns any classifier, specifically binary classifiers, into 
 * multi-class classifiers. For a problem with <i>k</i> target classes, OneVsALl
 * will create <i>k</i> different classifiers. Each one is a reducing of one 
 * class against all other classes. Then all <i>k</i> classifiers's results are
 * summed to produce a final classifier
 * 
 * @author Edward Raff
 */
public class OneVSAll implements Classifier
{
    private volatile Classifier[] oneVsAlls;
    private Classifier baseClassifier;
    private CategoricalData predicting;
    private boolean concurrentTraining;
    
    /**
     * Creates a new One VS All classifier. 
     * 
     * @param baseClassifier the base classifier to replicate
     * @param concurrentTraining controls whether or not classifiers are trained 
     * simultaneously or using sequentially using their 
     * {@link Classifier#trainC(jsat.classifiers.ClassificationDataSet, java.util.concurrent.ExecutorService) } method.  
     * @see #setConcurrentTraining(boolean) 
     */
    public OneVSAll(Classifier baseClassifier, boolean concurrentTraining)
    {
        this.baseClassifier = baseClassifier;
        this.concurrentTraining = concurrentTraining;
    }

    /**
     * Controls what method of parallel training to use when 
     * {@link #trainC(jsat.classifiers.ClassificationDataSet, java.util.concurrent.ExecutorService) } 
     * is called. If set to true, each of the <i>k</i> classifiers will be trained in parallel, using
     * their serial algorithms. If set to false, the <i>k</i> classifiers will be trained sequentially, 
     * calling the {@link Classifier#trainC(jsat.classifiers.ClassificationDataSet, java.util.concurrent.ExecutorService) }
     * for each classifier. <br>
     * <br>
     * This should be set to true for classifiers that do not support parallel training.<br>
     * Setting this to true also uses <i>k</i> times the memory, since each classifier is being created and trained at the same time. 
     * @param concurrentTraining whether or not to train the classifiers in parallel 
     */
    public void setConcurrentTraining(boolean concurrentTraining)
    {
        this.concurrentTraining = concurrentTraining;
    }
    
    
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());
        for(int i = 0; i < predicting.getNumOfCategories(); i++)
            if(oneVsAlls[i].classify(data).getProb(0) > 0)
            {
                double tmp = oneVsAlls[i].classify(data).getProb(0);
                cr.setProb(i, tmp);
            }
        
        cr.normalize();
        return cr;
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        oneVsAlls = new Classifier[dataSet.getPredicting().getNumOfCategories()];
        
        predicting = dataSet.getPredicting();
        
        List<List<DataPoint>> categorized = new ArrayList<List<DataPoint>>();
        for(int i = 0; i < oneVsAlls.length; i++)
        {
            List<DataPoint> tmp = dataSet.getSamples(i);
            ArrayList<DataPoint> oneCat = new ArrayList<DataPoint>(tmp.size());
            oneCat.addAll(tmp);
            categorized.add(oneCat);
        }
        
        int numer = dataSet.getDataPoint(0).getNumericalValues().length();
        CategoricalData[] categories = dataSet.getCategories();
        //Latch only used when all the classifiers are trained in parallel 
        final CountDownLatch latch = new CountDownLatch(oneVsAlls.length);
        for(int i = 0; i < oneVsAlls.length; i++)
        {
            final ClassificationDataSet cds = 
                    new ClassificationDataSet(numer, categories, new CategoricalData(2));
            for(DataPoint dp : categorized.get(i))//add the ones
                cds.addDataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), 0);
            //Add all the 'others'
            for(int j = 0; j < categorized.size(); j++)
                if(j != i)
                    for(DataPoint dp: categorized.get(j))
                        cds.addDataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), 1);

            if(!concurrentTraining)
            {
                baseClassifier.trainC(cds, threadPool);
                oneVsAlls[i] = baseClassifier.clone();
            }
            else
            {
                final Classifier aClassifier = baseClassifier.clone();
                final int ii = i;
                threadPool.submit(new Runnable() {

                    public void run()
                    {
                        aClassifier.trainC(cds);
                        oneVsAlls[ii] = aClassifier;
                        latch.countDown();
                    }
                });
            }
            
        }

        if (concurrentTraining)
            try
            {
                latch.await();
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(OneVSAll.class.getName()).log(Level.SEVERE, null, ex);
            }

        
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    @Override
    public Classifier clone()
    {
        OneVSAll clone = new OneVSAll(baseClassifier.clone(), concurrentTraining);
        if(this.predicting != null)
            clone.predicting = this.predicting.clone();
        if(this.oneVsAlls != null)
        {
            clone.oneVsAlls = new Classifier[this.oneVsAlls.length];
            for(int i = 0; i < oneVsAlls.length; i++)
                if(this.oneVsAlls[i] != null)
                    clone.oneVsAlls[i] = this.oneVsAlls[i].clone();
        }
        return clone;
    }

    public boolean supportsWeightedData()
    {
        return baseClassifier.supportsWeightedData();
    }
    
}
