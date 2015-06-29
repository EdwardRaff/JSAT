package jsat.classifiers;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;

import jsat.utils.FakeExecutor;

/**
 * A One VS One classifier extends binary decision classifiers into multi-class 
 * decision classifiers. This is done by creating a binary-classification 
 * problem for every possible pair of classes, and then classifying by taking 
 * the result of all possible combinations and choosing the class that got the 
 * most results. This allows for a soft decision result, however, it often 
 * proves to be a meaningless soft boundary. 
 * 
 * @author Edward Raff
 */
public class OneVSOne implements Classifier, Parameterized
{

    private static final long serialVersionUID = 733202830281869416L;
    /**
     * Main binary classifier
     */
    @ParameterHolder
    protected Classifier baseClassifier;
    /**
     * Uper-diagonal matrix of classifiers sans the first index since a 
     * classifier vs itself is useless. First index is the first dimensions is 
     * the first class, 2nd index + the value of the first + 1 is the opponent 
     * class
     */
    protected Classifier[][] oneVone;
    private boolean concurrentTrain;
    protected CategoricalData predicting;

    /**
     * Creates a new One-vs-One classifier
     * @param baseClassifier the binary classifier to extend
     */
    public OneVSOne(Classifier baseClassifier)
    {
        this(baseClassifier, false);
    }
    
    /**
     * Creates a new One-vs-One classifier 
     * @param baseClassifier the binary classifier to extend
     * @param concurrentTrain <tt>true</tt> to have training of individual 
     * classifiers occur in parallel, <tt>false</tt> to have them use their 
     * native parallel training method. 
     */
    public OneVSOne(Classifier baseClassifier, boolean concurrentTrain)
    {
        this.baseClassifier = baseClassifier;
        this.concurrentTrain = concurrentTrain;
    }

    /**
     * Controls whether or not training of the several classifiers occurs concurrently or sequentually. 
     * @param concurrentTrain <tt>true</tt> to have training of individual 
     * classifiers occur in parallel, <tt>false</tt> to have them use their 
     * native parallel training method. 
     */
    public void setConcurrentTraining(boolean concurrentTrain)
    {
        this.concurrentTrain = concurrentTrain;
    }

    /**
     * 
     * @return <tt>true</tt> if training of individual 
     * classifiers occur in parallel, <tt>false</tt> they use their 
     * native parallel training method.
     */
    public boolean isConcurrentTraining()
    {
        return concurrentTrain;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());
        for (int i = 0; i < oneVone.length; i++)
        {
            for (int j = 0; j < oneVone[i].length; j++)
            {
                CategoricalResults subRes = oneVone[i][j].classify(data);
                int mostLikely = subRes.mostLikely();
                if(mostLikely == 0)
                    cr.incProb(i, 1.0);
                else
                    cr.incProb(i+j+1, 1.0);
            }
        }
        
        cr.normalize();
        return cr;
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool)
    {
        oneVone = new Classifier[dataSet.getClassSize()][];
        
        List<List<DataPoint>> dataByCategory = new ArrayList<List<DataPoint>>(dataSet.getClassSize());
        for(int i = 0; i < dataSet.getClassSize(); i++)
            dataByCategory.add(dataSet.getSamples(i));
        
        final CountDownLatch latch = new CountDownLatch(oneVone.length*(oneVone.length-1)/2);
        
        for(int i = 0; i < oneVone.length; i++)
        {
            oneVone[i] = new Classifier[oneVone.length-i-1];
            
            for(int j = 0; j < oneVone.length-i-1; j++)
            {
                final Classifier curClassifier = baseClassifier.clone();
                
                oneVone[i][j] = curClassifier;
                final int otherClass = j+i+1;
                CategoricalData subPred = new CategoricalData(2);
                subPred.setOptionName(dataSet.getPredicting().getOptionName(i), 0);
                subPred.setOptionName(dataSet.getPredicting().getOptionName(otherClass), 1);
                
                final ClassificationDataSet subDataSet = new ClassificationDataSet(dataSet.getNumNumericalVars(), dataSet.getCategories(), subPred); 
                
                //Fill sub data set with the two classes
                for(DataPoint dp : dataByCategory.get(i))
                    subDataSet.addDataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), 0);
                for(DataPoint dp : dataByCategory.get(otherClass))
                    subDataSet.addDataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), 1);

                if(!concurrentTrain)
                {
                    if(threadPool != null && !(threadPool instanceof FakeExecutor))
                        curClassifier.trainC(subDataSet, threadPool);
                    else
                        curClassifier.trainC(subDataSet);
                    continue;
                }
                //Else, concurrent
                threadPool.submit(new Runnable() {

                    @Override
                    public void run()
                    {
                        curClassifier.trainC(subDataSet);
                        latch.countDown();
                    }
                });
            }
        }
        
        if(concurrentTrain)
            try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(OneVSOne.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        predicting = dataSet.getPredicting();
        
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return baseClassifier.supportsWeightedData();
    }

    @Override
    public OneVSOne clone()
    {
        OneVSOne clone = new OneVSOne(baseClassifier.clone(), concurrentTrain);
        if (oneVone != null)
        {
            clone.oneVone = new Classifier[oneVone.length][];
            for (int i = 0; i < oneVone.length; i++)
            {
                clone.oneVone[i] = new Classifier[oneVone[i].length];
                for (int j = 0; j < oneVone[i].length; j++)
                    clone.oneVone[i][j] = oneVone[i][j].clone();
            }
        }
        if(predicting != null)
            clone.predicting = predicting.clone();

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
