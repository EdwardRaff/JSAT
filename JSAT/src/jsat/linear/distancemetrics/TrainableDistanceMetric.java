
package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;
import jsat.utils.FakeExecutor;

/**
 * Some Distance Metrics require information that can be learned from the data set. 
 * Trainable Distance Metrics support this facility, and algorithms that rely on 
 * distance metrics should check if their metric needs training. This is needed 
 * priming the distance metric on the whole data set and then performing cross 
 * validation would bias the results, as the metric would have been trained on
 * the testing set examples. 
 * 
 * @author Edward Raff
 */
abstract public class TrainableDistanceMetric implements DistanceMetric
{
    

	private static final long serialVersionUID = 6356276953152869105L;

	/**
     * Trains this metric on the given data set
     * @param <V> the type of vectors in the list
     * @param dataSet the data set to train on
     * @throws UnsupportedOperationException if the metric can not be trained from unlabeled data
     */
    abstract public <V extends Vec> void train(List<V> dataSet);
    /**
     * Trains this metric on the given data set
     * @param <V> the type of vectors in the list
     * @param dataSet the data set to train on
     * @param threadpool the source of threads for parallel training
     * @throws UnsupportedOperationException if the metric can not be trained from unlabeled data
     */
    abstract public <V extends Vec> void train(List<V> dataSet, ExecutorService threadpool);
    
    /**
     * Trains this metric on the given data set
     * @param dataSet the data set to train on
     * @throws UnsupportedOperationException if the metric can not be trained from unlabeled data
     */
    abstract public void train(DataSet dataSet);
    /**
     * Trains this metric on the given data set
     * @param dataSet the data set to train on
     * @param threadpool the source of threads for parallel training
     * @throws UnsupportedOperationException if the metric can not be trained from unlabeled data
     */
    abstract public void train(DataSet dataSet, ExecutorService threadpool);
    
    /**
     * Trains this metric on the given classification problem data set
     * @param dataSet the data set to train on 
     * @throws UnsupportedOperationException if the metric can not be trained from classification problems
     */
    abstract public void train(ClassificationDataSet dataSet);
    /**
     * Trains this metric on the given classification problem data set
     * @param dataSet the data set to train on 
     * @param threadpool the source of threads for parallel training
     * @throws UnsupportedOperationException if the metric can not be trained from classification problems
     */
    abstract public void train(ClassificationDataSet dataSet, ExecutorService threadpool);
    /**
     * Some metrics might be special purpose, and not trainable for all types of data sets or tasks. 
     * This method returns <tt>true</tt> if this metric supports training for classification 
     * problems, and <tt>false</tt> if it does not. <br>
     * If a metric can learn from unlabeled data, it must return <tt>true</tt> 
     * for this method. 
     * 
     * @return <tt>true</tt> if this metric supports training for classification 
     * problems, and <tt>false</tt> if it does not
     */
    abstract public boolean supportsClassificationTraining();
            
    /**
     * Trains this metric on the given regression problem data set
     * @param dataSet the data set to train on 
     * @throws UnsupportedOperationException if the metric can not be trained from regression problems
     */
    abstract public void train(RegressionDataSet dataSet);
    
    /**
     * Trains this metric on the given regression problem data set
     * @param dataSet the data set to train on 
     * @param threadpool the source of threads for parallel training
     * @throws UnsupportedOperationException if the metric can not be trained from regression problems
     */
    abstract public void train(RegressionDataSet dataSet, ExecutorService threadpool);
    /**
     * Some metrics might be special purpose, and not trainable for all types of data sets tasks. 
     * This method returns <tt>true</tt> if this metric supports training for regression 
     * problems, and <tt>false</tt> if it does not. <br>
     * If a metric can learn from unlabeled data, it must return <tt>true</tt> 
     * for this method. 
     * 
     * @return <tt>true</tt> if this metric supports training for regression 
     * problems, and <tt>false</tt> if it does not
     */
    abstract public boolean supportsRegressionTraining();
    
    /**
     * Returns <tt>true</tt> if the metric needs to be trained. This may be false if 
     * the metric allows the parameters to be specified beforehand. If the information 
     * was specified before hand, or does not need training, <tt>false</tt> is returned. 
     * 
     * @return <tt>true</tt> if the metric needs training, <tt>false</tt> if it does not. 
     */
    abstract public boolean needsTraining();
    
    @Override
    abstract public TrainableDistanceMetric clone();
    
    /**
     * Static helper method for training a distance metric only if it is needed. 
     * This method can be safely called for any Distance Metric.
     * 
     * @param dm the distance metric to train
     * @param dataset the data set to train from
     */
    public static void trainIfNeeded(DistanceMetric dm, DataSet dataset)
    {
        if(!(dm instanceof TrainableDistanceMetric))
            return;
        TrainableDistanceMetric tdm = (TrainableDistanceMetric) dm;
        if(!tdm.needsTraining())
            return;
        if(dataset instanceof RegressionDataSet)
            tdm.train((RegressionDataSet) dataset);
        else if(dataset instanceof ClassificationDataSet)
            tdm.train((ClassificationDataSet) dataset);
        else
            tdm.train(dataset);
    }
    
    /**
     * Static helper method for training a distance metric only if it is needed. 
     * This method can be safely called for any Distance Metric.
     * 
     * @param dm the distance metric to train
     * @param dataset the data set to train from
     * @param threadpool the source of threads for parallel training. May be 
     * <tt>null</tt>, in which case {@link #trainIfNeeded(jsat.linear.distancemetrics.DistanceMetric, jsat.DataSet) } 
     * is used instead.
     */
    public static void trainIfNeeded(DistanceMetric dm, DataSet dataset, ExecutorService threadpool)
    {
        if(threadpool == null || threadpool instanceof FakeExecutor)
        {
            trainIfNeeded(dm, dataset);
            return;
        }
        if(!(dm instanceof TrainableDistanceMetric))
            return;
        TrainableDistanceMetric tdm = (TrainableDistanceMetric) dm;
        if(!tdm.needsTraining())
            return;
        if(dataset instanceof RegressionDataSet)
            tdm.train((RegressionDataSet) dataset, threadpool);
        else if(dataset instanceof ClassificationDataSet)
            tdm.train((ClassificationDataSet) dataset, threadpool);
        else
            tdm.train(dataset, threadpool);
    }
    
    /**
     * Static helper method for training a distance metric only if it is needed. 
     * This method can be safely called for any Distance Metric.
     * 
     * @param <V> the type of vectors in the list
     * @param dm the distance metric to train
     * @param dataset the data set to train from
     */
    public static <V extends Vec> void trainIfNeeded(DistanceMetric dm, List<V> dataset)
    {
        if(!(dm instanceof TrainableDistanceMetric))
            return;
        TrainableDistanceMetric tdm = (TrainableDistanceMetric) dm;
        if(!tdm.needsTraining())
            return;
        if(dataset instanceof RegressionDataSet)
            tdm.train((RegressionDataSet) dataset);
        else if(dataset instanceof ClassificationDataSet)
            tdm.train((ClassificationDataSet) dataset);
        else
            tdm.train(dataset);
    }
    
    /**
     * 
     * @param <V> the type of vectors in the list
     * @param dm the distance metric to train
     * @param dataset the data set to train from
     * @param threadpool the source of threads for parallel training. May be 
     * <tt>null</tt>, in which case {@link #trainIfNeeded(jsat.linear.distancemetrics.DistanceMetric, java.util.List) }
     * is used instead.
     */
    public static <V extends Vec> void trainIfNeeded(DistanceMetric dm, List<V> dataset, ExecutorService threadpool)
    {
        if(threadpool == null || threadpool instanceof FakeExecutor)
        {
            trainIfNeeded(dm, dataset);
            return;
        }
        if(!(dm instanceof TrainableDistanceMetric))
            return;
        TrainableDistanceMetric tdm = (TrainableDistanceMetric) dm;
        if(!tdm.needsTraining())
            return;
        if(dataset instanceof RegressionDataSet)
            tdm.train((RegressionDataSet) dataset, threadpool);
        else if(dataset instanceof ClassificationDataSet)
            tdm.train((ClassificationDataSet) dataset, threadpool);
        else
            tdm.train(dataset, threadpool);
    }
}
