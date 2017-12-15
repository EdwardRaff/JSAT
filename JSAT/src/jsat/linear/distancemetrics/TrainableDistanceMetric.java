
package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;

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
    public <V extends Vec> void train(List<V> dataSet)
    {
        train(dataSet, false);
    }
    
    /**
     * Trains this metric on the given data set
     * @param <V> the type of vectors in the list
     * @param dataSet the data set to train on
     * @param parallel {@code true} if multiple threads should be used for
     * training. {@code false} if it should be done in a single-threaded manner.
     * @throws UnsupportedOperationException if the metric can not be trained from unlabeled data
     */
    abstract public <V extends Vec> void train(List<V> dataSet, boolean parallel);
    
    /**
     * Trains this metric on the given data set
     * @param dataSet the data set to train on
     * @throws UnsupportedOperationException if the metric can not be trained from unlabeled data
     */
    public void train(DataSet dataSet)
    {
        train(dataSet, false);
    }
    
    /**
     * Trains this metric on the given data set
     * @param dataSet the data set to train on
     * @param parallel {@code true} if multiple threads should be used for
     * training. {@code false} if it should be done in a single-threaded manner.
     * @throws UnsupportedOperationException if the metric can not be trained from unlabeled data
     */
    abstract public void train(DataSet dataSet, boolean parallel);
    
    /**
     * Trains this metric on the given classification problem data set
     * @param dataSet the data set to train on 
     * @throws UnsupportedOperationException if the metric can not be trained from classification problems
     */
    public void train(ClassificationDataSet dataSet)
    {
        train(dataSet, false);
    }
    
    /**
     * Trains this metric on the given classification problem data set
     *
     * @param dataSet the data set to train on
     * @param parallel {@code true} if multiple threads should be used for
     * training. {@code false} if it should be done in a single-threaded manner.
     * @throws UnsupportedOperationException if the metric can not be trained
     * from classification problems
     */
    abstract public void train(ClassificationDataSet dataSet, boolean parallel);
    
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
     * @param parallel {@code true} if multiple threads should be used for
     * training. {@code false} if it should be done in a single-threaded manner.
     * @throws UnsupportedOperationException if the metric can not be trained from regression problems
     */
    abstract public void train(RegressionDataSet dataSet, boolean parallel);
    
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
        trainIfNeeded(dm, dataset, false);
    }
    
    /**
     * Static helper method for training a distance metric only if it is needed. 
     * This method can be safely called for any Distance Metric.
     * 
     * @param dm the distance metric to train
     * @param dataset the data set to train from
     * @param parallel {@code true} if multiple threads should be used for
     * training. {@code false} if it should be done in a single-threaded manner.
     */
    public static void trainIfNeeded(DistanceMetric dm, DataSet dataset, boolean parallel)
    {
        if(!(dm instanceof TrainableDistanceMetric))
            return;
        TrainableDistanceMetric tdm = (TrainableDistanceMetric) dm;
        if(!tdm.needsTraining())
            return;
        if(dataset instanceof RegressionDataSet)
            tdm.train((RegressionDataSet) dataset, parallel);
        else if(dataset instanceof ClassificationDataSet)
            tdm.train((ClassificationDataSet) dataset, parallel);
        else
            tdm.train(dataset, parallel);
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
     * @deprecated I WILL DELETE THIS METHOD SOON
     */
    public static void trainIfNeeded(DistanceMetric dm, DataSet dataset, ExecutorService threadpool)
    {
        //TODO I WILL DELETE, JUST STUBBING FOR NOW TO MAKE LIFE EASY AS I DO ONE CODE SECTION AT A TIME
        trainIfNeeded(dm, dataset);
    }
    
    /**
     * 
     * @param <V>
     * @param dm
     * @param dataset
     * @param threadpool 
     * @deprecated I WILL DELETE THIS METHOD SOON
     */
    public static <V extends Vec> void trainIfNeeded(DistanceMetric dm, List<V> dataset, ExecutorService threadpool)
    {
         //TODO I WILL DELETE, JUST STUBBING FOR NOW TO MAKE LIFE EASY AS I DO ONE CODE SECTION AT A TIME
        trainIfNeeded(dm, dataset, false);
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
        trainIfNeeded(dm, dataset, false);
    }
    
    /**
     * 
     * @param <V> the type of vectors in the list
     * @param dm the distance metric to train
     * @param dataset the data set to train from
     * @param parallel {@code true} if multiple threads should be used for
     * training. {@code false} if it should be done in a single-threaded manner.
     */
    public static <V extends Vec> void trainIfNeeded(DistanceMetric dm, List<V> dataset, boolean parallel)
    {
        if(!(dm instanceof TrainableDistanceMetric))
            return;
        TrainableDistanceMetric tdm = (TrainableDistanceMetric) dm;
        if(!tdm.needsTraining())
            return;
        if(dataset instanceof RegressionDataSet)
            tdm.train((RegressionDataSet) dataset, parallel);
        else if(dataset instanceof ClassificationDataSet)
            tdm.train((ClassificationDataSet) dataset, parallel);
        else
            tdm.train(dataset, parallel);
    }
}
