package jsat.classifiers;

import java.util.concurrent.ExecutorService;

/**
 * This interface is meant for models that support efficient warm starting from 
 * the solution of a previous model. Training with a warm start means that 
 * instead of solving the problem from scratch, the code can use a previous 
 * solution to start closer towards its goal. <br>
 * <br>
 * Some algorithm may be able to warm start from solutions of the same form, 
 * even if they were trained by a different algorithm. Other algorithms may only
 * be able to warm start from the same algorithm. There may also be restrictions 
 * that the warm start can only be from a solution trained on the exact same 
 * data set. The latter case is indicated by the {@link #warmFromSameDataOnly()}
 * method. <br>
 * <br>
 * Just because a classifier fits the type that the warm start interface states
 * doesn't mean that it is a valid classifier to warm start from. <i>Classifiers
 * of the same class trained on the same data must <b>always</b> be valid to
 * warm start from. </i>
 * <br>
 * <br>
 * Note: The use of this class is still under development, and may change in the
 * future. 
 *
 * @author Edward Raff
 */
public interface WarmClassifier extends Classifier
{
    /**
     * Some models can only be warm started from a solution trained on the 
     * exact same data set as the model it is warm starting from. If this is the 
     * case {@code true} will be returned. The behavior for training on a 
     * different data set when this is defined is undefined. It may cause an 
     * error, or it may cause the algorithm to take longer or reach a worse 
     * solution. <br>
     * When {@code true}, it is important that the data set be unaltered - this 
     * includes mutating the values stored or re-arranging the data points 
     * within the data set. 
     * 
     * @return {@code true} if the algorithm can only be warm started from the 
     * model trained on the exact same data set. 
     */
    public boolean warmFromSameDataOnly();
    
    /**
     * Trains the classifier and constructs a model for classification using the 
     * given data set. If the training method knows how, it will used the 
     * <tt>threadPool</tt> to conduct training in parallel. This method will 
     * block until the training has completed.
     * 
     * @param dataSet the data set to train on
     * @param warmSolution the solution to use to warm start this model
     * @param threadPool the source of threads to use.
     */
    public void trainC(ClassificationDataSet dataSet, Classifier warmSolution, ExecutorService threadPool);
    
     /**
     * Trains the classifier and constructs a model for classification using the 
     * given data set. 
     * 
     * @param dataSet the data set to train on
     * @param warmSolution the solution to use to warm start this model
     */
    public void trainC(ClassificationDataSet dataSet, Classifier warmSolution);
}
