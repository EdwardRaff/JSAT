
package jsat.classifiers.boosting;

import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import jsat.classifiers.ClassificationDataSet;
import jsat.exceptions.FailedToFitException;
import jsat.regression.Regressor;
import static jsat.utils.SystemInfo.*;

/**
 * An extension to the original LogitBoost algorithm for parallel training. 
 * This comes at an increase in classification time. 
 * <br>
 * Note: LogitBoost is a semi unstable algorithm, and this method does in fact increase the instability. 
 * In most cases, similar classification results will be obtained, however - under some circumstances 
 * the performance may be significantly degraded. Especially if there is insufficient data to distribute
 * for parallel computation. The results for LogitBoost seem to be over stated in the original paper. 
 * <br>
 * See: <i>Scalable and Parallel Boosting with MapReduce</i>, Indranil Palit and Chandan K. Reddy, IEEE Transactions on Knowledge and Data Engineering
 * @author Edward Raff
 */
public class LogitBoostPL extends LogitBoost
{

	private static final long serialVersionUID = -7932049860430324903L;

	public LogitBoostPL(Regressor baseLearner, int M)
    {
        super(baseLearner, M);
    }

    public LogitBoostPL(int M)
    {
        super(M);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        /*
         * Implementation Note:
         * In the original paper, we sort the weak hypotheses of the LogBoost workers 
         * by their unweighted accuracy. Then each regressor is averaged in its sorted
         * group index. However, if we have M worked, each merged regressor is the sum
         * of the parts divided by M, ie: the average. Applythis to the whole of the 
         * data set, we get the same result as if we add all regressors to the data 
         * set, getting M*Iteration hypothesis with the sume scaled by 1/(M) 
         * isntead of Iteration hypothesis scaled by 1/2
         * 
         * Applied this, we can simplify the implementation and avoid M sortings
         *
         */
        
        List<ClassificationDataSet> subSets = dataSet.cvSet(LogicalCores);
        List<Future<LogitBoost>> futuerBoosts = new ArrayList<Future<LogitBoost>>(LogicalCores);

        for (int i = 0; i < LogicalCores; i++)
        {
            final ClassificationDataSet subSet = subSets.get(i);
            
            futuerBoosts.add(threadPool.submit(new Callable<LogitBoost>()
            {

                public LogitBoost call() throws Exception
                {
                    LogitBoost boost = new LogitBoost(baseLearner.clone(), getMaxIterations());

                    boost.trainC(subSet);

                    return boost;
                }
            }));
        }
        
        try
        {
            this.baseLearners = new ArrayList<Regressor>(LogicalCores * getMaxIterations());
            this.fScaleConstant = 1.0 / LogicalCores;
            //We now collect all our regressors
            for(Future<LogitBoost> boost :  futuerBoosts)
                this.baseLearners.addAll(boost.get().baseLearners);
            
        }
        catch (InterruptedException interruptedException)
        {
            throw new FailedToFitException(interruptedException);
        }
        catch (ExecutionException executionException)
        {
            throw new FailedToFitException(executionException);
        }

    }

    @Override
    public LogitBoostPL clone()
    {
        LogitBoostPL clone = new LogitBoostPL(getMaxIterations());
        clone.setzMax(getzMax());
        if(this.baseLearner != null) 
            clone.baseLearner = this.baseLearner.clone();
        if(this.baseLearners != null)
        {
            clone.baseLearners = new ArrayList<Regressor>(this.baseLearners.size());
            for(Regressor r :  baseLearners)
                clone.baseLearners.add(r.clone());
        }
        return clone;
    }
    
    
    
}
