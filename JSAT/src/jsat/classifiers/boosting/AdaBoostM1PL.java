package jsat.classifiers.boosting;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.MajorityVote;
import jsat.exceptions.FailedToFitException;
import jsat.utils.DoubleList;
import jsat.utils.IndexTable;
import static jsat.utils.SystemInfo.*;

/**
 * An extension to the original AdaBoostM1 algorithm for parallel training. 
 * This comes at an increase in classification time. 
 * <br>
 * See: <i>Scalable and Parallel Boosting with MapReduce</i>, Indranil Palit and Chandan K. Reddy, IEEE Transactions on Knowledge and Data Engineering
 * 
 * 
 * @author Edward Raff
 */
public class AdaBoostM1PL extends AdaBoostM1
{
    

	private static final long serialVersionUID = 1027211688101553766L;


	public AdaBoostM1PL(Classifier weakLearner, int maxIterations)
    {
        super(weakLearner, maxIterations);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        predicting = dataSet.getPredicting();

        //Contains the Boostings we performed on subsets of the data 
        List<Future<AdaBoostM1>> futureBoostings = new ArrayList<Future<AdaBoostM1>>(LogicalCores);
        
        //We want an even, random split of the data into groups for each learner, the CV set does that for us! 
        List<ClassificationDataSet> subSets = dataSet.cvSet(LogicalCores);
        for(int i = 0; i < LogicalCores; i++)
        {
            final AdaBoostM1 learner = new AdaBoostM1(getWeakLearner().clone(), getMaxIterations());
            final ClassificationDataSet subDataSet = subSets.get(i);
            futureBoostings.add(threadPool.submit(new Callable<AdaBoostM1>() {

                public AdaBoostM1 call() throws Exception
                {
                    learner.trainC(subDataSet);
                    return learner;
                }
            }));
        }
        
        try
        {
            List<AdaBoostM1> boosts = new ArrayList<AdaBoostM1>(LogicalCores);
            List<List<Double>> boostWeights = new ArrayList<List<Double>>(LogicalCores);
            List<List<Classifier>> boostWeakLearners = new ArrayList<List<Classifier>>(LogicalCores);
            //Contains the tables to view the weights in sorted order 
            List<IndexTable> sortedViews = new ArrayList<IndexTable>(LogicalCores);
            for(Future<AdaBoostM1> futureBoost : futureBoostings)
            {
                AdaBoostM1 boost =  futureBoost.get();
                boosts.add(boost);
                sortedViews.add(new IndexTable(boost.hypWeights));
                boostWeights.add(boost.hypWeights);
                boostWeakLearners.add(boost.hypoths);
                
            }
            
            //Now we merge the results into our new classifer 
            int T = boosts.get(0).getMaxIterations();
            hypoths = new ArrayList<Classifier>(T);
            hypWeights = new DoubleList(T);
            for (int i = 0; i < T; i++)
            {
                Classifier[] toMerge = new Classifier[LogicalCores];
                double weight = 0.0;
                for (int m = 0; m < LogicalCores; m++)
                {
                    int mSortedIndex = sortedViews.get(m).index(i);
                    toMerge[m] = boostWeakLearners.get(m).get(mSortedIndex);
                    weight += boostWeights.get(m).get(mSortedIndex);
                }
                weight /= LogicalCores;
                hypWeights.add(weight);
                hypoths.add(new MajorityVote(toMerge));
            }
        }
        catch(Exception ex )
        {
            throw new FailedToFitException(ex);
        }
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        super.trainC(dataSet, null);
    }


    @Override
    public AdaBoostM1PL clone()
    {
        AdaBoostM1PL copy = new AdaBoostM1PL( getWeakLearner().clone(), getMaxIterations());
        if(hypWeights != null)
            copy.hypWeights = new DoubleList(this.hypWeights);
        if(this.hypoths != null)
        {
            copy.hypoths = new ArrayList<Classifier>(this.hypoths.size());
            for(int i = 0; i < this.hypoths.size(); i++)
                copy.hypoths.add(this.hypoths.get(i).clone());
        }
        if(this.predicting != null)
            copy.predicting = this.predicting.clone();
        return copy;
    }

}
