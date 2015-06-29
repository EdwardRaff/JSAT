package jsat.classifiers;

import java.util.PriorityQueue;

/**
 * Decision Directed Acyclic Graph (DDAG) classifier. DDAG extends
 * binary decision classifiers into multi-class decision classifiers. Unlike 
 * {@link OneVSOne}, DDAG results are hard classification decisions, and will 
 * not give probabilistic estimates. Accuracy is often very similar, but DDAG
 * classification speed can be significantly faster, as it does not evaluate all
 * possible combinations of classifiers. 
 * 
 * @author Edward Raff
 */
public class DDAG extends OneVSOne
{

    private static final long serialVersionUID = -9109002614319657144L;

    /**
     * Creates a new DDAG classifier to extend a binary classifier to handle multi-class problems. 
     * @param baseClassifier the binary classifier to extend
     * @param concurrentTrain <tt>true</tt> to have training of individual 
     * classifiers occur in parallel, <tt>false</tt> to have them use their 
     * native parallel training method.  
     */
    public DDAG(Classifier baseClassifier, boolean concurrentTrain)
    {
        super(baseClassifier, concurrentTrain);
    }

    /**
     * Creates a new DDAG classifier to extend a binary classifier to handle multi-class problems. 
     * @param baseClassifier the binary classifier to extend
     */
    public DDAG(Classifier baseClassifier)
    {
        super(baseClassifier);
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());
        
        //Use a priority que so that we always pick the two lowest value class labels, makes indexing into the oneVsOne array simple
        PriorityQueue<Integer> options = new PriorityQueue<Integer>(predicting.getNumOfCategories());
        for(int i = 0; i < cr.size(); i++)
            options.add(i);
        
        
        CategoricalResults subRes;
        int c1, c2;
        //We will now loop through and repeatedly pick two combinations, and eliminate the loser, until there is one winer
        while(options.size() > 1)
        {
            c1 = options.poll();
            c2 = options.poll();
            
            subRes = oneVone[c1][c2-c1-1].classify(data);
            
            if(subRes.mostLikely() == 0)//c1 wins, c2 no longer a candidate
                options.add(c1);
            else//c2 wins, c1 no onger a candidate
                options.add(c2);
        }
        
        cr.setProb(options.peek(), 1.0);
        
        
        return cr;
    }

    @Override
    public DDAG clone()
    {
        DDAG clone = new DDAG(baseClassifier.clone(), isConcurrentTraining());
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
}
