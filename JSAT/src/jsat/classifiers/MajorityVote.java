
package jsat.classifiers;

import java.util.List;
import java.util.concurrent.ExecutorService;

/**
 * The Majority Vote classifier is a simple ensemble classifier. Given a list of base classifiers, 
 * it will sum the most likely votes from each base classifier and return a result based on the 
 * majority votes. It does not take into account the confidence of the votes. 
 * 
 * @author Edward Raff
 */
public class MajorityVote implements Classifier
{

	private static final long serialVersionUID = 7945429768861275845L;
	private Classifier[] voters;

    /**
     * Creates a new Majority Vote classifier using the given voters. If already trained, the 
     * Majority Vote classifier can be used immediately. The MajorityVote does not make 
     * copies of the given classifiers. <br>
     * <tt>null</tt> values in the array will have no vote. 
     * 
     * @param voters the array of voters to use
     */
    public MajorityVote(Classifier... voters)
    {
        this.voters = voters;
    }
    
    /**
     * Creates a new Majority Vote classifier using the given voters. If already trained, the 
     * Majority Vote classifier can be used immediately. The MajorityVote does not make 
     * copies of the given classifiers. <br>
     * <tt>null</tt> values in the array will have no vote. 
     * 
     * @param voters the list of voters to use
     */
    public MajorityVote(List<Classifier> voters)
    {
        this.voters = voters.toArray(new Classifier[0]);
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults toReturn = null;

        for (Classifier classifier : voters)
            if (classifier != null)
                if (toReturn == null)
                {
                    toReturn = classifier.classify(data);
                    //Instead of allocating a new catResult, reuse the given one. Set the non likely to zero, and most to 1. 
                    for (int i = 0; i < toReturn.size(); i++)
                        if (i != toReturn.mostLikely())
                            toReturn.setProb(i, 0);
                        else
                            toReturn.setProb(i, 1.0);
                }
                else
                {
                    CategoricalResults vote = classifier.classify(data);
                    for (int i = 0; i < toReturn.size(); i++)
                        toReturn.incProb(vote.mostLikely(), 1.0);
                }

        toReturn.normalize();
        return toReturn;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        for(Classifier classifier : voters)
            classifier.trainC(dataSet, threadPool);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        for(Classifier classifier : voters)
            classifier.trainC(dataSet);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public Classifier clone()
    {
        Classifier[] votersClone = new Classifier[this.voters.length];
        for(int i = 0; i < voters.length; i++)
            if(voters[i] != null)
                votersClone[i] = voters[i].clone();
        return new MajorityVote(voters);
    }
    
}
