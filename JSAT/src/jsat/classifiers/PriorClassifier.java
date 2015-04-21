package jsat.classifiers;

import java.util.concurrent.ExecutorService;
import jsat.exceptions.UntrainedModelException;

/**
 * A Naive classifier that simply returns the prior probabilities as the 
 * classification decision. 
 * 
 * @author Edward Raff
 */
public class PriorClassifier implements Classifier
{

	private static final long serialVersionUID = 7763388716880766538L;
	private CategoricalResults cr;

    /**
     * Creates a new PriorClassifeir
     */
    public PriorClassifier()
    {
    }

    /**
     * Creates a new Prior Classifier that is given the results it should be 
     * returning
     * 
     * @param cr the prior probabilities for classification
     */
    public PriorClassifier(CategoricalResults cr)
    {
        this.cr = cr;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(cr == null)
            throw new UntrainedModelException("PriorClassifier has not been trained");
        return cr;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        cr = new CategoricalResults(dataSet.getPredicting().getNumOfCategories());
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            cr.incProb(dataSet.getDataPointCategory(i), dataSet.getDataPoint(i).getWeight());
        cr.normalize();
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }

    @Override
    public Classifier clone()
    {
        PriorClassifier clone = new PriorClassifier();
        if(this.cr != null)
            clone.cr = this.cr.clone();
        return clone;
    }
}
