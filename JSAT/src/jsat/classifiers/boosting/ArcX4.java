package jsat.classifiers.boosting;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.parameters.Parameterized;
import jsat.utils.concurrent.ParallelUtils;

/**
 * Arc-x4 is a ensemble-classifier that performs re-weighting of the data points 
 * based on the total number of errors that have occurred for the data point.
 * <br><br>
 * See: Breiman, L. (1998). <i>Arcing Classifiers</i>. The Annals of Statistics,
 * 26(3), 801–824.
 * 
 * @author Edward Raff
 */
public class ArcX4 implements Classifier, Parameterized
{

    private static final long serialVersionUID = 3831448932874147550L;
    private Classifier weakLearner;
    private int iterations;

    private double coef = 1;
    private double expo = 4;
    
    private CategoricalData predicing;
    private Classifier[] hypoths;
    
    /**
     * Creates a new Arc-X4 classifier 
     * 
     * @param weakLearner the weak learner to use
     * @param iterations the number of iterations to perform
     */
    public ArcX4(Classifier weakLearner, int iterations)
    {
        setWeakLearner(weakLearner);
        setIterations(iterations);
    }

    /**
     * Sets the weak learner used at each iteration of learning
     * @param weakLearner the weak learner to use
     */
    public void setWeakLearner(Classifier weakLearner)
    {
        if(!weakLearner.supportsWeightedData())
            throw new RuntimeException("Weak learners must support weighted data samples");
        this.weakLearner = weakLearner;
    }

    /**
     * Returns the weak learner used
     * @return the weak learner used
     */
    public Classifier getWeakLearner()
    {
        return weakLearner;
    }

    /**
     * Sets the number of iterations to perform
     * @param iterations the number of iterations to do
     */
    public void setIterations(int iterations)
    {
        this.iterations = iterations;
    }

    /**
     * Returns the number of iterations to learn
     * @return the number of iterations to learn
     */
    public int getIterations()
    {
        return iterations;
    }

    /**
     * Weights are updated as 1+coef*errors<sup>expo</sup>. This sets the 
     * coefficient used to update the errors
     * 
     * @param coef the multiplicative factor on the errors in weight construction
     */
    public void setCoefficient(double coef)
    {
        if(coef <= 0 || Double.isInfinite(coef) || Double.isNaN(coef))
            throw new ArithmeticException("The coefficient must be a positive constant");
        this.coef = coef;
    }

    /**
     * Returns the coefficient use when re-weighting
     * @return the coefficient use when re-weighting
     */
    public double getCoefficient()
    {
        return coef;
    }

    /**
     * Weights are updated as 1+coef*errors<sup>expo</sup>. This sets the 
     * exponent used to update the errors
     * @param expo the exponent to use
     */
    public void setExponent(double expo)
    {
        if(expo <= 0 || Double.isInfinite(expo) || Double.isNaN(expo))
            throw new ArithmeticException("The exponent must be a positive constant");
        this.expo = expo;
    }

    /**
     * Returns the exponent used when re-weighting 
     * @return the exponent used when re-weighting 
     */
    public double getExponent()
    {
        return expo;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(predicing.getNumOfCategories());
        
        for(Classifier hypoth : hypoths)
            cr.incProb(hypoth.classify(data).mostLikely(), 1.0);
        
        cr.normalize();
        
        return cr;
    }
   
    @Override
    public void train(ClassificationDataSet dataSet, boolean parallel)
    {
        //Create a low memory clone that only has different dataPoint Objects to save space
        ClassificationDataSet cds = dataSet.shallowClone();
        
        //Everyone starts with no errors
        int[] errors = new int[cds.size()];
        
        hypoths = new Classifier[iterations];
        for(int t = 0; t < hypoths.length; t++)
        {
            for(int i = 0; i < cds.size(); i++)
                cds.setWeight(i, 1+coef*Math.pow(errors[i], expo));
            
            Classifier hypoth = weakLearner.clone();
            
            hypoth.train(cds, parallel);
            
            hypoths[t] = hypoth;
            ParallelUtils.run(parallel, errors.length, (start, end) ->
            {
                for(int i = start; i < end; i++)
                    if(hypoth.classify(cds.getDataPoint(i)).mostLikely() != cds.getDataPointCategory(i))
                        errors[i]++;
            });
        }
        
        this.predicing = cds.getPredicting();
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public ArcX4 clone()
    {
        ArcX4 clone = new ArcX4(weakLearner.clone(), iterations);
        
        clone.coef = this.coef;
        clone.expo = this.expo;
        
        if(this.predicing != null)
            clone.predicing = this.predicing.clone();
        if(this.hypoths != null)
        {
            clone.hypoths = new Classifier[this.hypoths.length];
            for(int i = 0; i < clone.hypoths.length; i++)
                clone.hypoths[i] = this.hypoths[i].clone();
        }
        
        return clone;
    }
}
