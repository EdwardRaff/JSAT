package jsat.classifiers.bayesian;

import java.util.Arrays;
import jsat.classifiers.*;
import jsat.exceptions.FailedToFitException;

/**
 * One-Dependence Estimators (ODE) is an extension of Naive Bayes that, instead 
 * of assuming all features are independent, assumes all features are dependent 
 * on one other feature besides the target class. Because of this extra 
 * dependence requirement, the implementation only allows for categorical 
 * features. 
 * <br>
 * This class is primarily for use by {@link AODE}
 * <br><br> 
 * See: Webb, G., &amp; Boughton, J. (2005). <i>Not so naive bayes: Aggregating 
 * one-dependence estimators</i>. Machine Learning, 1–24. Retrieved from 
 * <a href="http://www.springerlink.com/index/U8W306673M1P866K.pdf">here</a>
 * 
 * @author Edward Raff
 */
public class ODE extends BaseUpdateableClassifier 
{

	private static final long serialVersionUID = -7732070257669428977L;
	/**
     * The attribute we will be dependent on
     */
    protected int dependent;
    /**
     * The number of possible values in the target class
     */
    protected int predTargets;
    /**
     * The number of possible values for the dependent variable
     */
    protected int depTargets;
    /**
     * First index is the number of target values <br>
     * 2nd index is the number of values for the dependent variable <br>
     * 3rd is the number of categorical variables, including the dependent one <br>
     * 4th is the count for the variable value <br>
     */
    protected double[][][][] counts;
    /**
     * The prior probability of each combination of target and dependent variable
     */
    protected double[][] priors;
    protected double priorSum;

    /**
     * Creates a new ODE classifier 
     * @param dependent the categorical feature to be dependent of
     */
    public ODE(int dependent)
    {
        this.dependent = dependent;
    }
    
    /**
     * Copy constructor
     * @param toClone the ODE to copy
     */
    protected ODE(ODE toClone)
    {
        this(toClone.dependent);
        this.predTargets = toClone.predTargets;
        this.depTargets = toClone.depTargets;
        if (toClone.counts != null)
        {
            this.counts = new double[toClone.counts.length][][][];
            for (int i = 0; i < this.counts.length; i++)
            {
                this.counts[i] = new double[i][][];
                for (int j = 0; j < this.counts[i].length; j++)
                {
                    this.counts[i][j] = new double[toClone.counts[i][j].length][];
                    for (int z = 0; z < this.counts[i][j].length; z++)
                    {
                        this.counts[i][j][z] =
                                Arrays.copyOf(toClone.counts[i][j][z],
                                toClone.counts[i][j][z].length);
                    }
                }
            }
            
            this.priors = new double[toClone.priors.length][];
            for(int i = 0; i < this.priors.length; i++)
                this.priors[i] = Arrays.copyOf(toClone.priors[i], toClone.priors[i].length);
            this.priorSum = toClone.priorSum;
        }
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(predTargets);
        
        int[] catVals = data.getCategoricalValues();
        for (int c = 0; c < predTargets; c++)
        {
            double logProb = getLogPrb(catVals, c);
            
            cr.setProb(c, Math.exp(logProb)); 
        }
        cr.normalize();

        return cr;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }

    @Override
    public ODE clone()
    {
        return new ODE(this);
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if(categoricalAttributes.length < 1)
            throw new FailedToFitException("At least 2 categorical varaibles are needed for ODE");
        CategoricalData[] catData = categoricalAttributes;
        predTargets = predicting.getNumOfCategories();
        depTargets = catData[dependent].getNumOfCategories();
        counts = new double[predTargets][depTargets][catData.length][];
        for(int i = 0; i < counts.length; i++)
            for(int j = 0; j < counts[i].length; j++)
                for(int z = 0; z < counts[i][j].length; z++)
                {
                    counts[i][j][z] = new double[catData[z].getNumOfCategories()];
                    Arrays.fill(counts[i][j][z], 1.0);//Fill will laplace
                }
        
        priors = new double[predTargets][depTargets];
        for(int i = 0; i < priors.length; i++)
        {
            Arrays.fill(priors[i], 1.0);
            priorSum += priors[i].length;
        }
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        int[] catVals = dataPoint.getCategoricalValues();
        double weight = dataPoint.getWeight();
        for (int j = 0; j < catVals.length; j++)
            if (j == dependent)
                continue;
            else
                counts[targetClass][catVals[dependent]][j][catVals[j]] += weight;
        priors[targetClass][catVals[dependent]] += weight;
        priorSum += weight;
    }

    /**
     * 
     * @param catVals the catigorical values for a data point
     * @param c the target value to get the probability of
     * @return the non normalized log probability of the data point belonging to
     * the target class <tt>c</tt>
     */
    protected double getLogPrb(int[] catVals, int c)
    {
        double logProb = 0.0;
        int xi = catVals[dependent];
        for (int j = 0; j < catVals.length; j++)
        {
            if(j == dependent)
                continue;
            double sum = 0;
            for(int z = 0; z < counts[c][xi][j].length; z++)
                sum += counts[c][xi][j][z];
            logProb += Math.log(counts[c][xi][j][catVals[j]]/sum);
        }
        
        return logProb + Math.log(priors[c][xi]/priorSum);
    }
}
