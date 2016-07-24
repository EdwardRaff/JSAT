
package jsat.datatransform;

import java.util.Arrays;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.math.IndexFunction;
import jsat.math.OnLineStatistics;
import jsat.utils.DoubleList;

/**
 * This transform applies a shifted Box-Cox transform for several fixed values 
 * of &lambda;, and selects the one that provides the greatest reduction in the 
 * skewness of the distribution. This is done in an attempt to make the 
 * individual features appear more normal. The shifted values are done to
 * preserve zeros and keep sparse inputs sparse. This is done with two passes
 * through the data set, but requires only O(D #&lambda; values) memory. 
 * <br><br>
 * The default values of &lambda; are -1, -1/2, 0, 1/2, 1. When using 
 * negative &lambda; values all zeros are skipped and left as zeros.  &lambda; =
 * 1 is an implicit value that is always included regardless of the input, as it
 * is equivalent to leaving the data unchanged when preserving zero values. 
 * The stated default values include the <i>log(x+1)</i> and <i>sqrt(x)</i> 
 * transforms that are commonly used for deskewing as special cases. 
 * <br><br>
 * Skewness can be calculated by including zero, but by default ignores them as 
 * "not-present" values.
 * 
 * @author Edward Raff
 */
public class AutoDeskewTransform implements InPlaceTransform
{
    private static final long serialVersionUID = -4894242802345656448L;
    private double[] finalLambdas;
    private double[] mins;
    private final IndexFunction transform = new IndexFunction()
    {
        private static final long serialVersionUID = -404316813485246422L;

        @Override
        public double indexFunc(double value, int index)
        {
            if(index < 0)
                return 0.0;
            return transform(value, finalLambdas[index], mins[index]);
        }
    };
    private static final DoubleList defaultList = new DoubleList(7);

    static
    {
        defaultList.add(-1.0);
        defaultList.add(-0.5);
        defaultList.add(0.0);
        defaultList.add(0.5);
        defaultList.add(1.0);
    }

    
    private List<Double> lambdas;
    private boolean ignorZeros;
    
    /**
     * Creates a new AutoDeskew transform
     *
     */
    public AutoDeskewTransform()
    {
        this(true, defaultList);
    }
    
    /**
     * Creates a new AutoDeskew transform
     *
     * @param lambdas the list of lambda values to evaluate
     */
    public AutoDeskewTransform(final double... lambdas)
    {
        this(true, DoubleList.view(lambdas, lambdas.length));
    }

    /**
     * Creates a new AutoDeskew transform
     *
     * @param lambdas the list of lambda values to evaluate
     */
    public AutoDeskewTransform(final List<Double> lambdas)
    {
        this(true, lambdas);
    }
    
    /**
     * Creates a new AutoDeskew transform
     *
     * @param ignorZeros {@code true} to ignore zero values when calculating the
     * skewness, {@code false} to include them.
     * @param lambdas the list of lambda values to evaluate
     */
    public AutoDeskewTransform(boolean ignorZeros, final List<Double> lambdas)
    {
        this.ignorZeros = ignorZeros;
        this.lambdas = lambdas;
    }
    
    /**
     * Creates a new deskewing object from the given data set
     *
     * @param dataSet the data set to deskew
     */
    public AutoDeskewTransform(DataSet dataSet)
    {
        this(dataSet, defaultList);
    }

    /**
     * Creates a new deskewing object from the given data set
     *
     * @param dataSet the data set to deskew
     * @param lambdas the list of lambda values to evaluate
     */
    public AutoDeskewTransform(DataSet dataSet, final List<Double> lambdas)
    {
        this(dataSet, true, lambdas);
    }

    /**
     * Creates a new deskewing object from the given data set
     *
     * @param dataSet the data set to deskew
     * @param ignorZeros {@code true} to ignore zero values when calculating the
     * skewness, {@code false} to include them.
     * @param lambdas the list of lambda values to evaluate
     */
    public AutoDeskewTransform(DataSet dataSet, boolean ignorZeros, final List<Double> lambdas)
    {
        this(ignorZeros, lambdas);
        fit(dataSet);
    }

    @Override
    public void fit(DataSet dataSet)
    {
        //going to try leaving things alone nomatter what
        if (!lambdas.contains(1.0))
            lambdas.add(1.0);

        OnLineStatistics[][] stats = new OnLineStatistics[lambdas.size()][dataSet.getNumNumericalVars()];
        for (int i = 0; i < stats.length; i++)
            for (int j = 0; j < stats[i].length; j++)
                stats[i][j] = new OnLineStatistics();
        mins = new double[dataSet.getNumNumericalVars()];
        Arrays.fill(mins, Double.POSITIVE_INFINITY);

        boolean containsSparseVecs = false;
        //First pass, get min/max values
        for (int i = 0; i < dataSet.getSampleSize(); i++)
        {
            Vec x = dataSet.getDataPoint(i).getNumericalValues();
            if (x.isSparse())
                containsSparseVecs = true;
            for (IndexValue iv : x)
            {
                final int indx = iv.getIndex();
                final double val = iv.getValue();

                mins[indx] = Math.min(val, mins[indx]);
            }
        }
        if (containsSparseVecs)
            for (int i = 0; i < mins.length; i++)//done b/c we only iterated the non-zeros
                mins[i] = Math.min(0, mins[i]);

        //Second pass, find the best skew transform
        for (int i = 0; i < dataSet.getSampleSize(); i++)
        {
            Vec x = dataSet.getDataPoint(i).getNumericalValues();
            double weight = dataSet.getDataPoint(i).getWeight();

            int lastIndx = -1;
            for (IndexValue iv : x)
            {
                int indx = iv.getIndex();
                double val = iv.getValue();
                updateStats(lambdas, stats, indx, val, mins, weight);

                if (!ignorZeros)//we have to do this here instead of bulk insert at the end b/c of different weight value combinations
                    for (int prevIndx = lastIndx + 1; prevIndx < indx; prevIndx++)
                        updateStats(lambdas, stats, prevIndx, 0.0, mins, weight);

                lastIndx = indx;
            }

            //Catch trailing zero values
            if (!ignorZeros)//we have to do this here instead of bulk insert at the end b/c of different weight value combinations
                for (int prevIndx = lastIndx + 1; prevIndx < mins.length; prevIndx++)
                    updateStats(lambdas, stats, prevIndx, 0.0, mins, weight);
        }

        //Finish by figureing out which did best
        finalLambdas = new double[mins.length];
        int lambdaOneIndex = lambdas.indexOf(1.0);
        for (int d = 0; d < finalLambdas.length; d++)
        {
            double minSkew = Double.POSITIVE_INFINITY;
            double bestLambda = 1;//done this way incase a NaN slips in, we will leave data unchanged

            for (int k = 0; k < lambdas.size(); k++)
            {
                double skew = Math.abs(stats[k][d].getSkewness());
                if (skew < minSkew)
                {
                    minSkew = skew;
                    bestLambda = lambdas.get(k);
                }
            }

            double origSkew = Math.abs(stats[lambdaOneIndex][d].getSkewness());

            if (origSkew > minSkew * 1.05)//only change if there is a reasonable improvment
                finalLambdas[d] = bestLambda;
            else
                finalLambdas[d] = 1.0;
        }
    }

    /**
     * Copy constructor
     *
     * @param toCopy the object to copy
     */
    protected AutoDeskewTransform(AutoDeskewTransform toCopy)
    {
        this.finalLambdas = Arrays.copyOf(toCopy.finalLambdas, toCopy.finalLambdas.length);
        this.mins = Arrays.copyOf(toCopy.mins, toCopy.mins.length);
    }

    private static double transform(final double val, final double lambda, final double min)
    {
        if (val == 0)
            return 0;
        //special cases
        if (lambda == 2)
        {
            return val * val;
        }
        if (lambda == 1)
        {
            return val;
        }
        else if (lambda == 0.5)
        {
            return Math.sqrt(val - min);
        }
        else if (lambda == 0)
        {
            return Math.log(val + 1 - min);//log(1) = 0
        }
        else if (lambda == -0.5)
        {
            return 1 / (Math.sqrt(val - min));
        }
        else if (lambda == -1)
        {
            return 1 / val;
        }
        else if (lambda == -2)
        {
            return 1 / (val * val);
        }
        else
        {
            //commented out case handled at top
            //if(lambda < 0 && val == 0)
            //    return 0;//should be Inf, but we want to preserve sparsity
            return Math.pow(val, lambda) / lambda;
        }
    }

    @Override
    public DataPoint transform(DataPoint dp)
    {
        DataPoint newDP = dp.clone();
        mutableTransform(newDP);
        return newDP;
    }
    
    @Override
    public void mutableTransform(DataPoint dp)
    {
        dp.getNumericalValues().applyIndexFunction(transform);
    }

    @Override
    public AutoDeskewTransform clone()
    {
        return new AutoDeskewTransform(this);
    }

    /**
     * Updates the online stats for each value of lambda
     *
     * @param lambdas the list of lambda values
     * @param stats the array of statistics trackers
     * @param indx the feature index to add to
     * @param val the value at the given feature index
     * @param mins the minimum value array
     * @param weight the weight to the given update
     */
    private void updateStats(final List<Double> lambdas, OnLineStatistics[][] stats, int indx, double val, double[] mins, double weight)
    {
        for (int k = 0; k < lambdas.size(); k++)
            stats[k][indx].add(transform(val, lambdas.get(k), mins[indx]), weight);
    }

    @Override
    public boolean mutatesNominal()
    {
        return false;
    }
}
