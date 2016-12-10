package jsat.classifiers.linear;

import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.exceptions.FailedToFitException;
import jsat.linear.Vec;
import static java.lang.Math.*;
import java.util.List;
import jsat.SingleWeightVectorModel;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.lossfunctions.LogisticLoss;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;

/**
 * This is an implementation of Bayesian Binary Regression for L<sub>1</sub> and
 * L<sub>2</sub> regularized logistic regression. This model requires additional
 * memory to perform efficient column wise passes on the data set, assuming the
 * data is sparse. <br><br>
 * BBR uses a Trust Region Newton algorithm that allows convergence to occur in
 * a small number of iterations, but each iteration may be costly.
 * <br><br>
 * See: Genkin, A., Lewis, D. D.,&amp;Madigan, D. (2007). <i>Large-Scale Bayesian
 * Logistic Regression for Text Categorization</i>. Technometrics, 49(3),
 * 291–304. doi:10.1198/004017007000000245
 *
 * @author Edward Raff
 */
public class BBR implements Classifier, Parameterized, SingleWeightVectorModel
{

	private static final long serialVersionUID = 8297213093357011082L;
	//weight vector w is refferd to as beta in the original paper, just replace beta with w
    private Vec w;
    private int maxIterations;
    private double regularization;
    private boolean autoSetRegularization = true;
    private double bias;
    private boolean useBias = true;
    private double tolerance = 0.0005;
    private Prior prior;

    /**
     * Valid priors that control what type of regularization is applied
     */
    public static enum Prior
    {
        /**
         * Laplace prior equivalent to L<sub>1</sub> regularization
         */
        LAPLACE,
        /**
         * Gaussian prior equivalent to L<sub>2</sub> regularization
         */
        GAUSSIAN
    }

    /**
     * Creates a new BBR for L<sub>1</sub> Logistic Regression object that will
     * use the given regularization value.
     *
     * @param regularization the regularization penalty to apply
     * @param maxIterations the maximum number of training iterations to perform
     */
    public BBR(double regularization, int maxIterations)
    {
        this(regularization, maxIterations, Prior.LAPLACE);
    }

    /**
     * Creates a new BBR Logistic Regression object that will use the given
     * regularization value.
     *
     * @param regularization the regularization penalty to apply
     * @param maxIterations the maximum number of training iterations to perform
     * @param prior the prior to apply for regularization
     */
    public BBR(double regularization, int maxIterations, Prior prior)
    {
        setMaxIterations(maxIterations);
        setRegularization(regularization);
        setAutoSetRegularization(false);
        setPrior(prior);
    }

    /**
     * Creates a new BBR for L<sub>1</sub> Logistic Regression object that will
     * attempt to automatically determine the regularization value to use.
     *
     * @param maxIterations the maximum number of training iterations to perform
     */
    public BBR(int maxIterations)
    {
        this(1e-3, maxIterations, Prior.LAPLACE);
    }

    /**
     * Creates a new BBR Logistic Regression object that will attempt to
     * automatically determine the regularization value to use.
     *
     * @param maxIterations the maximum number of training iterations to perform
     * @param prior the prior to apply for regularization
     */
    public BBR(int maxIterations, Prior prior)
    {
        setMaxIterations(maxIterations);
        setRegularization(0.01);
        setAutoSetRegularization(true);
        setPrior(prior);
    }

    /**
     * Copy constructor
     *
     * @param toCopy the object to copy
     */
    protected BBR(BBR toCopy)
    {
        if (toCopy.w != null)
            this.w = toCopy.w.clone();
        this.maxIterations = toCopy.maxIterations;
        this.regularization = toCopy.regularization;
        this.autoSetRegularization = toCopy.autoSetRegularization;
        this.bias = toCopy.bias;
        this.useBias = toCopy.useBias;
        this.tolerance = toCopy.tolerance;
        this.prior = toCopy.prior;
    }

    /**
     * Sets the regularization penalty to use if the algorithm has not been set
     * to choose one automatically.
     *
     * @param regularization sets the positive regularization penalty to use
     */
    public void setRegularization(double regularization)
    {
        if (Double.isNaN(regularization) || Double.isNaN(regularization) || regularization <= 0)
            throw new IllegalArgumentException("Regularization must be positive, not " + regularization);
        this.regularization = regularization;
    }

    /**
     * Returns the regularization penalty used if the auto value is not used
     *
     * @return the regularization penalty used if the auto value is not used
     */
    public double getRegularization()
    {
        return regularization;
    }

    /**
     * Sets whether or not the regularization term will be set automatically by
     * the algorithm, which is done as specified in the original paper. This may
     * choose a very large (and bad) value of the regularization term, and
     * should not be used with smaller data sets. This value is chosen
     * deterministically.
     * <br><br>
     * This value takes precedence over anything set with 
     * {@link #setRegularization(double) }
     *
     * @param autoSetRegularization {@code true} to choose the regularization
     * term automatically, {@code false} to use whatever value was set before.
     */
    public void setAutoSetRegularization(boolean autoSetRegularization)
    {
        this.autoSetRegularization = autoSetRegularization;
    }

    /**
     * Returns whether or not the algorithm will attempt to select the
     * regularization term automatically
     *
     * @return {@code true} if the regularization term is chosen automatically,
     * {@code false} otherwise.
     */
    public boolean isAutoSetRegularization()
    {
        return autoSetRegularization;
    }

    /**
     * Sets the maximum number of iterations allowed before halting the
     * algorithm early.
     *
     * @param maxIterations the maximum number of training iterations
     */
    public void setMaxIterations(int maxIterations)
    {
        this.maxIterations = maxIterations;
    }

    /**
     * Returns the maximum number of iterations allowed
     *
     * @return the maximum number of iterations allowed
     */
    public int getMaxIterations()
    {
        return maxIterations;
    }

    /**
     * Sets the convergence tolerance target. Relative changes that are smaller
     * than the given tolerance will determine convergence.
     * <br><br>
     * The default value used is that suggested in the original paper of 0.0005
     *
     * @param tolerance the positive convergence tolerance goal
     */
    public void setTolerance(double tolerance)
    {
        if (Double.isNaN(tolerance) || Double.isInfinite(tolerance) || tolerance <= 0)
            throw new IllegalArgumentException("Tolerance must be positive, not " + tolerance);
        this.tolerance = tolerance;
    }

    /**
     * Returns the tolerance parameter that controls convergence
     *
     * @return the tolerance parameter that controls convergence
     */
    public double getTolerance()
    {
        return tolerance;
    }

    /**
     * Sets whether or not an implicit bias term should be added to the model.
     *
     * @param useBias {@code true} to add a bias term, {@code false} to exclude
     * the bias term.
     */
    public void setUseBias(boolean useBias)
    {
        this.useBias = useBias;
    }

    /**
     * Returns {@code true} if a bias term is in use, {@code false} otherwise.
     *
     * @return {@code true} if a bias term is in use, {@code false} otherwise.
     */
    public boolean isUseBias()
    {
        return useBias;
    }

    /**
     * Sets the regularizing prior used
     *
     * @param prior the prior to use
     */
    public void setPrior(Prior prior)
    {
        this.prior = prior;
    }

    /**
     * Returns the regularizing prior in use
     *
     * @return the regularizing prior in use
     */
    public Prior getPrior()
    {
        return prior;
    }
    
    /**
     * Returns the weight vector used to compute results via a dot product. <br>
     * Do not modify this value, or you will alter the results returned.
     * @return the learned weight vector for prediction
     */
    public Vec getWeightVec()
    {
        return w;
    }
    
    @Override
    public Vec getRawWeight()
    {
        return w;
    }

    @Override
    public double getBias()
    {
        return bias;
    }

    @Override
    public Vec getRawWeight(int index)
    {
        if(index < 1)
            return getRawWeight();
        else
            throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }

    @Override
    public double getBias(int index)
    {
        if (index < 1)
            return getBias();
        else
            throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }

    @Override
    public int numWeightsVecs()
    {
        return 1;
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        return LogisticLoss.classify(w.dot(data.getNumericalValues()) + bias);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        final int D = dataSet.getNumNumericalVars();
        if (D <= 0)
            throw new FailedToFitException("Data set contains no numeric features");

        final Vec[] columnMajor = dataSet.getNumericColumns();
        w = new DenseVector(D);

        double[] delta = new double[useBias ? D + 1 : D];
        Arrays.fill(delta, 1.0);

        final int N = dataSet.getSampleSize();
        double[] r = new double[N];
        double[] y = new double[N];
        for (int i = 0; i < N; i++)
            y[i] = dataSet.getDataPointCategory(i) * 2 - 1;
        final double lambda;
        if (autoSetRegularization)
        {
            //see equation (21)
            double normSqrdSum = 0;
            for (int i = 0; i < N; i++)
                normSqrdSum += pow(dataSet.getDataPoint(i).getNumericalValues().pNorm(2), 2);

            double sigma = D * N / normSqrdSum;

            //no regularization less than precision 
            if (prior == Prior.LAPLACE)
                lambda = max(sqrt(2) / sigma, 1e-15);
            else
                lambda = max(sigma * sigma, 1e-15);
        }
        else
            lambda = regularization;

        double[] r_change = new double[N];

        for (int iter = 0; iter < maxIterations; iter++)
        {
            for (int j = 0; j < D; j++)
            {
                double delta_vj = 0;
                final double w_jOrig = w.get(j);
                if (prior == Prior.LAPLACE)
                {
                    //Algo 2 in the paper, computing delta_vj
                    if (w_jOrig == 0)
                    {
                        //(try positive direction)
                        delta_vj = tenativeUpdate(columnMajor, j, w_jOrig, y, r, lambda, 1.0, delta);
                        if (delta_vj <= 0)//(positive direction failed)
                        {
                            //(try negative direction)
                            delta_vj = tenativeUpdate(columnMajor, j, w_jOrig, y, r, lambda, -1.0, delta);
                            if (delta_vj >= 0)//(negative direction failed)
                                delta_vj = 0;
                        }
                    }
                    else
                    {
                        final double sign = signum(w_jOrig);
                        delta_vj = tenativeUpdate(columnMajor, j, w_jOrig, y, r, lambda, sign, delta);
                        if (sign * (w_jOrig + delta_vj) < 0)//(cross over 0)
                            delta_vj = -w_jOrig;//Done soe that w_j+-w_j = 0
                    }
                }
                else//Guassian prior
                {
                    delta_vj = tenativeUpdate(columnMajor, j, w_jOrig, y, r, lambda, 0, delta);
                }

                double delta_wj = min(max(delta_vj, -delta[j]), delta[j]);//(limit step to trust region)
                for (IndexValue iv : columnMajor[j])
                {
                    final int i = iv.getIndex();
                    final double change = delta_wj * iv.getValue() * y[i];
                    r[i] += change;
                    r_change[i] += change;
                }

                double newW_j = w_jOrig + delta_wj;
                //make tiny value zero
                if (abs(newW_j) < 1e-15)//Less than precions? its zero
                    newW_j = 0;
                w.set(j, newW_j);

                delta[j] = max(2 * abs(delta_wj), delta[j] / 2); //(update size of trust region)
            }

            if (useBias)//update the bias term
            {
                double delta_vj;
                //Algo 2 in the paper, computing delta_vj
                if (bias == 0)
                {
                    //(try positive direction)
                    delta_vj = tenativeUpdate(null, D, bias, y, r, lambda, 1.0, delta);
                    if (delta_vj <= 0)//(positive direction failed)
                    {
                        //(try negative direction)
                        delta_vj = tenativeUpdate(null, D, bias, y, r, lambda, -1.0, delta);
                        if (delta_vj >= 0)//(negative direction failed)
                            delta_vj = 0;
                    }
                }
                else
                {
                    final double sign = signum(bias);
                    delta_vj = tenativeUpdate(null, D, bias, y, r, lambda, sign, delta);
                    if (sign * (bias + delta_vj) < 0)//(cross over 0)
                        delta_vj = -bias;//Done soe that w_j+-w_j = 0
                }

                double delta_wj = min(max(delta_vj, -delta[D]), delta[D]);//(limit step to trust region)
                for (int i = 0; i < N; i++)
                {
                    final double change = delta_wj * y[i];
                    r[i] += change;
                    r_change[i] += change;
                }

                double newW_j = bias + delta_wj;
                //make tiny value zero
                if (abs(newW_j) < 1e-15)//Less than precions? its zero
                    newW_j = 0;

                bias = newW_j;

                delta[D] = max(2 * abs(delta_wj), delta[D] / 2); //(update size of trust region)
            }

            //Check for convergence
            double changeSum = 0, rSum = 0;
            for (int i = 0; i < N; i++)
            {
                changeSum += abs(r_change[i]);
                rSum += abs(r[i]);
            }

            if (changeSum / (1 + rSum) <= tolerance)//converged!
                break;
            Arrays.fill(r_change, 0.0);//resent changes for the next iteration
        }
    }

    private static double F(double r, double delta)
    {
        if (abs(r) <= delta)
            return 0.25;
        else
            return 1 / (2 + exp(abs(r) - delta) + exp(delta - abs(r)));
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public BBR clone()
    {
        return new BBR(this);
    }

    /**
     * Gets the tentative update &delta;<sub>vj</sub>
     *
     * @param columnMajor the column major vector array. May be null if using
     * the implicit bias term
     * @param j the column to work on
     * @param w_j the value of the coefficient, used only under Gaussian prior
     * @param y the array of label values
     * @param r the array of r values
     * @param lambda the regularization value to apply
     * @param s the update direction (should be +1 or -1). Used only under
     * Laplace prior
     * @param delta the array of delta values
     * @return the tentative update value
     */
    private double tenativeUpdate(final Vec[] columnMajor, final int j, final double w_j, final double[] y, final double[] r, final double lambda, final double s, final double[] delta)
    {
        double numer = 0, denom = 0;
        if (columnMajor != null)
        {
            Vec col_j = columnMajor[j];
            if (col_j.nnz() == 0)
                return 0;
            for (IndexValue iv : col_j)
            {
                final double x_ij = iv.getValue();
                final int i = iv.getIndex();
                numer += x_ij * y[i] / (1 + exp(r[i]));
                denom += x_ij * x_ij * F(r[i], delta[j] * abs(x_ij));
                if (prior == Prior.LAPLACE)
                    numer -= lambda * s;
                else
                {
                    numer -= w_j / lambda;
                    denom += 1 / lambda;
                }
            }
        }
        else//bias term, all x_ij = 1
            for (int i = 0; i < y.length; i++)
            {
                numer += y[i] / (1 + exp(r[i])) - lambda * s;
                denom += F(r[i], delta[j]);
            }

        return numer / denom;
    }

    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
}
